#!/usr/bin/env python3
# ==============================================================================
# WEDA LLM Suite — RAG runtime (chromadb + llama-server /v1/embeddings)
# ==============================================================================
# App-driven: prompt, collection, top_k, docs_dir and endpoints all come from
# the application block in config/llm_config.yaml. Invoked via app.sh:
#
#   ./app.sh docs-qa                 # REPL
#   ./app.sh docs-qa ingest [dir]    # chunk + embed + store
#   ./app.sh docs-qa ask "question"
import json
# readline wires line editing (backspace, arrows, history) into input();
# importing it is the whole integration
try:
    import readline  # noqa: F401
except ImportError:
    pass
import os
import re
import sys
import threading
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import AppConfig, load_app
from agent_demo import system_prompt_for

# Compose recreation wipes container /tmp -- persist the store on a volume.
STORE = os.environ.get("RAG_STORE_DIR", "/wise-edge/app/rag_store")
# Single conventioned mount root (compose: ${DOCS_DIR} -> /wise-edge/app/docs).
# App docs_dir is RELATIVE to it, so new corpora never require a compose edit.
DOCS_ROOT = os.environ.get("DOCS_ROOT", "/wise-edge/app/docs")


def _post(base, path, payload, timeout=180):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        from agent_demo import upstream_error
        sys.exit(f"[error] {upstream_error(e)}")
    except (urllib.error.URLError, OSError) as e:
        sys.exit(f"[error] {base}{path}: {getattr(e, 'reason', e)} -- "
                 "server down or too slow; ./build.sh --status")


EMBED_BATCH = 64


def embed(texts, base):
    # batched: one request per EMBED_BATCH chunks keeps a large file inside
    # the request timeout and the server's batch limits
    out = []
    for i in range(0, len(texts), EMBED_BATCH):
        data = _post(base, "/embeddings", {"model": "embed", "input": texts[i:i + EMBED_BATCH]})
        vecs = [d["embedding"] for d in sorted(data["data"], key=lambda x: x["index"])]
        # llama-server may nest one level depending on pooling shape
        out += [v[0] if v and isinstance(v[0], list) else v for v in vecs]
    return out


_chroma_lock = threading.Lock()
_chroma_client = {}

def collection(name):
    import chromadb
    with _chroma_lock:
        if "client" not in _chroma_client:
            _chroma_client["client"] = chromadb.PersistentClient(path=STORE)
        return _chroma_client["client"].get_or_create_collection(name)


# ---- document readers: one function per format, all standard library except
# PDF (pypdf ships in the image). Each returns plain text or None (unreadable).
TEXT_EXT = {".txt", ".md", ".log", ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml", ".rst", ".ini", ".conf"}
HTML_EXT = {".html", ".htm"}
OFFICE_EXT = {".docx", ".pptx", ".xlsx"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_EXT = TEXT_EXT | HTML_EXT | OFFICE_EXT | IMAGE_EXT | {".pdf"}
_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
       "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
       "s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _read_text(fp):
    with open(fp, encoding="utf-8", errors="replace") as f:
        return f.read()


def _read_html(fp):
    from html.parser import HTMLParser

    class _Strip(HTMLParser):
        def __init__(self):
            super().__init__()
            self.out, self._skip = [], 0

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style"):
                self._skip += 1
            elif tag in ("p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"):
                self.out.append("\n")

        def handle_endtag(self, tag):
            if tag in ("script", "style") and self._skip:
                self._skip -= 1

        def handle_data(self, data):
            if not self._skip:
                self.out.append(data)

    sp = _Strip()
    sp.feed(_read_text(fp))
    return "".join(sp.out)


def _ocr_available():
    import shutil
    return bool(shutil.which("tesseract"))


def _ocr_image(path):
    import subprocess
    r = subprocess.run(["tesseract", path, "-", "--psm", "3"],
                       capture_output=True, text=True, timeout=600)
    return r.stdout.strip()


def _read_image(fp):
    if not _ocr_available():
        print(f"[rag] skip {os.path.basename(fp)}: OCR (tesseract) not available in this image")
        return None
    return _ocr_image(fp)


def _ocr_pdf(fp):
    """Render each page with pdftoppm and OCR it -- for PDFs without a text
    layer (scans). Slow on an edge CPU: seconds per page."""
    import glob
    import shutil
    import subprocess
    import tempfile
    if not (_ocr_available() and shutil.which("pdftoppm")):
        return None
    pages = []
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["pdftoppm", "-r", "200", "-png", fp, d + "/p"],
                       check=True, capture_output=True, timeout=1800)
        for i, img in enumerate(sorted(glob.glob(d + "/p-*.png")), 1):
            text = _ocr_image(img)
            if text:
                pages.append(f"[page {i}]\n{text}")
    return "\n\n".join(pages)


def _read_pdf(fp):
    try:
        from pypdf import PdfReader
    except ImportError:
        print(f"[rag] skip {os.path.basename(fp)}: pypdf not available in this image")
        return None
    reader = PdfReader(fp)
    pages = []
    for i, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"[page {i}]\n{text}")
    if pages:
        return _doc_header(fp, reader.metadata, len(reader.pages)) + "\n\n".join(pages)
    # no text layer: a scanned document -> OCR the rendered pages
    print(f"[rag] {os.path.basename(fp)}: no text layer, running OCR (seconds per page)")
    return _ocr_pdf(fp)


def _doc_header(fp, meta, pages=None):
    """A metadata line in front of the text so questions about the document
    itself (author, title, date) retrieve it -- those facts sit on a title
    page that similarity search rarely selects on its own."""
    fields = [("title", "title"), ("author", "author"), ("subject", "subject"),
              ("created", "creation_date")]
    parts = [f"file: {os.path.basename(fp)}"]
    for label, attr in fields:
        try:
            v = getattr(meta, attr, None) if meta is not None else None
        except Exception:
            v = None
        if v:
            parts.append(f"{label}: {str(v).strip()}")
    if pages:
        parts.append(f"pages: {pages}")
    return "[document] " + " | ".join(parts) + "\n\n"


def _xml_texts(zf, member, tag):
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(zf.read(member))
    except (KeyError, ET.ParseError):
        return []
    return [t.text or "" for t in root.iter(tag)]


class _Core:
    """Title/author/subject/created from an Office package's docProps/core.xml."""
    def __init__(self, zf):
        import xml.etree.ElementTree as ET
        self.title = self.author = self.subject = self.creation_date = None
        try:
            root = ET.fromstring(zf.read("docProps/core.xml"))
        except Exception:
            return
        dc, cp = "{http://purl.org/dc/elements/1.1/}", "{http://purl.org/dc/terms/}"
        self.title = (root.findtext(dc + "title") or "").strip() or None
        self.author = (root.findtext(dc + "creator") or "").strip() or None
        self.subject = (root.findtext(dc + "subject") or "").strip() or None
        self.creation_date = (root.findtext(cp + "created") or "").strip() or None


def _read_docx(fp):
    import zipfile
    import xml.etree.ElementTree as ET
    with zipfile.ZipFile(fp) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))
        core = _Core(zf)
    paras = []
    for para in root.iter(f"{{{_NS['w']}}}p"):
        line = "".join(t.text or "" for t in para.iter(f"{{{_NS['w']}}}t")).strip()
        if line:
            paras.append(line)
    return _doc_header(fp, core) + "\n".join(paras)


def _read_pptx(fp):
    import zipfile
    out = []
    with zipfile.ZipFile(fp) as zf:
        slides = sorted((m for m in zf.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", m)),
                        key=lambda m: int(re.search(r"(\d+)", m.rsplit("/", 1)[1]).group(1)))
        for n, member in enumerate(slides, 1):
            texts = [t.strip() for t in _xml_texts(zf, member, f"{{{_NS['a']}}}t") if t.strip()]
            if texts:
                out.append(f"[slide {n}]\n" + "\n".join(texts))
    return "\n\n".join(out)


def _read_xlsx(fp):
    import zipfile
    import xml.etree.ElementTree as ET
    out = []
    with zipfile.ZipFile(fp) as zf:
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.iter(f"{{{_NS['s']}}}si"):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{_NS['s']}}}t")))
        sheets = sorted(m for m in zf.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", m))
        for n, member in enumerate(sheets, 1):
            root = ET.fromstring(zf.read(member))
            rows = []
            for row in root.iter(f"{{{_NS['s']}}}row"):
                cells = []
                for c in row.iter(f"{{{_NS['s']}}}c"):
                    v = c.find(f"{{{_NS['s']}}}v")
                    if v is None or v.text is None:
                        continue
                    cells.append(shared[int(v.text)] if c.get("t") == "s" and v.text.isdigit()
                                 and int(v.text) < len(shared) else v.text)
                if cells:
                    rows.append(", ".join(cells))
            if rows:
                out.append(f"[sheet {n}]\n" + "\n".join(rows))
    return "\n\n".join(out)


def read_document(fp):
    """Plain text of a supported file, or None when the format is unsupported
    or the file cannot be parsed (the caller reports and skips it)."""
    ext = os.path.splitext(fp)[1].lower()
    try:
        if ext in TEXT_EXT:
            return _read_text(fp)
        if ext in HTML_EXT:
            return _read_html(fp)
        if ext == ".pdf":
            return _read_pdf(fp)
        if ext == ".docx":
            return _read_docx(fp)
        if ext == ".pptx":
            return _read_pptx(fp)
        if ext == ".xlsx":
            return _read_xlsx(fp)
        if ext in IMAGE_EXT:
            return _read_image(fp)
    except Exception as e:
        print(f"[rag] skip {os.path.basename(fp)}: cannot parse ({e})")
    return None


def chunk(text, size, overlap):
    step = max(1, size - overlap)
    out, i = [], 0
    while i < len(text):
        piece = text[i:i + size].strip()
        if piece:
            out.append(piece)
        i += step
    return out


_SECTION = re.compile(r"^(#{1,6}\s+.+|\[(?:page|slide|sheet) \d+\]|\[document\] .+)$")


def chunk_structured(text, size, overlap):
    """Split at structure first -- Markdown headings and the [page N] /
    [slide N] / [sheet N] / [document] markers the readers emit -- then by
    size within a section. Every chunk carries its section label, so a
    snippet retrieved on its own still says where it comes from."""
    sections, label, buf = [], "", []
    for line in text.splitlines():
        if _SECTION.match(line.strip()):
            if buf:
                sections.append((label, "\n".join(buf)))
            label, buf = line.strip(), []
        else:
            buf.append(line)
    if buf or label:
        sections.append((label, "\n".join(buf)))
    out = []
    for label, body in sections:
        body = body.strip()
        if not body and not label:
            continue
        head = (label + "\n") if label else ""
        pieces = chunk(body, max(64, size - len(head)), overlap) or [""]
        out += [s for piece in pieces if (s := (head + piece).strip())]
    return out


_bm25_cache = {}
_bm25_lock = threading.Lock()
_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")


def _tokens(text):
    return _TOKEN.findall(text.lower())


def _bm25_index(col):
    """BM25 over every chunk of a collection, rebuilt when its size changes.
    The ids come from the same get() as the documents, so rank positions,
    texts and ids are aligned by construction -- no second fetch at query
    time and no cross-call ordering assumption."""
    from rank_bm25 import BM25Okapi
    key = (col.name, col.count())
    with _bm25_lock:
        hit = _bm25_cache.get(col.name)
        if hit and hit[0] == key:
            return hit[1]
        got = col.get(include=["documents", "metadatas"])
        docs, metas, ids = got["documents"], got["metadatas"], got["ids"]
        index = (BM25Okapi([_tokens(d) for d in docs]), docs, metas, ids)
        _bm25_cache[col.name] = (key, index)
        return index


def _rrf(rankings, k=60):
    """Reciprocal-rank fusion of several ranked id lists -> ids by fused score."""
    score = {}
    for ranked in rankings:
        for rank, doc_id in enumerate(ranked):
            score[doc_id] = score.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(score, key=score.get, reverse=True)


_reranker = {}


def _cross_encoder():
    """The configured cross-encoder, loaded once from the shared cache; never
    downloaded here (download_models.py --rerank)."""
    if "model" in _reranker:
        return _reranker["model"]
    from config.loader import load_reranker_config
    cfg = load_reranker_config()
    os.environ.setdefault("HF_HOME", cfg["model_cache_dir"])
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from sentence_transformers import CrossEncoder
        try:
            model = CrossEncoder(cfg["hf_repo"], device="cpu", revision=cfg.get("revision"))
        except TypeError:
            model = CrossEncoder(cfg["hf_repo"], device="cpu")
    except Exception as e:
        raise LookupError(f"reranker '{cfg['hf_repo']}' is not in the model cache "
                          f"({e.__class__.__name__}); download it: "
                          "python3 scripts/download_models.py --rerank") from None
    _reranker["model"] = model
    return model


def retrieval_query(question: str, hint: str = "") -> str:
    """Short follow-ups ("and who wrote it?") retrieve poorly on their own:
    fold the previous user turn into the retrieval query -- never into the
    prompt -- when the question is short or pronoun-led."""
    words = question.split()
    if hint and (len(words) < 8 or words[0].lower() in
                 ("it", "its", "this", "that", "they", "their", "he", "she", "and", "who", "what", "when")):
        return f"{question} {hint}"
    return question


def retrieve(app: AppConfig, question: str, hint: str = "", n: int = None):
    """Top-n (documents, sources) for a question. Dense + BM25 fused by
    reciprocal rank when hybrid is on, then optional cross-encoder rerank.
    Also backs the agent's search_documents tool and the verified pipeline."""
    col = collection(app.collection)
    stored = col.count()
    if stored == 0:
        return None
    top_k = int(n or app.top_k or 4)
    query = retrieval_query(question, hint)
    pool = max(top_k * 5, 20) if (app.hybrid or app.rerank) else top_k
    qv = embed([query], app.embeddings_base())[0]
    dense = col.query(query_embeddings=[qv], n_results=min(pool, stored),
                      include=["documents", "metadatas"])
    ids = list(dense["ids"][0])
    by_id = {i: (d, m["source"]) for i, d, m in
             zip(ids, dense["documents"][0], dense["metadatas"][0])}
    if app.hybrid:
        bm25, docs, metas, all_ids = _bm25_index(col)
        scores = bm25.get_scores(_tokens(query))
        ranked = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:pool]
        sparse = [all_ids[i] for i in ranked if scores[i] > 0]
        for i in ranked:
            by_id.setdefault(all_ids[i], (docs[i], metas[i]["source"]))
        ids = _rrf([ids, sparse])
    if app.rerank and len(ids) > 1:
        model = _cross_encoder()
        cand = ids[:pool]
        scored = model.predict([(query, by_id[i][0]) for i in cand])
        ids = [i for _, i in sorted(zip(scored, cand), key=lambda x: -x[0])]
    ids = ids[:top_k]
    return [by_id[i][0] for i in ids], [by_id[i][1] for i in ids]


def _file_digest(fp):
    import hashlib
    h = hashlib.sha256()
    with open(fp, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _stored_digest(col, rel):
    got = col.get(where={"source": rel}, include=["metadatas"], limit=1)
    metas = got.get("metadatas") or []
    return metas[0].get("digest") if metas else None


def ingest(app: AppConfig, path=None):
    target = os.path.join(DOCS_ROOT, app.docs_dir or ".") if path is None else path
    target = os.path.normpath(target)
    if not os.path.isdir(target):
        sys.exit(f"FATAL: docs directory '{target}' does not exist in the container.\n"
                 "       Set DOCS_DIR in .env to your corpus (restart compose) or pass "
                 "a mounted path explicitly.")
    files, skipped = [], {}
    for root, _, names in os.walk(target):
        for n in names:
            ext = os.path.splitext(n)[1].lower()
            if ext in SUPPORTED_EXT:
                files.append(os.path.join(root, n))
            else:
                skipped[ext or "(no extension)"] = skipped.get(ext or "(no extension)", 0) + 1
    if skipped:
        print("[rag] skipped (unsupported format): " +
              ", ".join(f"{k} x{v}" for k, v in sorted(skipped.items())) +
              "  -- supported: " + " ".join(sorted(SUPPORTED_EXT)))
    if not files:
        sys.exit(f"FATAL: no supported documents under {target} — nothing would be indexed.")
    col = collection(app.collection)
    size, overlap = int(app.chunk_chars or 800), int(app.chunk_overlap or 120)
    total, unchanged, seen = 0, 0, set()
    for fp in sorted(files):
        rel = os.path.relpath(fp, target)
        seen.add(rel)
        digest = _file_digest(fp)
        # unchanged since its last ingest: keep the stored chunks, skip the
        # parse/OCR/embed entirely (the digest check costs one file read)
        if _stored_digest(col, rel) == digest:
            unchanged += 1
            continue
        text = read_document(fp)
        if text is None:
            continue
        chunks = chunk_structured(text, size, overlap)
        if not chunks:
            print(f"[rag] skip {rel}: no extractable text")
            continue
        vecs = embed(chunks, app.embeddings_base())
        # a changed file may produce FEWER chunks than before; upsert alone
        # would leave the old tail behind, so clear the file's chunks first
        col.delete(where={"source": rel})
        for i in range(0, len(chunks), EMBED_BATCH):
            j = i + EMBED_BATCH
            col.upsert(ids=[f"{rel}::{k}" for k in range(i, min(j, len(chunks)))],
                       embeddings=vecs[i:j], documents=chunks[i:j],
                       metadatas=[{"source": rel, "digest": digest}] * len(chunks[i:j]))
        total += len(chunks)
        print(f"[rag] {rel}: {len(chunks)} chunks")
    # files deleted from the corpus disappear from the index too
    got = col.get(include=["metadatas"])
    gone = sorted({m["source"] for m in got["metadatas"]} - seen)
    for rel in gone:
        col.delete(where={"source": rel})
        print(f"[rag] removed from index (file no longer in corpus): {rel}")
    with _bm25_lock:
        _bm25_cache.pop(app.collection, None)
    print(f"[rag] ingest complete: {total} new/updated chunks "
          f"({unchanged} unchanged file(s) skipped) in '{app.collection}' at {STORE}")


def ask(app: AppConfig, question: str, hint: str = ""):
    if app.kind == "rag-verified":
        from rag_verified import answer as verified_answer
        result = verified_answer(app, question, hint, trace=lambda m: print(f"[verify] {m}"))
        print("\n" + result["answer"])
        if result["sources"]:
            print("\nsources: " + ", ".join(result["sources"]))
        return
    got = retrieve(app, question, hint)
    if got is None:
        print(f"[rag] collection '{app.collection}' is empty — run: "
              f"./app.sh {app.name} ingest")
        return
    docs, srcs = got
    import agent_demo
    data = _post(app.generation_base(), "/chat/completions", {
        **agent_demo.chat_body(app, max_tokens=768),
        "messages": [
            {"role": "system", "content": system_prompt_for(app) or DEFAULT_RAG_PROMPT},
            {"role": "user", "content": rag_user_message(docs, srcs, question)}]})
    print("\n" + (data["choices"][0]["message"].get("content") or "(empty)"))
    print("\nsources: " + ", ".join(sorted(set(srcs))))


DEFAULT_RAG_PROMPT = "Answer using ONLY the context. Cite snippets as [n]."


def rag_user_message(docs, srcs, question):
    """The numbered-context user turn every RAG path sends to the model."""
    context = "\n\n".join(f"[{i+1}] ({srcs[i]})\n{d}" for i, d in enumerate(docs))
    return f"Context:\n{context}\n\nQuestion: {question}"


def preflight(app: AppConfig):
    checks = [("generation", app.generation_base(), "./build.sh"),
              ("embedding", app.embeddings_base(), "./build.sh --embed")]
    for name, base, remedy in checks:
        try:
            urllib.request.urlopen(base + "/models", timeout=5).read()
        except Exception:
            sys.exit(f"FATAL: {name} server not reachable at {base}.\n"
                     f"       app.sh never starts servers — run: {remedy}")
    from agent_demo import check_profile
    check_profile(app)


def run(app: AppConfig, argv):
    preflight(app)
    if argv and argv[0] == "ingest":
        ingest(app, argv[1] if len(argv) > 1 else None)
        return
    if argv and argv[0] == "ask":
        ask(app, " ".join(argv[1:]))
        return
    print("======================================================================")
    print(f"  RAG app '{app.name}' — collection '{app.collection}', top_k {app.top_k}")
    print("  'ingest [dir]' indexes; anything else asks; 'exit' quits")
    print("======================================================================")
    previous = ""
    while True:
        try:
            q = input("\nrag> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not q or q in ("exit", "quit"):
            break
        if q.startswith("ingest"):
            parts = q.split(None, 1)
            ingest(app, parts[1] if len(parts) > 1 else None)
        else:
            ask(app, q, previous)
            previous = q


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: rag_demo.py <app-name> [ingest [dir] | ask <question>]")
    run(load_app(sys.argv[1]), sys.argv[2:])
