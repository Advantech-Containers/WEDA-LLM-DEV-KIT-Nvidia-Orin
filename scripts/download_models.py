#!/usr/bin/env python3
# ==============================================================================
# WEDA LLM Suite — model pre-fetch (host side)
# ==============================================================================
# Pre-fetches the GGUF files referenced by config/llm_config.yaml into the HF
# cache, so first serve/RAG start is instant and HF_HUB_OFFLINE=1 works.
#
# Runs on the HOST (needs: pip install huggingface-hub pyyaml). Downloads
# SINGLE GGUF FILES, not full snapshots -- a Qwen3-4B
# repo holds every quantisation (~15 GiB); the profile needs one file (2.3 GiB).
#
#   ./download_models.py                  # interactive picker (TTY) or default
#   ./download_models.py --profile NAME   # one profile (+ embedding)
#   ./download_models.py --all            # every profile in the config
#   ./download_models.py --status         # cached vs missing, nothing downloaded
#   ./download_models.py --no-embedding   # skip the RAG embedding model

import argparse
import os
import pathlib
import sys

BUNDLE = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH = BUNDLE / "config" / "llm_config.yaml"


def _apply_env():
    """Same cache and token as the containers: read HF_CACHE / HF_TOKEN from
    the bundle's .env (the compose source of truth) into HF_HOME / HF_TOKEN
    before huggingface_hub is imported. Explicit environment wins."""
    env = {}
    env_file = BUNDLE / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    cache = os.environ.get("HF_CACHE") or env.get("HF_CACHE") or "./model_cache/huggingface"
    cache = os.path.expanduser(cache)
    if not os.path.isabs(cache):
        cache = str(BUNDLE / cache)
    os.environ.setdefault("HF_HOME", cache)
    token = os.environ.get("HF_TOKEN") or env.get("HF_TOKEN")
    if token:
        os.environ.setdefault("HF_TOKEN", token)
    return cache


HF_CACHE = _apply_env()

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("[ERROR] huggingface-hub is not installed on the host.")
    print("Please install it using: pip install huggingface-hub")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("[ERROR] pyyaml is not installed on the host.")
    print("Please install it using: pip install pyyaml")
    sys.exit(1)

def load_targets(profile=None, everything=False, embedding=True, rerank=False):
    """(repo, file) pairs to fetch, always resolved THROUGH llm_config.yaml.
    No selection path ever names a repo id directly."""
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    profiles = cfg.get("profiles") or {}
    if not profiles:
        sys.exit(f"[ERROR] no profiles found in {CONFIG_PATH}")

    targets = []
    if everything:
        chosen = profiles.items()
    else:
        name = profile or cfg.get("default_profile")
        if name not in profiles:
            sys.exit(f"[ERROR] profile '{name}' not in config. "
                     f"Available: {', '.join(sorted(profiles))}")
        chosen = [(name, profiles[name])]
    for name, p in chosen:
        targets.append((name, p["hf_repo"], p["gguf_file"], p.get("revision") or None))

    if embedding and cfg.get("embedding"):
        e = cfg["embedding"]
        targets.append(("embedding", e["hf_repo"], e["gguf_file"], e.get("revision") or None))
    # the cross-encoder is a whole repo, not one file: gguf_file None marks it
    apps = cfg.get("applications") or {}
    rag_defaults = cfg.get("rag") or {}
    if rerank or any((a.get("rerank", rag_defaults.get("rerank", False))) for a in apps.values()):
        r = cfg.get("reranker") or {}
        targets.append(("reranker", r.get("hf_repo", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                        None, r.get("revision") or None))
    # de-dup: several profiles may share one GGUF
    seen, out = set(), []
    for name, repo, fn, rev in targets:
        if (repo, fn, rev) not in seen:
            seen.add((repo, fn, rev))
            out.append((name, repo, fn, rev))
    return out


def cached_path(repo, fn, revision=None):
    try:
        if fn is None:
            from huggingface_hub import snapshot_download
            return snapshot_download(repo_id=repo, revision=revision, local_files_only=True)
        return hf_hub_download(repo_id=repo, filename=fn, revision=revision, local_files_only=True)
    except Exception:
        return None


def _write_ref(snapshot_path, revision):
    # A snapshot fetched by commit hash carries no refs/main; a loader that resolves the
    # repository by name in offline mode needs that pointer, so it is written here.
    if not revision or not snapshot_path:
        return
    repo_dir = os.path.dirname(os.path.dirname(snapshot_path))
    refs = os.path.join(repo_dir, "refs")
    os.makedirs(refs, exist_ok=True)
    main = os.path.join(refs, "main")
    if not os.path.exists(main):
        with open(main, "w") as fh:
            fh.write(os.path.basename(snapshot_path))


def fetch(repo, fn, revision=None):
    if fn is None:
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo_id=repo, revision=revision)
        _write_ref(path, revision)
        return path
    return hf_hub_download(repo_id=repo, filename=fn, revision=revision)


def show_status(targets):
    missing = 0
    for name, repo, fn, rev in targets:
        p = cached_path(repo, fn, rev)
        if p:
            path = pathlib.Path(p)
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) \
                if path.is_dir() else path.stat().st_size
            print(f"  cached   {name:26s} {fn or '(repo)'}  "
                  f"({size / 2**30:.2f} GiB)")
        else:
            missing += 1
            print(f"  MISSING  {name:26s} {repo} / {fn}")
    print(f"\n{len(targets) - missing} cached, {missing} missing"
          + ("" if missing else " — offline-ready (set HF_HUB_OFFLINE=1)"))


def pick_interactive(embedding=True):
    everything = load_targets(everything=True, embedding=embedding)
    print("Profiles (models to download):")
    for i, (name, repo, fn, rev) in enumerate(everything, 1):
        state = "cached" if cached_path(repo, fn, rev) else "not cached"
        print(f"  [{i:2d}] {name:26s} {fn or '(whole repo: ' + repo + ')'}  ({state})")
    raw = input("\nPick numbers (e.g. 1,3), 'a' for all, Enter for default, 'q' to quit: ").strip()
    if raw.lower() in ("q", "quit", "exit"):
        sys.exit(0)
    if not raw:
        return load_targets(embedding=embedding)
    if raw.lower() in ("a", "all"):
        return everything
    try:
        idx = sorted({int(x) for x in raw.replace(",", " ").split()})
        if any(i < 1 or i > len(everything) for i in idx):
            raise IndexError(raw)
        return [everything[i - 1] for i in idx]
    except (ValueError, IndexError):
        sys.exit(f"[ERROR] invalid selection '{raw}' — use numbers from the list, or 'a'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=None)
    ap.add_argument("--all", action="store_true", help="every profile in the config")
    ap.add_argument("--status", action="store_true",
                    help="show cached vs missing; download nothing")
    ap.add_argument("--no-embedding", action="store_true")
    ap.add_argument("--rerank", action="store_true",
                    help="also cache the cross-encoder used by rerank: true apps")
    args = ap.parse_args()
    print(f"[download] cache: {HF_CACHE}  (HF_CACHE from .env; the containers mount the same path)")

    if args.status:
        show_status(load_targets(everything=True, embedding=not args.no_embedding, rerank=args.rerank))
        return
    if not args.profile and not args.all and sys.stdin.isatty():
        targets = pick_interactive(not args.no_embedding)
    else:
        targets = load_targets(args.profile, args.all, not args.no_embedding, rerank=args.rerank)
    print(f"[download] {len(targets)} GGUF file(s) to ensure in cache")
    failed = 0
    for name, repo, fn, rev in targets:
        print(f"[download] {name}: {repo}" + (f" / {fn}" if fn else " (whole repo)")
              + (f" @ {rev[:12]}" if rev else ""))
        try:
            path = fetch(repo, fn, rev)
            print(f"[download]   -> {path}")
        except Exception as e:
            failed += 1
            print(f"[download]   FAILED: {e}")
    if failed:
        sys.exit(f"[ERROR] {failed} download(s) failed")
    print("[download] done — set HF_HUB_OFFLINE=1 in .env for guaranteed-offline runs")


if __name__ == "__main__":
    main()
