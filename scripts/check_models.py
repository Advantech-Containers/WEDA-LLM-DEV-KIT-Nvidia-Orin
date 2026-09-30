#!/usr/bin/env python3
# ==============================================================================
# WEDA LLM Suite — profile fit-check ("will it fit?" before any download)
# ==============================================================================
# Per profile: GGUF size, KV-cache estimate from the profile's own settings,
# verdict against this machine's free memory, engine architecture support.
# Reads the GGUF header from the local cache when present, otherwise via an
# HTTP range request (a few MB, never the whole file).
#
#   check_models.py              # every profile
#   check_models.py <profile>    # one profile
import os
import struct
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import load_llm_config

T_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
KV_BYTES = {"f16": 2.0, "q8_0": 1.0625}
RUNTIME_OVERHEAD_GIB = 1.2


def need_from_meta(p, meta, size_gib):
    """(need_gib, kv_gib, arch) for a profile from parsed GGUF metadata --
    the one place the KV/footprint arithmetic lives."""
    def g(suffix, default=None):
        for k, v in meta.items():
            if k.endswith(suffix):
                return v
        return default
    arch = str(meta.get("general.architecture", "?"))
    layers = g(".block_count", 0) or 0
    heads = g(".attention.head_count", 1) or 1
    if isinstance(heads, list):
        heads = max(heads)
    h_kv = g(".attention.head_count_kv")
    if h_kv is None:
        h_kv = heads
    if isinstance(h_kv, list):
        h_kv = max(h_kv)
    emb = g(".embedding_length", 0) or 0
    klen = g(".attention.key_length") or (emb // heads if heads else 128)
    vlen = g(".attention.value_length") or klen
    ctx_total = p.n_ctx * max(1, p.parallel)
    bp_k = KV_BYTES.get(p.cache_type_k, 2.0)
    bp_v = KV_BYTES.get(p.cache_type_v, 2.0)
    kv_gib = layers * ctx_total * h_kv * (klen * bp_k + vlen * bp_v) / 2**30
    return size_gib + kv_gib + RUNTIME_OVERHEAD_GIB, kv_gib, arch


def estimate_need(p, path):
    """(need_gib, kv_gib, size_gib, hybrid) for a profile with a local GGUF at
    `path` -- the same arithmetic the check verdict prints."""
    meta = gguf_meta(_Src(path=path))
    size_gib = os.path.getsize(path) / 2**30
    need, kv_gib, arch = need_from_meta(p, meta, size_gib)
    return need, kv_gib, size_gib, arch.endswith("_h")


class _Src:
    def __init__(self, path=None, url=None):
        self.f = open(path, "rb") if path else None
        self.url, self.data, self.pos = url, b"", 0

    def take(self, n):
        if self.f:
            b = self.f.read(n)
        else:
            while self.pos + n > len(self.data):
                start = len(self.data)
                req = urllib.request.Request(
                    self.url, headers={"Range": f"bytes={start}-{start + 8*2**20 - 1}"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    chunk = r.read()
                if not chunk:
                    raise EOFError
                self.data += chunk
            b = self.data[self.pos:self.pos + n]
            self.pos += n
        return b

    def skip(self, n):
        if self.f:
            self.f.seek(n, 1)
        else:
            self.take(n)


def _read_str(s):
    (n,) = struct.unpack("<Q", s.take(8))
    return s.take(n).decode(errors="replace")


def _read_val(s, t, keep):
    if t == 8:
        v = _read_str(s)
        return v if keep else None
    if t == 9:
        (et,) = struct.unpack("<I", s.take(4))
        (n,) = struct.unpack("<Q", s.take(8))
        if et == 8:
            for _ in range(n):
                (ln,) = struct.unpack("<Q", s.take(8))
                s.skip(ln)
        elif et == 9:
            for _ in range(n):
                _read_val(s, 9, False)
        else:
            s.skip(T_SIZES[et] * n)
        return None
    raw = s.take(T_SIZES[t])
    if not keep:
        return None
    fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
           7: "<?", 10: "<Q", 11: "<q", 12: "<d"}[t]
    return struct.unpack(fmt, raw)[0]


WANT = ("general.architecture", ".block_count", ".attention.head_count",
        ".attention.head_count_kv", ".attention.key_length",
        ".attention.value_length", ".embedding_length")


def gguf_meta(src):
    assert src.take(4) == b"GGUF", "not a GGUF file"
    _, _, n_kv = struct.unpack("<IQQ", src.take(20))
    out = {}
    for _ in range(n_kv):
        key = _read_str(src)
        (t,) = struct.unpack("<I", src.take(4))
        keep = any(key.endswith(w) for w in WANT)
        v = _read_val(src, t, keep)
        if keep:
            out[key] = v
    return out


def local_gguf(repo, filename, revision=""):
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=repo, filename=filename,
                               revision=revision or None, local_files_only=True)
    except Exception:
        return None


def remote_size(repo, filename, revision=""):
    url = f"https://huggingface.co/{repo}/resolve/{revision or 'main'}/{filename}"
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers.get("Content-Length") or 0), url


def engine_blob():
    """Raw bytes of the local engine library for substring probes; else None."""
    for lib in ("/usr/local/lib/llamacpp/libllama.so",):
        if os.path.exists(lib):
            with open(lib, "rb") as f:
                blob = f.read()
            return blob
    return None


def mem_available_gib():
    try:
        import device_tools
        return device_tools.meminfo()["MemAvailable"] / 2**20
    except Exception:
        return None


def check(name):
    p = load_llm_config(name)
    path = local_gguf(p.hf_repo, p.gguf_file, getattr(p, "revision", ""))
    if path:
        size_gib = os.path.getsize(path) / 2**30
        meta = gguf_meta(_Src(path=path))
        cached = "cached"
    else:
        size, url = remote_size(p.hf_repo, p.gguf_file, getattr(p, "revision", ""))
        size_gib = size / 2**30
        meta = gguf_meta(_Src(url=url))
        cached = "not cached"

    need, kv_gib, arch = need_from_meta(p, meta, size_gib)

    blob = engine_blob()
    if blob is None:
        arch_note = "engine not inspectable here"
    elif arch.encode() in blob:
        arch_note = "engine OK"
    else:
        arch_note = ("python backend LACKS this arch" if p.backend == "python"
                     else "not in inspectable lib (C++ server may still support it)")

    avail = mem_available_gib()
    if avail is None:
        verdict = "UNKNOWN (no /proc/meminfo)"
    elif need <= avail:
        verdict = f"FITS ({need:.1f} needed, {avail:.1f} available)"
    else:
        verdict = f"DOES NOT FIT NOW ({need:.1f} needed, {avail:.1f} available)"
    hybrid = " [hybrid arch: KV is an upper bound]" if arch.endswith("_h") else ""

    print(f"{name}")
    print(f"  file      {p.gguf_file}  {size_gib:.2f} GiB  ({cached})")
    print(f"  arch      {arch}  ({arch_note})")
    print(f"  kv cache  {kv_gib:.2f} GiB @ ctx {p.n_ctx} x {p.parallel} slots, "
          f"{p.cache_type_k}{hybrid}")
    print(f"  verdict   {verdict}")


def main():
    import yaml
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "config", "llm_config.yaml")
    names = sys.argv[1:] or sorted(
        yaml.safe_load(open(cfg_path, encoding="utf-8"))["profiles"])
    failures = 0
    for n in names:
        try:
            check(n)
        except Exception as e:
            failures += 1
            print(f"{n}\n  ERROR: {e}")
        print()
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
