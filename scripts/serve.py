#!/usr/bin/env python3
# ==============================================================================
# WEDA LLM Suite — Model Server Launcher (llama.cpp backends)
# ==============================================================================
# One launcher, three shapes:
#   serve      generation server per config profile
#                backend llamacpp -> C++ llama-server (tool calls, slots)
#                backend python   -> llama_cpp.server (classic API parity)
#   --embedding  nomic embedding server (RAG; second process, port WEDA_EMBED_PORT)
#
# The GGUF is resolved through huggingface_hub INTO the mounted HF cache, then
# passed as a local --model path to either backend, so both backends share
# one cache, warm runs are offline-capable, and llama-server's
# `-hf repo:TAG` quirk (tag, not filename) is out of the picture entirely.
import argparse
import json
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.loader import load_llm_config, load_embedding_config
import device_tools

LLAMA_SERVER = "/usr/local/bin/llama-server"


def gguf_arch(path: str) -> str:
    """general.architecture from the GGUF header; '' when unreadable."""
    try:
        from check_models import gguf_meta, _Src
        return str(gguf_meta(_Src(path=path)).get("general.architecture", ""))
    except Exception:
        return ""


def _read_int(path):
    try:
        return int(open(path).read().strip())
    except (OSError, ValueError):
        return 0


# processes that identify a cgroup as part of THIS stack; anything else on
# the device (a customer's own containers) is never reclaimed from
_STACK_CMDLINES = ("open_webui", "open-webui", "llama-server",
                   "app_gateway", "/wise-edge/app")


def _cgroup_is_stack(d):
    try:
        pids = open(os.path.join(d, "cgroup.procs")).read().split()
    except OSError:
        return False
    for pid in pids[:8]:
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if any(s in cmd for s in _STACK_CMDLINES):
            return True
    return False


def _sibling_cgroups():
    """memory cgroups of this stack's OTHER containers (never our own, never
    a third-party workload -- membership is decided by the processes inside,
    read via pid: host). Covers both docker cgroup drivers: systemd
    (system.slice/docker-*.scope) and cgroupfs (/sys/fs/cgroup/docker/*).
    Needs the privileged container's host cgroup view (cgroup: host in
    compose) -- absent that, returns [] and the caller degrades."""
    import glob as _glob
    import re as _re
    try:
        own = _re.search(r"[0-9a-f]{64}", open("/proc/self/cgroup").read())
        own_id = own.group(0) if own else "\0"
    except OSError:
        return []
    dirs = (_glob.glob("/sys/fs/cgroup/system.slice/docker-*.scope") +
            _glob.glob("/sys/fs/cgroup/docker/*"))
    return [d for d in dirs
            if own_id not in d and os.path.isfile(os.path.join(d, "memory.current"))
            and _cgroup_is_stack(d)]


def restore_sibling_limits():
    """Self-heal a clamp orphaned by a mid-swap kill: if this process died
    between clamping a sibling's memory.high and the finally-restore, the
    sibling would stay throttled forever. Run at every startup; a no-op when
    nothing is clamped."""
    for d in _sibling_cgroups():
        path = os.path.join(d, "memory.high")
        try:
            if open(path).read().strip() != "max":
                with open(path, "w") as f:
                    f.write("max")
                print(f"[serve] restored memory.high on {os.path.basename(d)} "
                      "(orphaned clamp from an interrupted swap)", flush=True)
        except OSError:
            continue


def free_headroom(deficit_gib):
    """Reclaim idle memory from the other containers into swap so a swap-in
    model gets cold-start headroom without stopping anything. Prefers the
    kernel's one-shot memory.reclaim; on kernels without it (Orin's 5.15),
    temporarily clamps memory.high and lets the kernel push idle pages out,
    then always restores the limit. Anonymous pages only move if swap is
    enabled -- without swap this frees file cache at best."""
    want = int(deficit_gib * (1 << 30))
    sibs = sorted(_sibling_cgroups(),
                  key=lambda d: -_read_int(os.path.join(d, "memory.current")))
    if not sibs:
        print("[serve] headroom reclaim: no sibling cgroups visible -- skipping", flush=True)
        return
    clamped = []
    try:
        for d in sibs:
            cur = _read_int(os.path.join(d, "memory.current"))
            if cur < (64 << 20):
                continue
            name = os.path.basename(d)
            try:
                with open(os.path.join(d, "memory.reclaim"), "w") as f:
                    f.write(str(min(want, cur)))
                print(f"[serve] headroom reclaim: memory.reclaim on {name}", flush=True)
                continue
            except OSError:
                pass
            high_path = os.path.join(d, "memory.high")
            try:
                old = open(high_path).read().strip()
                with open(high_path, "w") as f:
                    f.write(str(128 << 20))
                clamped.append((high_path, old))
                print(f"[serve] headroom reclaim: clamped memory.high on {name}", flush=True)
            except OSError:
                continue
        def _avail_kb():
            return device_tools.meminfo().get("MemAvailable", 0)
        base = _avail_kb()
        deadline = time.time() + 15
        while time.time() < deadline:
            time.sleep(1)
            gained = (_avail_kb() - base) << 10
            if gained >= want:
                break
        print(f"[serve] headroom reclaim: +{max(0, (_avail_kb() - base)) >> 10} MiB freed", flush=True)
    finally:
        for path, old in clamped:
            try:
                with open(path, "w") as f:
                    f.write(old)
            except OSError:
                pass


def fit_gate(p, model_path):
    """Refuse a load whose computed need exceeds available memory. The same
    arithmetic as ./app.sh check; hybrid architectures are an upper bound, so
    they get a small grace margin instead of a hard refusal at the line.
    Before refusing, reclaim idle memory from the other containers -- the
    headroom a cold start gets, without restarting anything."""
    try:
        from check_models import estimate_need, mem_available_gib
        need, kv, size, hybrid = estimate_need(p, model_path)
        avail = mem_available_gib()
    except Exception as e:
        print(f"[serve] WARN: fit check unavailable ({e}) -- loading anyway")
        return
    if avail is None:
        return
    margin = 0.5 if hybrid else 0.0
    if need - margin > avail:
        deficit = need - margin - avail
        print(f"[serve] '{p.name}' needs {need:.1f} GiB but {avail:.1f} GiB is free -- "
              f"reclaiming idle memory from other containers...", flush=True)
        free_headroom(deficit + 0.3)
        wait_memory_settle(timeout=10)
        avail = mem_available_gib() or avail
    if need - margin > avail:
        sys.exit(f"FATAL: '{p.name}' does not fit: {need:.1f} GiB needed "
                 f"({size:.1f} weights + {kv:.1f} KV + 1.2 runtime) vs "
                 f"{avail:.1f} GiB available (after reclaiming from other containers).\n"
                 f"       Free memory (./build.sh --stop other workloads, close the desktop)\n"
                 f"       or pick a smaller shape: lower n_ctx / parallel, or a smaller model.\n"
                 f"       ./app.sh check {p.name} shows this verdict any time.")
    print(f"[serve] fit check: {need:.1f} GiB needed, {avail:.1f} GiB available -- OK", flush=True)


def resolve_gguf(repo: str, filename: str, revision: str = "") -> str:
    """Local cache path of a GGUF. Downloads are the job of
    scripts/download_models.py alone; a missing file is refused here so a
    start or swap never turns into a silent multi-GB fetch."""
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(repo_id=repo, filename=filename,
                               revision=revision or None, local_files_only=True)
    except Exception:
        sys.exit(f"FATAL: {filename} ({repo}) is not in the model cache.\n"
                 f"       Download it first, then start again:\n"
                 f"         python3 scripts/download_models.py\n"
                 f"       (cache: {os.environ.get('HF_HOME')})")
    print(f"[serve] model cached: {path}")
    return path


def wait_memory_settle(timeout=20):
    """Block until the previous server's memory is actually back. Unified-
    memory allocations take a few seconds to return after the process exits;
    launching into that window is how a swap OOMs. Settled = MemAvailable
    stops rising between one-second samples."""
    def available():
        return device_tools.meminfo().get("MemAvailable", 0) // 1024
    start = available()
    prev, stable = start, 0
    for _ in range(timeout):
        time.sleep(1)
        now = available()
        if now - prev < 50:
            stable += 1
            if stable >= 2:
                break
        else:
            stable = 0
        prev = now
    print(f"[serve] memory settled: {prev} MiB available "
          f"(+{max(0, prev - start)} MiB reclaimed)", flush=True)


def drop_page_caches():
    """Flush page cache/dentries before loading a model. On 8 GB unified
    memory the file cache from a previous model download/load can leave too
    little contiguous headroom and the load OOMs. Needs a privileged
    container (compose sets it); degrades to a warning, never a failure,
    when the write is not permitted."""
    ok, msg = device_tools.drop_caches()
    if ok:
        print("[serve] page caches flushed (drop_caches=3)")
    else:
        print(f"[serve] WARN: could not flush page caches ({msg}) -- "
              "on a tight-memory start, run the container privileged")


def require_llama_server():
    if not os.access(LLAMA_SERVER, os.X_OK):
        sys.exit(
            f"FATAL: {LLAMA_SERVER} not present in this image.\n"
            "       backend 'llamacpp' needs weda-llm-base >= 1.0.0-orin, which\n"
            "       ships the C++ llama-server. Pull the current image, or switch\n"
            "       this profile to backend: python.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=os.environ.get("LLM_PROFILE") or None)
    ap.add_argument("--embedding", action="store_true",
                    help="launch the RAG embedding server instead of the generation server")
    args = ap.parse_args()

    host = os.environ.get("WEDA_LLM_HOST", "127.0.0.1")

    if args.embedding:
        e = load_embedding_config()
        os.environ.setdefault("HF_HOME", e.model_cache_dir)
        port = os.environ.get("WEDA_EMBED_PORT", "9001")
        require_llama_server()
        model = resolve_gguf(e.hf_repo, e.gguf_file, getattr(e, "revision", ""))
        # batch sizes = the context window: an embedding input must fit in
        # ONE physical batch, and the 512-token default rejects longer texts
        cmd = [LLAMA_SERVER, "--host", host, "--port", port,
               "--model", model, "--embeddings", "--pooling", e.pooling,
               "--ctx-size", str(e.n_ctx),
               "--batch-size", str(e.n_ctx), "--ubatch-size", str(e.n_ctx),
               "--n-gpu-layers", str(e.n_gpu_layers)]
        print(f"[serve] embedding server: {' '.join(cmd)}", flush=True)
        os.execv(cmd[0], cmd)

    config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "config", "llm_config.yaml")

    def build_cmd():
        p = load_llm_config(args.profile)
        os.environ.setdefault("HF_HOME", p.model_cache_dir)
        port = os.environ.get("WEDA_LLM_PORT", "9000")
        print(f"[serve] profile: {p.name}  backend: {p.backend}")
        model = resolve_gguf(p.hf_repo, p.gguf_file, getattr(p, "revision", ""))
        fit_gate(p, model)
        if p.backend == "llamacpp":
            require_llama_server()
            # --alias = the PROFILE name: /v1/models then reports which profile
            # is actually being served, so apps detect a mismatch instead of
            # assuming.
            # --context-shift: a conversation that outgrows the slot window
            # drops its oldest tokens and continues, instead of erroring the
            # request (open-ended web chats hit the window routinely)
            # --cache-reuse: a follow-up turn re-uses the KV prefix it shares
            # with the previous request (system prompt + earlier turns) instead
            # of re-processing it, which is most of the time-to-first-token
            # in a conversation
            cmd = [LLAMA_SERVER, "--host", host, "--port", port,
                   "--model", model, "--alias", p.name,
                   "--ctx-size", str(p.n_ctx * max(1, p.parallel)),
                   "--parallel", str(p.parallel),
                   "--cache-reuse", "256",
                   "--n-gpu-layers", str(p.n_gpu_layers)]
            # context shift exists for pure-attention models only; recurrent/
            # hybrid architectures (Mamba-based *_h) cannot roll their state
            if gguf_arch(model).endswith("_h"):
                print(f"[serve] {p.name}: hybrid architecture -- context shift "
                      "unavailable; long conversations are trimmed by the "
                      "gateway instead")
            else:
                cmd.append("--context-shift")
            if p.jinja:
                cmd.append("--jinja")
            # this llama-server takes a value: --flash-attn [on|off|auto]
            if p.flash_attn:
                cmd += ["--flash-attn", "on"]
            # KV quantisation is a MEMORY lever, not a speed one -- f16 is
            # faster when it fits (both cache types require flash-attn).
            if p.cache_type_k != "f16":
                cmd += ["--cache-type-k", p.cache_type_k]
            if p.cache_type_v != "f16":
                cmd += ["--cache-type-v", p.cache_type_v]
            return cmd + list(p.extra_serve_args or []), p
        if p.backend == "python":
            ignored = [k for k, v in (("parallel", p.parallel > 1), ("jinja", p.jinja),
                                      ("flash_attn", p.flash_attn),
                                      ("cache_type_k", p.cache_type_k != "f16"),
                                      ("cache_type_v", p.cache_type_v != "f16")) if v]
            if ignored:
                print(f"[serve] WARN: backend 'python' ignores {ignored} "
                      f"(llama-server options; profile '{p.name}')")
            return ([sys.executable, "-m", "llama_cpp.server",
                     "--host", host, "--port", port,
                     "--model", model, "--model_alias", p.name,
                     "--n_ctx", str(p.n_ctx),
                     "--n_gpu_layers", str(p.n_gpu_layers)]
                    + list(p.extra_serve_args or []), p)
        sys.exit(f"FATAL: unknown backend '{p.backend}' in profile '{p.name}' "
                 f"(expected: llamacpp | python)")

    def announce_ready(port, profile_name, proc):
        # poll until the server answers, then print one unmistakable banner --
        # the readiness signal without digging through llama-server's log
        import urllib.request
        deadline = time.time() + 900
        while time.time() < deadline and proc.poll() is None:
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/v1/models", timeout=2) as r:
                    served = (json.loads(r.read()).get("data") or [{}])[0].get("id", "?")
                print("\n" + "=" * 70 +
                      f"\n[serve] READY -- serving '{served}' "
                      f"(profile {profile_name}) on :{port}\n" + "=" * 70 + "\n",
                      flush=True)
                write_status("serving", profile_name)
                return
            except Exception:
                time.sleep(2)

    def launch():
        drop_page_caches()
        cmd, p_ = build_cmd()
        port = os.environ.get("WEDA_LLM_PORT", "9000")
        print(f"[serve] executing: {' '.join(cmd)}", flush=True)
        write_status("loading", p_.name)
        proc = subprocess.Popen(cmd)
        threading.Thread(target=announce_ready,
                         args=(port, p_.name, proc), daemon=True).start()
        return proc

    def stop(proc):
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    # Hot swap: the supervisor watches
    # config/llm_config.yaml (host-editable through the ro-mount) and restarts
    # the server on change -- edit default_profile or the profile body, no
    # container restart. Child crash exits the supervisor with the child's
    # code so compose surfaces real failures instead of hiding them.
    import signal

    status_file = "/wise-edge/app/test_logs/serve_status.json"

    def write_status(state, reason=""):
        # the gateway reads this over the shared test_logs mount and attaches
        # the reason to its 503, so a failed swap is announced in the client's
        # own error message instead of only in ./build.sh --logs
        try:
            with open(status_file, "w", encoding="utf-8") as f:
                json.dump({"state": state, "reason": reason,
                           "ts": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
        except OSError:
            pass

    def idle(reason):
        print(f"\n[serve] HOT SWAP FAILED: {reason}\n"
              "[serve] idle -- fix config/llm_config.yaml and save it (or "
              "touch it) to relaunch; the container stays up\n", flush=True)
        write_status("idle", reason)
        return subprocess.Popen(["sleep", "infinity"])

    # mtime is captured BEFORE the first launch so an edit made during a
    # multi-minute cold download is still picked up
    restore_sibling_limits()
    last_mtime = os.path.getmtime(config_file)
    swapped_at = None
    stopping = {"flag": False}
    try:
        proc = launch()
    except (SystemExit, Exception) as e:
        proc = idle(str(e))

    def _on_term(signum, frame):
        stopping["flag"] = True
        stop(proc)
        sys.exit(0)
    signal.signal(signal.SIGTERM, _on_term)

    try:
        while True:
            time.sleep(2)
            rc = proc.poll()
            if rc is not None and not stopping["flag"]:
                # a server that dies shortly after a hot swap is a bad new
                # config (flag typo, over-sized n_ctx, unsupported model):
                # stay up and idle rather than take the container down
                if swapped_at and time.time() - swapped_at < 120:
                    proc = idle(f"the new server exited with code {rc} "
                                "(see the llama-server lines above)")
                    swapped_at = None
                    continue
                sys.exit(rc)
            try:
                mtime = os.path.getmtime(config_file)
            except OSError:
                continue
            if mtime != last_mtime:
                last_mtime = mtime
                print("\n[serve] llm_config.yaml changed -- hot-swapping the "
                      "model (container stays up)\n", flush=True)
                stop(proc)
                # the old model's memory must be back before the new one loads
                wait_memory_settle()
                try:
                    proc = launch()
                    swapped_at = time.time()
                except (SystemExit, Exception) as e:
                    if stopping["flag"]:
                        sys.exit(0)
                    proc = idle(str(e))
    except KeyboardInterrupt:
        stop(proc)


if __name__ == "__main__":
    main()
