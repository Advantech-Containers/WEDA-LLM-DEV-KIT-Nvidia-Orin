#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[entrypoint] FATAL: unexpected error at line ${LINENO} (command: ${BASH_COMMAND})" >&2' ERR

cat <<'BANNER'
======================================================================
  WISE-Edge WEDA LLM Suite — llama.cpp / GGUF / Jetson Orin (8 GB UMA)
======================================================================
BANNER

APP=/wise-edge/app

# ── 1. Environment report (static checks only -- no CUDA dlopen at start) ────
python3 - <<'PY'
import importlib.metadata as im
import os, subprocess
def v(name):
    try: return im.version(name)
    except Exception: return "absent"
print(f"  llama-cpp-python  {v('llama-cpp-python')}")
print(f"  chromadb          {v('chromadb')}")
srv = "/usr/local/bin/llama-server"
if os.access(srv, os.X_OK):
    try:
        line = subprocess.run([srv, "--version"], capture_output=True, text=True,
                              timeout=10).stderr.strip().splitlines()
        print(f"  llama-server      {line[-1] if line else 'present'}")
    except Exception:
        print("  llama-server      present (version probe failed)")
else:
    print("  llama-server      ABSENT -- llamacpp-backend profiles and RAG")
    print("                    embeddings need weda-llm-base >= 1.0.0-orin")
PY

# ── 2. Memory pre-flight (8 GB device: a stale server strands the GPU heap) ──
avail_mb=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
echo "  MemAvailable      ${avail_mb:-unknown} MiB"
if [ -n "${avail_mb}" ] && [ "${avail_mb}" -lt 3000 ]; then
  echo "  WARN: <3000 MiB available. A previous server may still hold memory;"
  echo "        'docker compose down' any stale container before serving 4B models."
fi

echo "[entrypoint] Environment ready."
echo "======================================================================"

MODE="${1:-serve}"
shift || true

case "${MODE}" in
  serve)
    exec python3 "${APP}/scripts/serve.py" "$@"
    ;;

  embed)
    exec python3 "${APP}/scripts/serve.py" --embedding "$@"
    ;;

  gateway)
    exec python3 "${APP}/scripts/app_gateway.py" "$@"
    ;;

  shell|bash)
    exec /bin/bash "$@"
    ;;

  python|python3)
    exec python3 "$@"
    ;;

  *)
    exec "${MODE}" "$@"
    ;;
esac
