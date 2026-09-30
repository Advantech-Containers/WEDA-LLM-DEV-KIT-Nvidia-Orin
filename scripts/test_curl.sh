#!/usr/bin/env bash
# ==============================================================================
# WEDA LLM Suite — one-shot API smoke query
# ==============================================================================
# Asks the served model one question and prints the answer with timing and
# token usage. Endpoint follows .env; always loopback (runs on-device).
set -euo pipefail
trap 'echo "[test_curl] FATAL: unexpected error at line ${LINENO} (command: ${BASH_COMMAND})" >&2' ERR

PORT="${WEDA_LLM_PORT:-9000}"
API="http://127.0.0.1:${PORT}"

MODEL_ID=$(curl -fsS "${API}/v1/models" 2>/dev/null \
  | python3 -c "import sys,json; print((json.load(sys.stdin).get('data') or [{}])[0].get('id',''))" 2>/dev/null \
  || true)
if [ -z "${MODEL_ID}" ]; then
  echo "[ERROR] no served model at ${API} — start the server first (./build.sh)" >&2
  exit 1
fi

QUESTION="Why does running a quantized LLM locally on an 8 GB Jetson Orin edge \
device benefit industrial operations compared to cloud-based LLM services? \
Answer in one brief, professional sentence."

echo "[query] model: ${MODEL_ID}"
echo "[query] ${QUESTION}"
echo

# payload built by python: the question and model id are JSON-encoded, never
# shell-interpolated into the request body
PAYLOAD=$(MODEL_ID="${MODEL_ID}" QUESTION="${QUESTION}" python3 -c "
import json, os
print(json.dumps({'model': os.environ['MODEL_ID'],
                  'messages': [{'role': 'user', 'content': os.environ['QUESTION']}],
                  'max_tokens': 1024}))")

T_START=$(date +%s%N)
RESPONSE=$(curl -fsS -X POST "${API}/v1/chat/completions" \
  -H "Content-Type: application/json" -d "${PAYLOAD}") \
  || { echo "[ERROR] request to ${API} failed (server busy, swapping, or stopped) -- ./build.sh --status" >&2; exit 1; }
T_END=$(date +%s%N)

ELAPSED_MS=$(( (T_END - T_START) / 1000000 ))
RESPONSE="${RESPONSE}" python3 - "${ELAPSED_MS}" <<'PY'
import json
import os
import sys

data = json.loads(os.environ["RESPONSE"])
elapsed_s = float(sys.argv[1]) / 1000.0
msg = data.get("choices", [{}])[0].get("message", {})
content = (msg.get("content") or "").strip()
reasoning = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()
usage = data.get("usage", {})
completion = usage.get("completion_tokens", 0)

print("Response:")
print(f"  {content}" if content else "  [no content — output empty or reasoning-only]")
if reasoning:
    print("Reasoning:")
    for line in reasoning.splitlines():
        print(f"  {line}")
print()
print(f"served model:      {data.get('model', 'N/A')}")
print(f"prompt tokens:     {usage.get('prompt_tokens', 0)}")
print(f"completion tokens: {completion}")
print(f"latency:           {elapsed_s:.2f} s")
if completion and elapsed_s:
    print(f"throughput:        {completion / elapsed_s:.2f} tok/s")
PY
