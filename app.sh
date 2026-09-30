#!/usr/bin/env bash
# ==============================================================================
# WEDA LLM Suite (JetPack 6 / Orin) — application launcher (CLIENTS ONLY)
#
#   ./app.sh                          # list applications from the config
#   ./app.sh <name> [args]            # run an application (REPL or one-shot)
#   ./app.sh <rag-app> ingest [dir]   # index documents
#   ./app.sh <rag-app> ask "q"        # one-shot RAG question
#   ./app.sh check [profile]          # will-it-fit check (size, KV, memory, arch)
#   ./app.sh curl                     # quick canned query (scripts/test_curl.sh)
#
# This script NEVER starts, stops or restarts containers/servers — that is
# build.sh's job. If a required server is down, the app tells you exactly
# which build.sh invocation is missing and exits.
# ==============================================================================
set -euo pipefail
trap 'echo "[weda-llm-suite] FATAL: unexpected error at line ${LINENO} (command: ${BASH_COMMAND})" >&2' ERR
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ── Banner ────────────────────────────────────────────────────────────────────
echo -e "\033[1;36m"
cat << "EOF"
   ╦ ╦╔═╗╔╦╗╔═╗   ╦  ╦  ╔╦╗  ╔═╗╦ ╦╦╔╦╗╔═╗
   ║║║║╣  ║║╠═╣───║  ║  ║║║──╚═╗║ ║║ ║ ║╣
   ╚╩╝╚═╝═╩╝╩ ╩   ╩═╝╩═╝╩ ╩  ╚═╝╚═╝╩ ╩ ╚═╝
          Jetson Orin Optimized  |  Powered by Advantech Co. Ltd
EOF
echo -e "\033[0m"

die() { echo -e "\033[0;31m[weda-app] FATAL:\033[0m $*" >&2; exit 1; }

[[ -f .env ]] || die ".env missing — copy the template first."
set -a; source .env; set +a
CTR="${WEDA_LLM_CONTAINER_NAME:-weda-llm-suite}"

running() { [[ "$(docker inspect -f '{{.State.Running}}' "${CTR}" 2>/dev/null)" == "true" ]]; }
running || die "container '${CTR}' is not running. Start it first: ./build.sh"

if [[ $# -eq 0 ]]; then
  echo "Applications (config/llm_config.yaml):"
  docker exec "${CTR}" python3 -c "
import sys; sys.path.insert(0, '/wise-edge/app')
from config.loader import list_apps, load_app
for a in list_apps():
    app = load_app(a)
    extra = f' tools={\",\".join(app.tools)}' if app.tools else ''
    print(f'  {a:24s} kind={app.kind:5s} profile={app.profile}{extra}')
    for ex in app.examples[:2]:
        print(f'      e.g. {ex}')"
  echo
  echo "Run one:  ./app.sh <name>"
  exit 0
fi

NAME="$1"; shift || true

case "${NAME}" in
  check)
    exec docker exec -t "${CTR}" python3 scripts/check_models.py "$@"
    ;;
  curl)
    exec docker exec -t "${CTR}" bash scripts/test_curl.sh
    ;;
  *)
    # resolve in-container so config errors come from one place (loader.py)
    KIND=$(docker exec "${CTR}" python3 -c "
import sys; sys.path.insert(0, '/wise-edge/app')
from config.loader import load_app
print(load_app(sys.argv[1]).kind)" "${NAME}") || exit 1
    case "${KIND}" in
      rag|rag-verified) exec docker exec -it "${CTR}" python3 scripts/rag_demo.py "${NAME}" "$@" ;;
      agent|chat) exec docker exec -it "${CTR}" python3 scripts/agent_demo.py "${NAME}" "$@" ;;
      *)          die "unhandled kind '${KIND}' for app '${NAME}'" ;;
    esac
    ;;
esac
