#!/usr/bin/env bash
# ==============================================================================
# WEDA LLM Suite (JetPack 6 / Orin) — launcher
#
# LIFECYCLE ONLY: start containers, set the environment, start servers.
# Applications are run separately with ./app.sh — this script never runs one.
#
#   ./build.sh                  # generation server + Open WebUI
#   ./build.sh -p <profile>     # ...serving a specific config profile
#   ./build.sh --embed          # also start the embedding server (RAG apps)
#   ./build.sh --shell          # shell inside the running container
#   ./build.sh --status         # containers, health, served model at a glance
#   ./build.sh --logs | --stop
# ==============================================================================
set -euo pipefail
usage() {
  cat <<'USAGE'
WEDA LLM Suite (JetPack 6 / Orin) — launcher

  ./build.sh                  # generation server + Open WebUI
  ./build.sh -p <profile>     # ...serving a specific config profile
  ./build.sh --embed          # also start the embedding server (RAG apps)
  ./build.sh --shell          # shell inside the running container
  ./build.sh --status         # containers, health, served model at a glance
  ./build.sh --logs | --stop
USAGE
}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
trap 'echo -e "\033[0;31m[weda-llm-suite] FATAL:\033[0m unexpected error at build.sh line ${LINENO} (command: ${BASH_COMMAND})" >&2' ERR

# ── Banner ────────────────────────────────────────────────────────────────────
echo -e "\033[1;36m"
cat << "EOF"
   ╦ ╦╔═╗╔╦╗╔═╗   ╦  ╦  ╔╦╗  ╔═╗╦ ╦╦╔╦╗╔═╗
   ║║║║╣  ║║╠═╣───║  ║  ║║║──╚═╗║ ║║ ║ ║╣
   ╚╩╝╚═╝═╩╝╩ ╩   ╩═╝╩═╝╩ ╩  ╚═╝╚═╝╩ ╩ ╚═╝
          Jetson Orin Optimized  |  Powered by Advantech Co. Ltd
EOF
echo -e "\033[0m"

log()      { echo -e "\033[0;36m[weda-llm-suite]\033[0m $*"; }
log_warn() { echo -e "\033[0;33m[weda-llm-suite] WARN:\033[0m $*"; }
die()      { echo -e "\033[0;31m[weda-llm-suite] FATAL:\033[0m $*" >&2; exit 1; }

MODE="run"; PROFILE_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --embed)      MODE="embed" ;;
    --shell)      MODE="shell" ;;
    --status)     MODE="status" ;;
    --logs)       MODE="logs" ;;
    --stop)       MODE="stop" ;;
    -p|--profile) PROFILE_ARG="${2:?-p needs a profile name}"; shift ;;
    -h|--help)    usage; exit 0 ;;
    *)            die "unknown flag '$1' (see ./build.sh --help)" ;;
  esac
  shift
done

# CLI beats .env (capture-before-source).
_CLI_PROFILE="${PROFILE_ARG}"
[[ -f .env ]] || die ".env missing — copy the template and set WEDA_LLM_IMAGE/HF_CACHE."
set -a; source .env; set +a
[[ -n "${_CLI_PROFILE}" ]] && export LLM_PROFILE="${_CLI_PROFILE}"
# a shell does not expand ~ inside a variable; compose would create a literal '~' dir
HF_CACHE="${HF_CACHE:-./model_cache/huggingface}"
HF_CACHE="${HF_CACHE/#\~/$HOME}"; export HF_CACHE

docker info --format '{{range $k, $v := .Runtimes}}{{$k}} {{end}}' 2>/dev/null | grep -qw nvidia \
  || log_warn "nvidia runtime not visible to docker — GPU serving will fail."
mkdir -p "${HF_CACHE}" test_logs

CTR="${WEDA_LLM_CONTAINER_NAME:-weda-llm-suite}"

# -p pins a profile for the container's lifetime: a later invocation without
# -p must not present a different LLM_PROFILE (compose would recreate the
# container on the default). Carry the running container's pin forward.
if [[ -z "${_CLI_PROFILE}" && "${MODE}" =~ ^(run|embed)$ ]]; then
  running_pin=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "${CTR}" 2>/dev/null \
    | sed -n 's/^LLM_PROFILE=//p' || true)
  [[ -n "${running_pin}" ]] && { export LLM_PROFILE="${running_pin}"; log "keeping pinned profile '${running_pin}' (started with -p)"; }
fi

# A port of the stack answered by something that is not the stack: the launcher refuses
# before starting anything, naming the holder, instead of declaring a foreign server ready.
preflight_ports() {
  local p holder
  for p in "${WEDA_LLM_PORT:-9000}" "${WEDA_EMBED_PORT:-9001}" "${WEDA_GATEWAY_PORT:-9002}" 3000; do
    holder=$(ss -ltnp 2>/dev/null | awk -v p=":${p}" '$4 ~ p"$" {print $0}')
    [[ -z "${holder}" ]] && continue
    if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qE "^(${CTR}|${CTR}-embed|${CTR}-gateway|open-webui)$"; then
      log_warn "  ${holder}"
      die "port ${p} is already served by another process (see above); stop it or change the port in .env"
    fi
  done
}

wait_ready() { # $1 = port, $2 = timeout sec, $3 = label, $4 = container to watch, $5 = log line that proves it is ours
  local t=0
  log "waiting for ${3} on :${1}..."
  until curl -fsS "http://127.0.0.1:${1}/v1/models" >/dev/null 2>&1 \
        && { [[ -z "${5:-}" ]] || docker logs "${4}" 2>&1 | grep -qF "${5}"; }; do
    sleep 5; t=$((t+5))
    [[ $t -ge $2 ]] && die "${3} not ready after ${2}s — check: docker compose logs"

    if [[ "$(docker inspect -f '{{.State.Running}}' "${4}" 2>/dev/null)" == "false" ]]; then
      docker logs --tail 30 "${4}"; die "${3} exited during startup (logs above)"
    fi
  done
  log "${3} ready."
}

# Models are downloaded by scripts/download_models.py only. Starting with a
# model that is not in the cache is refused here, before any container runs.
require_cached() { # $1 = 'generation' | 'embedding'
  local missing
  python3 -c 'import yaml' 2>/dev/null \
    || die "python3 with pyyaml is required on the host (pip install pyyaml)"
  missing=$(HF_CACHE="${HF_CACHE}" LLM_PROFILE="${LLM_PROFILE:-}" python3 - "$1" <<'PY'
import glob, os, sys, yaml
cfg = yaml.safe_load(open("config/llm_config.yaml")) or {}
cache = os.path.expanduser(os.environ["HF_CACHE"])
if sys.argv[1] == "embedding":
    e = cfg.get("embedding") or {}
    repo, fn, label = e.get("hf_repo"), e.get("gguf_file"), "embedding"
else:
    name = os.environ.get("LLM_PROFILE") or cfg.get("default_profile")
    p = (cfg.get("profiles") or {}).get(name)
    if not p:
        sys.exit(0)  # the loader reports unknown profiles with the valid list
    repo, fn, label = p["hf_repo"], p["gguf_file"], name
hit = glob.glob(os.path.join(cache, "hub", "models--" + repo.replace("/", "--"), "snapshots", "*", fn))
if not hit:
    print(f"{label}: {fn} ({repo})")
PY
  ) || die "config/llm_config.yaml could not be read -- check it parses and that the profile declares hf_repo and gguf_file"
  if [[ -n "${missing}" ]]; then
    log_warn "model not in the cache (${HF_CACHE}):"
    log_warn "  ${missing}"
    die "download it first, then start again:   python3 scripts/download_models.py"
  fi
}

case "${MODE}" in
  run)
    require_cached generation
    preflight_ports
    docker compose up -d
    wait_ready "${WEDA_LLM_PORT:-9000}" 900 "generation server" "${CTR}" "READY -- serving"
    wait_ready "${WEDA_GATEWAY_PORT:-9002}" 60 "application gateway" "${CTR}-gateway"
    log "OpenAI-compatible API: http://127.0.0.1:${WEDA_LLM_PORT:-9000}/v1"
    log "application gateway:   http://127.0.0.1:${WEDA_GATEWAY_PORT:-9002}/v1"
    log "Open WebUI:            http://<device-ip>:3000"
    log "applications:          ./app.sh        (RAG apps also need --embed)"
    log "follow logs:           ./build.sh --logs"
    ;;
  embed)
    # the embedding server is lifecycle (a second llama-server process for
    # RAG-class apps), so it starts here, not in app.sh
    require_cached generation
    require_cached embedding
    preflight_ports
    docker compose --profile rag up -d
    wait_ready "${WEDA_LLM_PORT:-9000}" 900 "generation server" "${CTR}" "READY -- serving"
    wait_ready "${WEDA_EMBED_PORT:-9001}" 600 "embedding server" "${CTR}-embed"
    log "servers up. Run an application with: ./app.sh <name>"
    ;;
  shell)
    exec docker exec -it "${CTR}" /bin/bash
    ;;
  status)
    for c in "${CTR}" "${CTR}-gateway" "${CTR}-embed" open-webui; do
      state=$(docker inspect -f '{{.State.Status}}{{if .State.Health}} ({{.State.Health.Status}}){{end}}' "$c" 2>/dev/null) \
        || state="not created"
      printf '  %-28s %s\n' "$c" "$state"
    done
    # the kernel's view: model weights and KV cache are CUDA allocations in
    # unified memory and never appear in docker stats
    awk '/MemAvailable/{a=int($2/1024)} /SwapTotal/{st=int($2/1024)} /SwapFree/{sf=int($2/1024)} END{printf "  host memory available        %d MiB   (swap in use %d / %d MiB)\n", a, st-sf, st}' /proc/meminfo
    served=$(curl -fsS "http://127.0.0.1:${WEDA_LLM_PORT:-9000}/v1/models" 2>/dev/null \
      | python3 -c "import sys,json; print((json.load(sys.stdin).get('data') or [{}])[0].get('id','?'))" 2>/dev/null \
      || true)
    if [[ -n "${served}" ]]; then
      log "model served on :${WEDA_LLM_PORT:-9000}:  ${served}"
      # the config's default is what apps without a profile key expect; a
      # mismatch means every such app is refused with a 409 until reconciled
      expected=$(python3 -c "import yaml; print((yaml.safe_load(open('config/llm_config.yaml')) or {}).get('default_profile',''))" 2>/dev/null \
        || sed -n 's/^default_profile:[[:space:]]*//p' config/llm_config.yaml | head -1)
      if [[ -n "${expected}" && "${expected}" != "${served}" ]]; then
        log_warn "config default_profile is '${expected}' but '${served}' is being served --"
        log_warn "apps following the default will refuse (409). Reconcile with one of:"
        log_warn "  ./build.sh --stop && ./build.sh          # serve the config's default"
        log_warn "  set default_profile: ${served} in config/llm_config.yaml   # keep the served model"
      fi
    else
      read -r state exit_code <<< "$(docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' "${CTR}" 2>/dev/null || echo '')"
      if [[ "${state}" == "exited" ]]; then
        log_warn "model server container EXITED (code ${exit_code}) --"
        log_warn "  last log lines:  docker logs --tail 30 ${CTR}"
        log_warn "  kernel OOM kill? dmesg -T | grep -iE 'oom|killed process' | tail -3"
        log_warn "  restart:         ./build.sh"
      elif [[ "${state}" == "running" ]] && swap_logs=$(docker logs --tail 40 "${CTR}" 2>&1) \
           && grep -qE "HOT SWAP FAILED|does not fit" <<< "${swap_logs}"; then
        log_warn "model server is IDLE after a failed hot swap:"
        grep -A1 -E "HOT SWAP FAILED|does not fit" <<< "${swap_logs}" | tail -2 | sed 's/^/         /'
        log_warn "  fix config/llm_config.yaml, then save or: touch config/llm_config.yaml"
      else
        log_warn "no model answering on :${WEDA_LLM_PORT:-9000} (loading or swapping -- READY banner in ./build.sh --logs)"
      fi
    fi
    curl -fsS "http://127.0.0.1:${WEDA_EMBED_PORT:-9001}/v1/models" >/dev/null 2>&1 \
      && log "embedding server on :${WEDA_EMBED_PORT:-9001}: up" \
      || log "embedding server on :${WEDA_EMBED_PORT:-9001}: down (RAG apps need --embed)"
    apps=$(curl -fsS "http://127.0.0.1:${WEDA_GATEWAY_PORT:-9002}/v1/models" 2>/dev/null \
      | python3 -c "import sys,json; print(', '.join(m['id'] for m in json.load(sys.stdin).get('data',[])))" 2>/dev/null \
      || true)
    if [[ -n "${apps}" ]]; then
      log "gateway on :${WEDA_GATEWAY_PORT:-9002}:        ${apps}"
    else
      log_warn "gateway not answering on :${WEDA_GATEWAY_PORT:-9002}"
    fi
    ;;
  logs)
    exec docker compose logs -f
    ;;
  stop)
    docker compose --profile rag down
    log "stopped."
    ;;
esac
