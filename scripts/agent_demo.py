#!/usr/bin/env python3
# ==============================================================================
# WEDA LLM Suite — chat/agent runtime (native OpenAI tool_calls, C++ llama-server)
# ==============================================================================
# App-driven: system prompt, tool selection and endpoints come from the
# application block in config/llm_config.yaml. kind=chat runs the same loop
# with no tools. Invoked via app.sh:
#
#   ./app.sh device-assistant                 # REPL
#   ./app.sh device-assistant "one question"  # one-shot
#
# New tools: add a function + schema pair to REGISTRY and the name to
# VALID_TOOLS in config/loader.py. Config selects tools by name only.
import ast
import json
# readline wires line editing (backspace, arrows, history) into input();
# importing it is the whole integration
try:
    import readline  # noqa: F401
except ImportError:
    pass
import operator
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.loader import AppConfig, load_app

import device_tools
import ops_memory
import docker_tools
# the one hardware-specific import: porting to different silicon means
# providing a platform_<name>.py with the same five functions
import platform_jetson as platform

MAX_TOOL_ROUNDS = 5

# ── Tool implementations (safe, local; ctx carries the app for rag tools) ─────

def tool_get_time(ctx, args):
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def tool_device_stats(ctx, args):
    out = {}
    try:
        out.update(device_tools.mem_summary_mb())
    except Exception as e:
        out["mem_error"] = str(e)
    try:
        out["loadavg_1_5_15"] = device_tools.loadavg()
    except Exception as e:
        out["load_error"] = str(e)
    try:
        cpu = device_tools.cpu_percent()
        out["cpu_percent_overall"] = cpu.get("cpu")
        out["cpu_percent_per_core"] = {k: v for k, v in cpu.items() if k != "cpu"}
    except Exception as e:
        out["cpu_error"] = str(e)
    gpu = platform.gpu_percent()
    out["gpu_percent"] = gpu if gpu is not None else "unavailable on this kernel"
    freq = platform.gpu_freq_mhz()
    out["gpu_freq_mhz"] = freq if freq is not None else "unavailable on this kernel"
    try:
        out["disk_free_gb"] = device_tools.disk_free_gb()
    except Exception:
        out["disk_free_gb"] = "unavailable"
    try:
        out["uptime_hours"] = device_tools.uptime_hours()
    except Exception:
        out["uptime_hours"] = "unavailable"
    try:
        out["thermal_c"] = device_tools.thermal_zones()
    except Exception:
        pass
    try:
        out["top_processes_by_cpu"] = device_tools.top_processes()
    except Exception:
        pass
    return json.dumps(out)


_ALLOWED_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                ast.Div: operator.truediv, ast.Mod: operator.mod,
                ast.USub: operator.neg, ast.UAdd: operator.pos}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"disallowed expression element: {ast.dump(node)[:60]}")


def tool_calculator(ctx, args):
    try:
        return str(_safe_eval(ast.parse(args.get("expression", ""), mode="eval")))
    except Exception as e:
        return f"error: {e}"


def tool_search_documents(ctx, args):
    import rag_demo
    query = args.get("query", "")
    try:
        got = rag_demo.retrieve(ctx["app"], query)
    except Exception as e:
        return (f"error: document search failed ({e}). Is the embedding server "
                "up (./build.sh --embed)?")
    if got is None:
        return (f"error: collection '{ctx['app'].collection}' is empty — "
                f"ingest first: ./app.sh docs-qa ingest")
    docs, srcs = got
    return json.dumps([{"source": s, "text": d[:600]} for s, d in zip(srcs, docs)])


def tool_get_power_modes(ctx, args):
    modes = platform.power_modes()
    _, raw = platform.current_power_mode()
    return json.dumps({
        "modes": modes or "unavailable (power-mode table not readable here)",
        "current": raw or "unavailable -- the platform power tool is not "
                          "runnable inside this container"})


def tool_set_power_mode(ctx, args):
    mode_id = str(args.get("mode_id", "")).strip()
    known = platform.power_modes()
    valid_ids = {m["id"] for m in known}
    if not valid_ids:
        return json.dumps({"error": "the power modes of this device could not be read, "
                                    "so no mode id can be verified. Set the mode from the "
                                    "host with nvpmodel -m <id>.",
                           "known_modes": known})
    if mode_id not in valid_ids:
        return json.dumps({"error": f"'{mode_id}' is not a known mode id. "
                                     f"Call get_power_modes first and use one of its ids.",
                            "known_modes": known})
    ok, detail = platform.set_power_mode(mode_id)
    ops_memory.record_action("set_power_mode", f"mode_id={mode_id}",
                             detail if ok else f"FAILED: {detail}")
    if not ok:
        return json.dumps({"error": detail})
    return json.dumps({"result": f"power mode set to id {mode_id}",
                        "verified_current": detail})


def tool_free_memory(ctx, args):
    """Flushes reclaimable OS page cache -- the same operation serve.py runs
    automatically before every model load. Does not touch the running model
    or its allocations; only reclaims cache other processes can give back."""
    before = device_tools.mem_summary_mb()["mem_available_mb"]
    ok, msg = device_tools.drop_caches()
    if not ok:
        return json.dumps({"error": f"could not flush page caches ({msg}) -- "
                                     "this container needs to run privileged"})
    after = device_tools.mem_summary_mb()["mem_available_mb"]
    ops_memory.record_action("free_memory", "drop_caches",
                             f"reclaimed {max(0, after - before)} MiB")
    return json.dumps({"result": "page caches flushed",
                        "mem_available_before_mb": before,
                        "mem_available_after_mb": after,
                        "reclaimed_mb": max(0, after - before)})


def tool_get_network_info(ctx, args):
    gen = os.environ.get("WEDA_LLM_PORT", "9000")
    emb = os.environ.get("WEDA_EMBED_PORT", "9001")
    gw = os.environ.get("WEDA_GATEWAY_PORT", "9002")
    labels = {int(gen): "generation server (internal -- use the gateway)",
              int(emb): "embedding server (internal -- use the gateway)",
              int(gw): "application gateway (the public API)",
              3000: "web UI"}
    ips = device_tools.iface_ips()
    primary = device_tools.primary_ip(ips)
    ports = device_tools.listening_ports()
    base = f"http://{primary or '<device-ip>'}:{gw}/v1"
    return json.dumps({
        "interfaces": ips or "unavailable",
        "device_ip": primary or "unavailable",
        "listening_ports": [{"port": p, "role": labels.get(p, "")} for p in ports],
        "network_mode": "host -- the stack's servers listen on the device's own "
                        "interfaces, no port mapping involved",
        "api_for_new_applications": {
            "base_url": base,
            "endpoints": [
                {"method": "GET", "path": "/models",
                 "purpose": "list the available applications (offered as models)"},
                {"method": "POST", "path": "/chat/completions",
                 "purpose": "ask an application by model name; streaming supported"}],
            "auth": "none -- any placeholder API key is accepted",
            "note": "any OpenAI-compatible client works; set model to an "
                    "application name from GET /models"}})


def tool_get_telemetry_history(ctx, args):
    hours = int(args.get("hours") or 24)
    return json.dumps(ops_memory.history(max(1, min(hours, 168))))


def tool_get_recent_alerts(ctx, args):
    hours = int(args.get("hours") or 24)
    return json.dumps(ops_memory.recent_alerts(max(1, min(hours, 168))))


def tool_get_action_history(ctx, args):
    return json.dumps(ops_memory.action_history())


def tool_check_anomalies(ctx, args):
    return json.dumps(ops_memory.anomalies())


def tool_remember_note(ctx, args):
    note = str(args.get("note", "")).strip()
    if not note:
        return json.dumps({"error": "an empty note cannot be saved"})
    return json.dumps(ops_memory.remember(note))


def tool_recall_notes(ctx, args):
    query = str(args.get("query", "")).strip()
    if not query:
        return json.dumps({"error": "give a topic to search the notes for"})
    return json.dumps(ops_memory.recall(query))


def tool_explain_recent_errors(ctx, args):
    return json.dumps(ops_memory.recent_errors())


def tool_generate_report(ctx, args):
    hours = int(args.get("hours") or 24)
    out = ops_memory.write_report(max(1, min(hours, 168)))
    ops_memory.record_action("generate_report", f"hours={hours}", out.get("path", ""))
    return json.dumps(out)


def tool_list_containers(ctx, args):
    try:
        return json.dumps(docker_tools.list_containers())
    except LookupError as e:
        return json.dumps({"error": str(e)})


def tool_inspect_container(ctx, args):
    try:
        return json.dumps(docker_tools.inspect_container(str(args.get("name", "")).strip()))
    except LookupError as e:
        return json.dumps({"error": str(e)})


def tool_container_stats(ctx, args):
    try:
        return json.dumps(docker_tools.container_stats(str(args.get("name", "")).strip()))
    except LookupError as e:
        return json.dumps({"error": str(e)})


REGISTRY = {
    "get_time": (tool_get_time, {
        "type": "function", "function": {
            "name": "get_time", "description": "Current date and time on the edge device",
            "parameters": {"type": "object", "properties": {}}}}),
    "get_device_stats": (tool_device_stats, {
        "type": "function", "function": {
            "name": "get_device_stats",
            "description": "Live system telemetry for this Jetson device: memory and swap "
                           "usage, load average, per-core and overall CPU percent, GPU percent, "
                           "thermal zones, and the top processes by CPU -- the same data as "
                           "top/htop/jtop/nvtop, in one call",
            "parameters": {"type": "object", "properties": {}}}}),
    "calculator": (tool_calculator, {
        "type": "function", "function": {
            "name": "calculator", "description": "Evaluate an arithmetic expression",
            "parameters": {"type": "object", "properties": {
                "expression": {"type": "string", "description": "e.g. (3.5 + 2) * 4"}},
                "required": ["expression"]}}}),
    "search_documents": (tool_search_documents, {
        "type": "function", "function": {
            "name": "search_documents",
            "description": "Search the local document knowledge base; returns the most "
                           "relevant snippets with their sources",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "what to look for"}},
                "required": ["query"]}}}),
    "get_power_modes": (tool_get_power_modes, {
        "type": "function", "function": {
            "name": "get_power_modes",
            "description": "List the Jetson power modes available on this device and which "
                           "one is currently active. Call this before set_power_mode.",
            "parameters": {"type": "object", "properties": {}}}}),
    "set_power_mode": (tool_set_power_mode, {
        "type": "function", "function": {
            "name": "set_power_mode",
            "description": "Switch the Jetson to a different power mode by id (from "
                           "get_power_modes). Takes effect immediately; changes CPU/GPU clock "
                           "ceilings and power draw, not a reboot.",
            "parameters": {"type": "object", "properties": {
                "mode_id": {"type": "string", "description": "the power mode id, e.g. '0'"}},
                "required": ["mode_id"]}}}),
    "free_memory": (tool_free_memory, {
        "type": "function", "function": {
            "name": "free_memory",
            "description": "Flush reclaimable OS page cache to free up RAM before starting a "
                           "large workload. Does not affect the currently running model or any "
                           "other process's actual memory.",
            "parameters": {"type": "object", "properties": {}}}}),
    "get_network_info": (tool_get_network_info, {
        "type": "function", "function": {
            "name": "get_network_info",
            "description": "Device IP addresses, listening TCP ports with their roles, the "
                           "networking mode, and the exact API base URL and endpoints a new "
                           "application (or an OpenAPI/Swagger spec) should target.",
            "parameters": {"type": "object", "properties": {}}}}),
    "get_telemetry_history": (tool_get_telemetry_history, {
        "type": "function", "function": {
            "name": "get_telemetry_history",
            "description": "Recorded history of temperature, CPU/GPU load, memory, swap and "
                           "disk over the last hours: min/avg/max, latest, and when each "
                           "metric peaked. Use for any question about the past ('overnight', "
                           "'earlier', 'while I was away').",
            "parameters": {"type": "object", "properties": {
                "hours": {"type": "integer", "description": "look-back window, default 24, max 168"}}}}}),
    "get_recent_alerts": (tool_get_recent_alerts, {
        "type": "function", "function": {
            "name": "get_recent_alerts",
            "description": "Threshold breaches (temperature, memory, swap, disk) raised and "
                           "cleared by the monitoring watchdog, plus which alerts are active "
                           "right now.",
            "parameters": {"type": "object", "properties": {
                "hours": {"type": "integer", "description": "look-back window, default 24"}}}}}),
    "get_action_history": (tool_get_action_history, {
        "type": "function", "function": {
            "name": "get_action_history",
            "description": "The audit journal: every device-changing action taken through "
                           "this assistant (power-mode switches, memory frees), with time "
                           "and result.",
            "parameters": {"type": "object", "properties": {}}}}),
    "check_anomalies": (tool_check_anomalies, {
        "type": "function", "function": {
            "name": "check_anomalies",
            "description": "Statistical check of the recent telemetry against the 7-day "
                           "baseline (z-score) -- reports metrics behaving unusually, with "
                           "the numbers.",
            "parameters": {"type": "object", "properties": {}}}}),
    "remember_note": (tool_remember_note, {
        "type": "function", "function": {
            "name": "remember_note",
            "description": "Save an operator note for later shifts (observations, handover "
                           "items). Notes persist across restarts and are searchable.",
            "parameters": {"type": "object", "properties": {
                "note": {"type": "string", "description": "the note text to save"}},
                "required": ["note"]}}}),
    "recall_notes": (tool_recall_notes, {
        "type": "function", "function": {
            "name": "recall_notes",
            "description": "Search previously saved operator notes by topic.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "what to look for"}},
                "required": ["query"]}}}),
    "explain_recent_errors": (tool_explain_recent_errors, {
        "type": "function", "function": {
            "name": "explain_recent_errors",
            "description": "Read the docker logs / container logs of this stack's own "
                           "containers and return recent lines, each tagged level "
                           "'error' or 'warning'. Report each at its stated level; "
                           "warnings are usually harmless. Use for any request to check "
                           "the logs, check the docker logs, see container errors, or "
                           "answer 'why did the model restart' / 'what went wrong'.",
            "parameters": {"type": "object", "properties": {}}}}),
    "generate_report": (tool_generate_report, {
        "type": "function", "function": {
            "name": "generate_report",
            "description": "Write a device health report (telemetry, alerts, actions, "
                           "anomalies) as a markdown file and return its path and summary.",
            "parameters": {"type": "object", "properties": {
                "hours": {"type": "integer", "description": "window, default 24"}}}}}),
    "list_containers": (tool_list_containers, {
        "type": "function", "function": {
            "name": "list_containers",
            "description": "List all docker containers on this device (running and stopped) "
                           "with name, id, image and state -- the docker ps -a view, plus a "
                           "running count. Read-only.",
            "parameters": {"type": "object", "properties": {}}}}),
    "inspect_container": (tool_inspect_container, {
        "type": "function", "function": {
            "name": "inspect_container",
            "description": "Details of one docker container: image, state, networking mode "
                           "and IPs, restart policy and count, OOM/exit status. Read-only.",
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string", "description": "container name or id"}},
                "required": ["name"]}}}),
    "container_stats": (tool_container_stats, {
        "type": "function", "function": {
            "name": "container_stats",
            "description": "Live docker stats of one container: CPU percent, memory usage "
                           "and pid count. Read-only.",
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string", "description": "container name or id"}},
                "required": ["name"]}}}),
}


def system_prompt_for(app):
    """The app's system prompt, carrying Qwen's /no_think soft switch when
    thinking is off and the profile runs the Python backend -- that backend
    ignores chat_template_kwargs, the template-level switch the C++ server
    honors."""
    prompt = app.system_prompt or ""
    if not app.thinking:
        try:
            from config.loader import load_llm_config
            if load_llm_config(app.profile).backend == "python":
                prompt = (prompt + "\n/no_think").strip()
        except Exception:
            pass
    return prompt


def error_body(err):
    """The upstream error body, decoded once and kept on the exception (an
    HTTPError body can only be read once)."""
    if not hasattr(err, "weda_body"):
        try:
            err.weda_body = err.read().decode(errors="replace")
        except Exception:
            err.weda_body = ""
    return err.weda_body


def _context_overflow(err):
    if err.code != 400:
        return False
    try:
        return json.loads(error_body(err)).get("error", {}).get("type") == "exceed_context_size_error"
    except Exception:
        return False


def trim_oldest(messages):
    """Drop the oldest turn as a unit: a user message with the assistant reply
    and tool results that followed it, up to the next user message. Returns
    None when only the current turn is left."""
    keep_from = 1 if messages and messages[0].get("role") == "system" else 0
    body = messages[keep_from:]
    i = 1
    while i < len(body) and body[i].get("role") != "user":
        i += 1
    if i >= len(body):
        return None
    return messages[:keep_from] + body[i:]


def post_chat(base, payload):
    """POST /chat/completions; on a context-size rejection, drop the oldest
    turns and retry so a long conversation keeps going -- the request-layer
    equivalent of context shift, which only covers generation overflow."""
    msgs = list(payload.get("messages") or [])
    while True:
        req = urllib.request.Request(base + "/chat/completions",
                                     data=json.dumps({**payload, "messages": msgs}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(req, timeout=300), msgs
        except urllib.error.HTTPError as e:
            if not _context_overflow(e):
                raise
            trimmed = trim_oldest(msgs)
            if trimmed is None:
                raise
            print(f"[gateway] context overflow: dropping oldest turn "
                  f"({len(msgs)} -> {len(trimmed)} messages) and retrying")
            msgs = trimmed


def upstream_error(e):
    """Human-readable text for an HTTPError from the model server."""
    body = error_body(e)
    try:
        detail = json.loads(body).get("error", {}).get("message") or e.reason
    except Exception:
        detail = body[:300] or e.reason
    return f"model server returned {e.code}: {detail}"


def chat_body(app, max_tokens=512):
    """The request-body base every app shares: sampling and the per-app
    thinking switch."""
    body = {"model": "weda", "max_tokens": max_tokens}
    body.update(app.sampling or {})
    if not app.thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


def chat(ctx, messages):
    body = chat_body(ctx["app"], max_tokens=1024)
    body["messages"] = messages
    if ctx["tools"]:
        body["tools"] = [REGISTRY[t][1] for t in ctx["tools"]]
        body["tool_choice"] = "auto"
    resp, used = post_chat(ctx["base"], body)
    if len(used) != len(messages):
        # keep the REPL's history in step with what the server accepted
        messages[:] = used
    with resp:
        return json.loads(resp.read())


def tool_schema_echo(content):
    """A model too small for native tool calling regurgitates the tool
    schemas as text instead of emitting tool_calls -- detect that so the
    user gets an explanation, never the schema dump."""
    c = content or ""
    return "<tool_call>" in c or ("<tool" in c and '"function"' in c)


ECHO_EXPLANATION = (
    "The served model echoed the tool schema as text instead of calling a "
    "tool -- it is too small for native tool calling. Swap to a tool-capable "
    "profile (e.g. the default qwen3-4b-q4km) to use agent applications.")


def dispatch_tool_calls(ctx, calls, msgs, trace=None):
    """Execute each tool_call against REGISTRY and append the tool messages --
    the one dispatcher shared by the terminal loop and the gateway."""
    for c in calls:
        fn = c["function"]["name"]
        try:
            fargs = json.loads(c["function"].get("arguments") or "{}")
        except json.JSONDecodeError:
            fargs = {}
        impl = REGISTRY.get(fn)
        result = impl[0](ctx, fargs) if impl else f"error: unknown tool {fn}"
        if trace:
            trace(fn, fargs, result)
        msgs.append({"role": "tool", "tool_call_id": c["id"],
                     "name": fn, "content": str(result)})


def run_turn(ctx, messages):
    for _ in range(MAX_TOOL_ROUNDS):
        msg = chat(ctx, messages)["choices"][0]["message"]
        calls = msg.get("tool_calls")
        if not calls:
            content = msg.get("content") or "(empty response)"
            if ctx.get("tools") and tool_schema_echo(content):
                return "error: " + ECHO_EXPLANATION
            return content
        messages.append(msg)
        dispatch_tool_calls(ctx, calls, messages,
                            trace=lambda fn, fargs, result: print(
                                f"  [tool] {fn}({fargs}) -> {str(result)[:120]}"))
    return "(stopped: exceeded max tool rounds)"


def served_profile(base):
    try:
        with urllib.request.urlopen(base + "/models", timeout=5) as r:
            data = json.loads(r.read())
        return (data.get("data") or [{}])[0].get("id", "")
    except Exception:
        return ""


def unavailable_reason(app, served):
    """Why `app` cannot run on the served profile, or None. Mutates an unpinned
    app to follow the served profile."""
    if not served:
        return None
    if not app.pinned:
        app.profile = served
    elif app.profile != served:
        return (f"application '{app.name}' expects profile '{app.profile}' but the "
                f"server is serving '{served}'")
    if app.kind == "agent":
        try:
            from config.loader import load_llm_config
            p = load_llm_config(served)
            if p.backend == "python":
                return (f"application '{app.name}' needs tool calls, which the Python "
                        f"backend profile '{served}' cannot emit")
            if not p.tool_calls:
                return (f"application '{app.name}' needs tool calls, and profile "
                        f"'{served}' is marked tool_calls: false (a model this "
                        f"size echoes tool schemas as text instead of calling "
                        f"them) -- swap to a tool-capable profile")
        except Exception:
            pass
    return None


def check_profile(app):
    served = served_profile(app.generation_base())
    reason = unavailable_reason(app, served)
    if reason and "expects profile" in reason:
        sys.exit(f"FATAL: {reason}.\n"
                 f"       Either restart the server (./build.sh --stop && "
                 f"./build.sh -p {app.profile})\n"
                 f"       or point the app's profile key at '{served}' in "
                 "config/llm_config.yaml.")
    if reason:
        sys.exit(f"FATAL: {reason}.\n"
                 "       Serve a C++-backend profile: ./build.sh --stop && ./build.sh")


def preflight(ctx):
    app = ctx["app"]
    try:
        urllib.request.urlopen(f"{ctx['base']}/models", timeout=5).read()
    except Exception:
        sys.exit(f"FATAL: no generation server at {ctx['base']}.\n"
                 "       app.sh never starts servers — run: ./build.sh")
    check_profile(app)
    if app.needs_embeddings():
        try:
            urllib.request.urlopen(app.embeddings_base() + "/models", timeout=5).read()
        except Exception:
            sys.exit(f"FATAL: '{app.name}' uses search_documents but the embedding "
                     f"server is not reachable at {app.embeddings_base()}.\n"
                     "       Run: ./build.sh --embed")
    if ctx["tools"]:
        probe = chat(ctx, [{"role": "user",
                            "content": "What time is it? Use the tool."}])
        msg = probe["choices"][0]["message"]
        if not msg.get("tool_calls") and tool_schema_echo(msg.get("content")):
            sys.exit("FATAL: server returned the tool call as TEXT, not tool_calls —\n"
                     "       the running server is the python backend. Restart with a\n"
                     "       backend: llamacpp profile: ./build.sh --stop && ./build.sh")


def run(app: AppConfig, argv):
    ctx = {"app": app, "base": app.generation_base(), "tools": list(app.tools or [])}
    print("======================================================================")
    print(f"  app '{app.name}' — kind {app.kind}, profile {app.profile}"
          + (f", tools: {', '.join(ctx['tools'])}" if ctx["tools"] else ""))
    print("======================================================================")
    preflight(ctx)
    system = {"role": "system", "content": system_prompt_for(app) or "Be concise."}
    if argv:
        try:
            print(run_turn(ctx, [system, {"role": "user", "content": " ".join(argv)}]))
        except urllib.error.HTTPError as e:
            sys.exit(f"[error] {upstream_error(e)}")
        return
    msgs = [system]
    prompt = "agent> " if ctx["tools"] else "chat> "
    while True:
        try:
            q = input(f"\n{prompt}").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not q or q in ("exit", "quit"):
            break
        msgs.append({"role": "user", "content": q})
        try:
            answer = run_turn(ctx, msgs)
        except urllib.error.HTTPError as e:
            msgs.pop()
            print(f"[error] {upstream_error(e)}")
            continue
        msgs.append({"role": "assistant", "content": answer})
        print(answer)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: agent_demo.py <app-name> [one-shot question]")
    run(load_app(sys.argv[1]), sys.argv[2:])
