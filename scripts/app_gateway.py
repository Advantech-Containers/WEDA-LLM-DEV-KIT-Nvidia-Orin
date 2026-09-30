#!/usr/bin/env python3
# ==============================================================================
# WEDA LLM Suite — application gateway (apps as models for Web UI clients)
# ==============================================================================
# OpenAI-compatible endpoint on WEDA_GATEWAY_PORT. Every application from
# config/llm_config.yaml is listed as a model; selecting one in Open WebUI (or
# any OpenAI client) runs that application server-side:
#   chat  -> app system prompt injected, forwarded (streaming passthrough)
#   rag   -> retrieval + cited context built per request, then forwarded
#   agent -> tool rounds executed here, final answer returned (streamed as a
#            single chunk when the client asked for streaming)
# Sampling parameters sent by the client (temperature, top_p, ...) pass
# through untouched, so UI-side controls keep working.
import http.client
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.loader import list_apps, load_app

import agent_demo
import rag_demo

PORT = int(os.environ.get("WEDA_GATEWAY_PORT", "9002"))
HOST = os.environ.get("WEDA_LLM_HOST", "127.0.0.1")
ALLOWED_ORIGINS = tuple(o.strip() for o in
                        os.environ.get("WEDA_GATEWAY_ALLOWED_ORIGINS", "").split(",")
                        if o.strip())
# generous for chat + base64 images, small enough that a runaway client
# cannot fill 8 GB of unified memory through one request body
MAX_BODY = 32 * 1024 * 1024
# socket timeout per connection: a client that declares a Content-Length it
# never sends would otherwise park a handler thread forever
CONN_TIMEOUT = 300


def allowed_origin(origin):
    """Return the origin to echo back, or None when it is not allowed.
    No origin is allowed unless WEDA_GATEWAY_ALLOWED_ORIGINS names it."""
    if not origin or not ALLOWED_ORIGINS:
        return None
    if origin in ALLOWED_ORIGINS or "*" in ALLOWED_ORIGINS:
        return origin
    return None

_served_cache = {"base": None, "id": "", "t": 0.0}
_served_lock = threading.Lock()


def served_profile(base, max_age=5.0):
    with _served_lock:
        if _served_cache["base"] == base and time.time() - _served_cache["t"] < max_age:
            return _served_cache["id"]
    sid = agent_demo.served_profile(base)
    with _served_lock:
        _served_cache.update(base=base, id=sid, t=time.time())
    return sid


def openai_error(code, message):
    return code, {"error": {"message": message, "type": "invalid_request_error"}}


def apply_system_prompt(app, messages):
    prompt = agent_demo.system_prompt_for(app)
    if not prompt:
        return messages
    ours = {"role": "system", "content": prompt}
    rest = [m for m in messages if m.get("role") != "system"] \
        if messages and messages[0].get("role") == "system" else messages
    return [ours] + rest


def _ocr_data_url(url):
    """Text read from a base64 data: image, or None when OCR is unavailable."""
    import base64
    import shutil
    import subprocess
    import tempfile
    if not shutil.which("tesseract"):
        return None
    header, _, payload = url.partition(",")
    ext = ".png" if "png" in header else ".jpg"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=True) as f:
        f.write(base64.b64decode(payload))
        f.flush()
        r = subprocess.run(["tesseract", f.name, "-", "--psm", "3"],
                           capture_output=True, text=True, timeout=120)
    return r.stdout.strip()


def images_to_text(messages):
    """The served models are text-only: every image part in the conversation
    is replaced by the text OCR reads off it (screenshots, labels, scans).
    Raises ValueError naming the limitation when OCR is not available."""
    out = []
    for m in messages:
        c = m.get("content")
        if not isinstance(c, list):
            out.append(m)
            continue
        parts = []
        for part in c:
            if not (isinstance(part, dict) and part.get("type") == "image_url"):
                parts.append(part)
                continue
            url = (part.get("image_url") or {}).get("url", "")
            if not url.startswith("data:image/"):
                raise ValueError("only attached images are supported (not remote image URLs)")
            text = _ocr_data_url(url)
            if text is None:
                raise ValueError("this deployment serves a text model: images cannot be "
                                 "viewed, and OCR is not available in this image -- "
                                 "paste the text, or attach the image to a RAG corpus")
            parts.append({"type": "text",
                          "text": "[text read from the attached image]\n" + (text or "(no readable text)")})
        out.append({**m, "content": parts})
    return out


def flatten_content(messages):
    """Content given as a list of text parts (the multimodal message form
    Open WebUI sends) becomes one string: the C++ server accepts both, the
    Python backend's chat template renders strings only."""
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            text = "\n".join(p.get("text", "") for p in c
                             if isinstance(p, dict) and p.get("type") == "text")
            out.append({**m, "content": text})
        else:
            out.append(m)
    return out


def _user_text(messages, nth_last):
    seen = 0
    for m in reversed(messages):
        if m.get("role") == "user":
            seen += 1
            if seen == nth_last:
                c = m.get("content")
                if isinstance(c, list):
                    return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
                return c or ""
    return ""


def last_user_text(messages):
    return _user_text(messages, 1)


def previous_user_text(messages):
    """The user turn before the last one, for conversation-aware retrieval."""
    return _user_text(messages, 2)


def build_rag_messages(app, messages):
    question = last_user_text(messages)
    got = rag_demo.retrieve(app, question, previous_user_text(messages))
    if got is None:
        raise LookupError(
            f"collection '{app.collection}' is empty — ingest first: "
            f"./app.sh {app.name} ingest")
    docs, srcs = got
    out = apply_system_prompt(app, messages)
    out = out[:-1] + [{"role": "user",
                       "content": rag_demo.rag_user_message(docs, srcs, question)}]
    return out, sorted(set(srcs))


def run_agent(app, messages, body):
    ctx = {"app": app, "base": app.generation_base(), "tools": list(app.tools or [])}
    msgs = apply_system_prompt(app, list(messages))
    data = None
    for _ in range(agent_demo.MAX_TOOL_ROUNDS):
        req = {k: v for k, v in body.items()
               if k in ("temperature", "top_p", "top_k", "min_p", "max_tokens",
                        "repeat_penalty")}
        req.update({"model": "weda", "messages": msgs})
        if not app.thinking:
            req["chat_template_kwargs"] = {"enable_thinking": False}
        if ctx["tools"]:
            req["tools"] = [agent_demo.REGISTRY[t][1] for t in ctx["tools"]]
            req["tool_choice"] = "auto"
        resp, msgs = agent_demo.post_chat(ctx["base"], req)
        with resp:
            data = json.loads(resp.read())
        msg = data["choices"][0]["message"]
        calls = msg.get("tool_calls")
        if not calls:
            if ctx["tools"] and agent_demo.tool_schema_echo(msg.get("content")):
                msg["content"] = agent_demo.ECHO_EXPLANATION
            return data
        msgs.append(msg)
        agent_demo.dispatch_tool_calls(ctx, calls, msgs)
    if data is None:
        raise RuntimeError("no tool round ran: agent_demo.MAX_TOOL_ROUNDS must be at least 1")
    data["choices"][0]["message"]["content"] = "(stopped: exceeded max tool rounds)"
    return data


class Handler(BaseHTTPRequestHandler):
    timeout = CONN_TIMEOUT
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print(f"[gateway] {fmt % args}")

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_as_stream(self, data, model):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        # no Content-Length on a stream: close delimits the body for
        # keep-alive HTTP/1.1 clients
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        msg = data["choices"][0]["message"]
        chunk = {"id": data.get("id", "gw"), "object": "chat.completion.chunk",
                 "model": model, "choices": [{"index": 0, "delta":
                     {"role": "assistant", "content": msg.get("content") or ""},
                     "finish_reason": None}]}
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        done = {"id": data.get("id", "gw"), "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        self.wfile.write(f"data: {json.dumps(done)}\n\ndata: [DONE]\n\n".encode())

    def _forward(self, base, body, stream):
        resp, _ = agent_demo.post_chat(base, body)
        self.send_response(200)
        ctype = resp.headers.get("Content-Type", "application/json")
        self.send_header("Content-Type", ctype)
        if stream:
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()
            try:
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except http.client.IncompleteRead as e:
                # the model server closed the stream early: deliver what it
                # produced and end the stream cleanly so the client keeps the
                # partial answer instead of discarding it as an error
                if e.partial:
                    self.wfile.write(e.partial)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                print("[gateway] WARN: upstream closed the stream early "
                      f"({len(e.partial)} trailing bytes) -- check the model server log")
        else:
            data = resp.read()
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    def _cors(self):
        self.send_header("Vary", "Origin")
        origin = allowed_origin(self.headers.get("Origin"))
        if not origin:
            return
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        # browser clients preflight cross-origin POSTs
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        try:
            self._get()
        except Exception as e:
            # a broken config must surface as an error, never as a silently
            # empty model list / closed socket; detail goes to the log only
            print(f"[gateway] ERROR on {self.path}: {e!r}", flush=True)
            self._send_json(*openai_error(500, "internal gateway error -- "
                                               "see ./build.sh --logs"))

    def _get(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            models = []
            served = ""
            for a in list_apps():
                try:
                    app = load_app(a)
                except Exception as e:
                    # one invalid block must not hide every other app
                    models.append({"id": a, "object": "model", "owned_by": "weda",
                                   "name": f"{a} -- CONFIG ERROR: {e}"})
                    continue
                if not served:
                    served = served_profile(app.generation_base())
                # an app that cannot answer on the served profile (pinned
                # elsewhere, or an agent on the Python backend) is left out so
                # the UI offers nothing that would only refuse. The index page
                # (/) still lists it with the reason.
                if agent_demo.unavailable_reason(app, served):
                    continue
                # "name" is display metadata (Open WebUI shows it); "id" stays
                # the request key
                models.append({"id": a, "object": "model", "owned_by": "weda",
                               "name": f"{a} ({app.profile})"})
            self._send_json(200, {"object": "list", "data": models})
        elif self.path == "/health":
            self._send_json(200, {"status": "ok"})
        elif self.path == "/":
            apps = []
            served_now = ""
            for a in list_apps():
                try:
                    app = load_app(a)
                except Exception as e:
                    apps.append({"name": a, "error": str(e)})
                    continue
                if not served_now:
                    served_now = served_profile(app.generation_base())
                why = agent_demo.unavailable_reason(app, served_now)
                apps.append({"name": a, "kind": app.kind, "profile": app.profile,
                             "pinned": app.pinned,
                             "available": why is None, "unavailable_reason": why,
                             "tools": app.tools, "examples": app.examples,
                             "curl": (f"curl http://127.0.0.1:{PORT}/v1/chat/completions "
                                      f"-H 'Content-Type: application/json' -d "
                                      f'\'{{"model":"{a}","messages":[{{"role":"user",'
                                      f'"content":"hello"}}]}}\'')})
            self._send_json(200, {
                "service": "WEDA-LLM application gateway",
                "serving": served_now,
                "endpoints": ["/v1/models", "/v1/chat/completions"],
                "note": "applications with available=false are pinned to a profile "
                        "that is not being served and are hidden from /v1/models",
                "applications": apps})
        else:
            self._send_json(*openai_error(404, f"unknown path {self.path}"))

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self._send_json(*openai_error(404, f"unknown path {self.path}"))
            return
        try:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                self._send_json(*openai_error(400, "Content-Length is not a number"))
                return
            if length <= 0:
                self._send_json(*openai_error(411, "Content-Length required"))
                return
            if length > MAX_BODY:
                self._send_json(*openai_error(
                    413, f"request body {length} bytes exceeds the "
                         f"{MAX_BODY // (1024*1024)} MiB limit"))
                return
            raw = self.rfile.read(length)
            try:
                body = json.loads(raw)
            except ValueError as e:
                self._send_json(*openai_error(400, f"request body is not valid JSON: {e}"))
                return
            if not isinstance(body, dict):
                self._send_json(*openai_error(400, "request body must be a JSON object"))
                return
            name = body.get("model", "")
            try:
                app = load_app(name)
            except (KeyError, ValueError) as e:
                self._send_json(*openai_error(404, str(e)))
                return
            if app.needs_embeddings():
                try:
                    with urllib.request.urlopen(app.embeddings_base() + "/models",
                                                timeout=3):
                        pass
                except Exception:
                    self._send_json(*openai_error(
                        503, f"application '{name}' needs the embedding server "
                             "and it is not running -- start it with "
                             "./build.sh --embed (first use also needs "
                             f"./app.sh {name} ingest)"))
                    return
            served = served_profile(app.generation_base())
            why = agent_demo.unavailable_reason(app, served)
            if why:
                self._send_json(*openai_error(
                    409, f"{why}. Serve the right profile (./build.sh --stop && "
                         f"./build.sh -p <profile>) or adjust the app's profile key."))
                return
            stream = bool(body.get("stream"))
            for k, v in (app.sampling or {}).items():
                body.setdefault(k, v)
            try:
                messages = flatten_content(images_to_text(body.get("messages") or []))
            except ValueError as e:
                self._send_json(*openai_error(400, str(e)))
                return

            if app.kind == "agent":
                data = run_agent(app, messages, body)
                data["model"] = name
                if stream:
                    self._send_as_stream(data, name)
                else:
                    self._send_json(200, data)
                return

            if app.kind == "rag-verified":
                import rag_verified
                try:
                    result = rag_verified.answer(app, last_user_text(messages),
                                                 previous_user_text(messages),
                                                 trace=lambda m: print(f"[gateway][{name}] {m}"))
                except LookupError as e:
                    self._send_json(*openai_error(400, str(e)))
                    return
                text = result["answer"]
                if result["sources"]:
                    text += "\n\nSources: " + ", ".join(result["sources"])
                data = {"id": "gw-verified", "object": "chat.completion", "model": name,
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": text}}]}
                if stream:
                    self._send_as_stream(data, name)
                else:
                    self._send_json(200, data)
                return

            fwd = dict(body)
            fwd["model"] = "weda"
            # only the agent kind executes tools (from its own registry);
            # client-sent tool specs on chat/rag would be rendered into the
            # prompt by --jinja (thousands of tokens) and any resulting tool
            # call would never be executed -- drop them
            for k in ("tools", "tool_choice", "functions", "function_call"):
                fwd.pop(k, None)
            if not app.thinking:
                fwd["chat_template_kwargs"] = {"enable_thinking": False}
            if app.kind == "rag":
                try:
                    fwd["messages"], sources = build_rag_messages(app, messages)
                except LookupError as e:
                    self._send_json(*openai_error(400, str(e)))
                    return
            else:
                fwd["messages"] = apply_system_prompt(app, messages)
            self._forward(app.generation_base(), fwd, stream)
        except urllib.error.HTTPError as e:
            # surface the upstream server's own error body, not just the code
            detail = agent_demo.error_body(e)[:500]
            # per-message shape of the rejected request, for oversized-prompt
            # diagnosis (which message carried the bulk)
            try:
                for i, m in enumerate(body.get("messages") or []):
                    c = m.get("content")
                    ln = len(c) if isinstance(c, str) else len(json.dumps(c or ""))
                    print(f"[gateway] rejected msg[{i}] role={m.get('role')} chars={ln}")
                extras = sorted(set(body) - {"messages", "model", "stream"})
                print(f"[gateway] rejected request extra keys: {extras}")
            except Exception:
                pass
            try:
                self._send_json(*openai_error(
                    502, f"upstream {e.code} from the model server: {detail or e.reason}"))
            except Exception:
                pass
        except urllib.error.URLError as e:
            try:
                if isinstance(getattr(e, "reason", None), socket.timeout):
                    # the server answered nothing for 300 s: queued behind a
                    # busy slot or a very long generation, not down
                    self._send_json(*openai_error(
                        504, "the model server did not answer within 300 s -- it "
                             "is busy (requests queue per slot; check "
                             "./build.sh --status and the profile's parallel "
                             "setting) or generating a very long reply"))
                else:
                    # nothing listening: the serve supervisor's status file
                    # (shared test_logs mount) says whether this is a load in
                    # progress or a failed swap -- and why
                    state, reason = "", ""
                    try:
                        with open("/wise-edge/app/test_logs/serve_status.json",
                                  encoding="utf-8") as f:
                            st = json.load(f)
                        state, reason = st.get("state", ""), st.get("reason", "")
                    except (OSError, ValueError):
                        pass
                    if state == "idle":
                        self._send_json(*openai_error(
                            503, f"the model server is idle after a failed swap: "
                                 f"{reason} -- fix config/llm_config.yaml and "
                                 "save it to retry"))
                    elif state == "loading":
                        self._send_json(*openai_error(
                            503, f"model '{reason}' is loading (swap or startup "
                                 "in progress) -- retry after the READY banner"))
                    else:
                        self._send_json(*openai_error(
                            503, "the model server is not reachable -- a model swap or "
                                 "first-use download is likely in progress; "
                                 "./build.sh --status shows the state; retry after "
                                 "the READY banner"))
            except Exception:
                pass
        except socket.timeout:
            try:
                self._send_json(*openai_error(
                    504, "the model server stopped responding mid-request "
                         "(300 s); check ./build.sh --status"))
            except Exception:
                pass
        except Exception as e:
            print(f"[gateway] ERROR on {self.path}: {e!r}", flush=True)
            try:
                self._send_json(*openai_error(500, "internal gateway error -- "
                                                   "see ./build.sh --logs"))
            except Exception:
                pass


def main():
    apps = list_apps()
    print(f"[gateway] serving {len(apps)} applications as models on "
          f"http://{HOST}:{PORT}/v1: {', '.join(apps)}")
    # the operations-memory recorder lives with the gateway (the always-up
    # service): one thread samples telemetry and runs the alert watchdog, so
    # the agent's history/alert/anomaly tools have data to read
    try:
        import ops_memory
        from config.loader import load_monitoring_config
        ops_memory.start_recorder(load_monitoring_config())
    except Exception as e:
        print(f"[gateway] ops-memory recorder not started: {e!r}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
