#!/usr/bin/env python3
# ==============================================================================
# WEDA LLM Suite — operations memory (recorder, watchdog, journal, notes)
# ==============================================================================
# The agent's memory of time. A deterministic recorder thread (started by the
# gateway) samples telemetry into a ring, checks thresholds with hysteresis,
# and appends alerts; acting tools append to an audit journal. The agent only
# READS through the query functions here -- recorders write, tools witness.
# Storage lives on the shared test_logs mount so it survives restarts and is
# visible to every service.
import json
import os
import re
import threading
import time

import device_tools
try:
    import platform_jetson as platform
except ImportError:
    platform = None

# Paths resolve OPS_DIR at call time, not import time, so a runtime override
# (and test isolation) actually takes effect.
def _dir():
    return os.environ.get("OPS_DIR", "/wise-edge/app/test_logs/ops")


def _p(name):
    return os.path.join(_dir(), name)


# module-level aliases kept as functions -> callers use TELEMETRY() etc.
def TELEMETRY():
    return _p("telemetry.jsonl")


def ALERTS():
    return _p("alerts.jsonl")


def ACTIONS():
    return _p("actions.jsonl")


def NOTES():
    return _p("notes.jsonl")


def REPORTS():
    return _p("reports")

# ring cap: at the 30 s default interval this holds ~7 days
MAX_TELEMETRY_LINES = 20160

DEFAULTS = {
    "interval_sec": 30,
    "thresholds": {
        "temp_c": 85.0,
        "mem_used_pct": 92.0,
        "swap_used_pct": 80.0,
        "disk_free_gb_min": 2.0,
    },
}

METRIC_KEYS = ("temp_c", "cpu_pct", "gpu_pct", "mem_used_mb",
               "swap_used_mb", "disk_free_gb")

_lock = threading.Lock()
_active_alerts = {}


def _ensure_dirs():
    os.makedirs(REPORTS(), exist_ok=True)


def _append(path, obj):
    with _lock:
        _ensure_dirs()
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")


def _read_jsonl(path, since_ts=None, limit=None):
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return []
    if since_ts is not None:
        rows = [r for r in rows if r.get("ts", 0) >= since_ts]
    return rows[-limit:] if limit else rows


def _trim_ring():
    try:
        with open(TELEMETRY(), encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) > MAX_TELEMETRY_LINES * 1.2:
            with _lock:
                with open(TELEMETRY(), "w", encoding="utf-8") as f:
                    f.writelines(lines[-MAX_TELEMETRY_LINES:])
    except OSError:
        pass


def _sample():
    s = {"ts": time.time()}
    try:
        s.update(device_tools.mem_summary_mb())
    except Exception:
        pass
    try:
        s["cpu_pct"] = device_tools.cpu_percent(sample_sec=0.1).get("cpu")
    except Exception:
        pass
    try:
        zones = device_tools.thermal_zones()
        s["temp_c"] = max(zones) if zones else None
    except Exception:
        pass
    try:
        s["disk_free_gb"] = device_tools.disk_free_gb()
    except Exception:
        pass
    if platform:
        s["gpu_pct"] = platform.gpu_percent()
    return s


def _check_thresholds(s, th):
    """Alert on breach, clear with 5% hysteresis, never re-alert while a
    condition persists."""
    checks = []
    if s.get("temp_c") is not None:
        checks.append(("temp_c", s["temp_c"], th["temp_c"], "above",
                       f"temperature {s['temp_c']:.1f} C over {th['temp_c']} C"))
    if s.get("mem_total_mb"):
        pct = 100.0 * s.get("mem_used_mb", 0) / s["mem_total_mb"]
        checks.append(("mem_used_pct", pct, th["mem_used_pct"], "above",
                       f"memory {pct:.0f}% used, over {th['mem_used_pct']}%"))
    if s.get("swap_total_mb"):
        pct = 100.0 * s.get("swap_used_mb", 0) / s["swap_total_mb"]
        checks.append(("swap_used_pct", pct, th["swap_used_pct"], "above",
                       f"swap {pct:.0f}% used, over {th['swap_used_pct']}%"))
    if s.get("disk_free_gb") is not None:
        checks.append(("disk_free_gb_min", s["disk_free_gb"], th["disk_free_gb_min"],
                       "below", f"disk down to {s['disk_free_gb']} GB free"))
    for key, value, limit, direction, message in checks:
        breached = value > limit if direction == "above" else value < limit
        cleared = (value < limit * 0.95) if direction == "above" else (value > limit * 1.05)
        if breached and key not in _active_alerts:
            _active_alerts[key] = s["ts"]
            _append(ALERTS(), {"ts": s["ts"], "kind": key, "value": round(value, 1),
                             "threshold": limit, "message": message, "state": "raised"})
        elif key in _active_alerts and cleared:
            _append(ALERTS(), {"ts": s["ts"], "kind": key, "value": round(value, 1),
                             "threshold": limit, "state": "cleared",
                             "message": f"{key} back within limits"})
            del _active_alerts[key]


def start_recorder(config=None):
    cfg = dict(DEFAULTS)
    th = dict(DEFAULTS["thresholds"])
    if config:
        cfg.update({k: v for k, v in config.items() if k != "thresholds"})
        th.update(config.get("thresholds") or {})

    def loop():
        n = 0
        while True:
            try:
                s = _sample()
                _append(TELEMETRY(), s)
                _check_thresholds(s, th)
                n += 1
                if n % 200 == 0:
                    _trim_ring()
            except Exception as e:
                print(f"[ops] recorder: {e!r}", flush=True)
            time.sleep(cfg["interval_sec"])

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    print(f"[ops] recorder started: every {cfg['interval_sec']} s, "
          f"thresholds {th}", flush=True)
    return t


# ── query side (what the agent's tools call) ─────────────────────────────────

def history(hours=24):
    rows = _read_jsonl(TELEMETRY(), since_ts=time.time() - hours * 3600)
    if not rows:
        return {"samples": 0, "note": "no telemetry recorded yet for this window"}
    out = {"samples": len(rows), "window_hours": hours,
           "from": time.strftime("%Y-%m-%d %H:%M", time.localtime(rows[0]["ts"])),
           "to": time.strftime("%Y-%m-%d %H:%M", time.localtime(rows[-1]["ts"]))}
    for key in METRIC_KEYS:
        vals = [(r["ts"], r[key]) for r in rows if isinstance(r.get(key), (int, float))]
        if not vals:
            continue
        nums = [v for _, v in vals]
        peak_ts, peak = max(vals, key=lambda x: x[1])
        out[key] = {"min": round(min(nums), 1), "avg": round(sum(nums) / len(nums), 1),
                      "max": round(peak, 1), "latest": round(nums[-1], 1),
                      "peak_at": time.strftime("%Y-%m-%d %H:%M", time.localtime(peak_ts))}
    return out


def recent_alerts(hours=24):
    rows = _read_jsonl(ALERTS(), since_ts=time.time() - hours * 3600)
    for r in rows:
        r["at"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(r.pop("ts")))
    return {"window_hours": hours, "alerts": rows,
            "active_now": sorted(_active_alerts),
            "note": "" if rows else "no threshold breaches in this window"}


def record_action(tool, detail, result):
    _append(ACTIONS(), {"ts": time.time(), "tool": tool,
                      "detail": str(detail)[:200], "result": str(result)[:200]})


def action_history(limit=20):
    rows = _read_jsonl(ACTIONS(), limit=limit)
    for r in rows:
        r["at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.pop("ts")))
    return {"actions": rows,
            "note": "" if rows else "no device-changing actions recorded yet"}


def anomalies(hours=24, baseline_hours=168):
    """Plain z-score of the recent window against the longer baseline --
    honest statistics, not a model."""
    now = time.time()
    base = _read_jsonl(TELEMETRY(), since_ts=now - baseline_hours * 3600)
    recent = [r for r in base if r["ts"] >= now - hours * 3600]
    if len(base) < 50 or len(recent) < 5:
        return {"note": f"not enough history yet ({len(base)} baseline samples; "
                        "anomaly checks need a few hours of recording)"}
    found = []
    for key in ("temp_c", "cpu_pct", "gpu_pct", "mem_used_mb", "swap_used_mb"):
        b = [r[key] for r in base if isinstance(r.get(key), (int, float))]
        rvals = [r[key] for r in recent if isinstance(r.get(key), (int, float))]
        if len(b) < 50 or not rvals:
            continue
        mean = sum(b) / len(b)
        var = sum((x - mean) ** 2 for x in b) / len(b)
        sd = var ** 0.5 or 1.0
        rmean = sum(rvals) / len(rvals)
        z = (rmean - mean) / sd
        if abs(z) >= 2.0:
            found.append({"metric": key, "recent_avg": round(rmean, 1),
                          "baseline_avg": round(mean, 1), "z_score": round(z, 1),
                          "direction": "above" if z > 0 else "below"})
    return {"window_hours": hours, "anomalies": found,
            "note": "" if found else "all metrics within 2 standard deviations "
                                     "of the baseline"}


# ── operator notes: semantic when the embed server is up, substring otherwise ─

NOTES_COLLECTION = "operator_notes"


def _embed_base():
    return os.environ.get(
        "WEDA_EMBED_BASE", "http://127.0.0.1:%s" % os.environ.get("WEDA_EMBED_PORT", "9001"))


def remember(note, author=""):
    entry = {"ts": time.time(), "note": note, "author": author}
    _append(NOTES(), entry)
    try:
        import rag_demo
        col = rag_demo.collection(NOTES_COLLECTION)
        vec = rag_demo.embed([note], _embed_base())
        col.upsert(ids=[f"note::{entry['ts']}"], embeddings=vec, documents=[note],
                   metadatas=[{"ts": entry["ts"], "author": author}])
        indexed = True
    except BaseException:
        indexed = False
    return {"result": "note saved",
            "search": "semantic" if indexed else
                      "text-only (embedding server not running -- notes still "
                      "recorded and searchable by keyword)"}


def recall(query, limit=5):
    try:
        import rag_demo
        col = rag_demo.collection(NOTES_COLLECTION)
        stored = col.count()
        if stored > 0:
            vec = rag_demo.embed([query], _embed_base())
            got = col.query(query_embeddings=vec, n_results=min(limit, stored),
                            include=["documents", "metadatas"])
            hits = [{"note": d,
                     "at": time.strftime("%Y-%m-%d %H:%M", time.localtime(m.get("ts", 0)))}
                    for d, m in zip(got["documents"][0], got["metadatas"][0])]
            if hits:
                return {"matches": hits, "search": "semantic"}
    except BaseException:
        pass
    rows = _read_jsonl(NOTES())
    words = [w for w in query.lower().split() if len(w) > 2]
    hits = [r for r in rows if any(w in r["note"].lower() for w in words)] or rows
    return {"matches": [{"note": r["note"],
                         "at": time.strftime("%Y-%m-%d %H:%M", time.localtime(r["ts"]))}
                        for r in hits[-limit:]],
            "search": "keyword (embedding server not running)"}


# ── own-stack log reading (host log dir mounted ro; scoped by cgroup id) ─────

HOST_LOG_DIR = "/host/docker-logs"
_ERROR_RE = re.compile(r"error|fatal|fail|traceback|killed|exception|oom", re.IGNORECASE)
_WARN_RE = re.compile(r"\bwarn", re.IGNORECASE)


def _stack_container_ids():
    ids = []
    try:
        import glob
        for d in (glob.glob("/sys/fs/cgroup/system.slice/docker-*.scope") +
                  glob.glob("/sys/fs/cgroup/docker/*")):
            cid = d.rsplit("docker-", 1)[-1].split(".scope")[0].rsplit("/", 1)[-1]
            try:
                pids = open(os.path.join(d, "cgroup.procs")).read().split()
            except OSError:
                continue
            for pid in pids[:8]:
                try:
                    cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="replace")
                except OSError:
                    continue
                if any(s in cmd for s in ("open_webui", "open-webui", "llama-server",
                                          "app_gateway", "/wise-edge/app", "serve.py")):
                    ids.append(cid)
                    break
    except Exception:
        pass
    return ids


def recent_errors(lines_per_container=120):
    if not os.path.isdir(HOST_LOG_DIR):
        return {"note": "log directory not mounted -- add the read-only "
                        "/var/lib/docker/containers mount and recreate"}
    out = []
    for cid in _stack_container_ids():
        log = os.path.join(HOST_LOG_DIR, cid, f"{cid}-json.log")
        try:
            with open(log, encoding="utf-8") as f:
                tail = f.readlines()[-lines_per_container:]
        except OSError:
            continue
        for line in tail:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            text = rec.get("log", "").strip()
            if not text:
                continue
            if _ERROR_RE.search(text):
                level = "error"
            elif _WARN_RE.search(text):
                level = "warning"
            else:
                continue
            out.append({"container": cid[:12], "at": rec.get("time", "")[:19],
                        "level": level, "line": text[:200]})
    out = out[-40:]
    errors = sum(1 for x in out if x["level"] == "error")
    warnings = len(out) - errors
    return {"errors": errors, "warnings": warnings, "log_lines": out,
            "note": (f"{errors} error line(s), {warnings} warning line(s). "
                     "Warnings are usually harmless; report each line at its "
                     "stated level, do not call a warning an error.")
                    if out else "no error or warning lines in the stack's recent logs"}


# ── deterministic report ─────────────────────────────────────────────────────

def write_report(hours=24):
    _ensure_dirs()
    h = history(hours)
    al = recent_alerts(hours)
    ac = action_history(50)
    an = anomalies(hours)
    day = time.strftime("%Y-%m-%d_%H%M")
    lines = [f"# Device health report — {time.strftime('%Y-%m-%d %H:%M')}",
             f"\nWindow: last {hours} h · samples: {h.get('samples', 0)}\n",
             "## Telemetry\n",
             "| Metric | min | avg | max | latest | peak at |", "|---|---|---|---|---|---|"]
    for k in METRIC_KEYS:
        v = h.get(k)
        if isinstance(v, dict):
            lines.append(f"| {k} | {v['min']} | {v['avg']} | {v['max']} | "
                         f"{v['latest']} | {v['peak_at']} |")
    lines += ["\n## Alerts\n"]
    if al["alerts"]:
        lines += ["| At | Kind | State | Detail |", "|---|---|---|---|"]
        lines += [f"| {a['at']} | {a['kind']} | {a['state']} | {a.get('message', '')} |"
                  for a in al["alerts"]]
    else:
        lines.append("No threshold breaches in this window.")
    lines += ["\n## Actions taken through the assistant\n"]
    if ac["actions"]:
        lines += ["| At | Tool | Detail | Result |", "|---|---|---|---|"]
        lines += [f"| {a['at']} | {a['tool']} | {a['detail']} | {a['result']} |"
                  for a in ac["actions"]]
    else:
        lines.append("No device-changing actions recorded.")
    if an.get("anomalies"):
        lines += ["\n## Anomalies (z >= 2 vs 7-day baseline)\n",
                  "| Metric | recent avg | baseline avg | z |", "|---|---|---|---|"]
        lines += [f"| {a['metric']} | {a['recent_avg']} | {a['baseline_avg']} | {a['z_score']} |"
                  for a in an["anomalies"]]
    path = os.path.join(REPORTS(), f"report-{day}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return {"result": "report written", "path": path,
            "summary": {"samples": h.get("samples", 0),
                        "alerts": len(al["alerts"]),
                        "actions": len(ac["actions"]),
                        "anomalies": len(an.get("anomalies", []))}}
