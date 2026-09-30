#!/usr/bin/env python3
# ==============================================================================
# WEDA LLM Suite — read-only Docker introspection (Engine API over the socket)
# ==============================================================================
# list / inspect / stats only -- no start, stop or any state change. Talks to
# the Engine API over /var/run/docker.sock with stdlib HTTP (no docker CLI in
# the image). The socket is read-write at the OS level, so this module never
# issues a mutating verb: containment is by construction here, not by daemon
# permission.
import http.client
import json
import os
import re
import socket as _socket
from urllib.parse import quote

DOCKER_SOCK = "/var/run/docker.sock"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _name_or_error(name):
    """Return (safe_name, None) or (None, error dict). Only a container name or
    id the Engine itself accepts may reach the request path."""
    if not name:
        return None, {"error": "name is required -- list_containers shows what exists"}
    if not NAME_RE.match(name):
        return None, {"error": f"'{name}' is not a container name -- "
                               "list_containers shows what exists"}
    return quote(name, safe=""), None


def _api(path, timeout=15):
    """GET the Engine API. Returns (status, parsed) or raises LookupError with
    a remedy when the socket is not mounted."""
    if not os.path.exists(DOCKER_SOCK):
        raise LookupError("docker socket not mounted in this container -- add "
                          "/var/run/docker.sock and recreate (./build.sh --stop "
                          "&& ./build.sh)")

    class _Conn(http.client.HTTPConnection):
        def connect(self):
            s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
            s.settimeout(timeout)
            s.connect(DOCKER_SOCK)
            self.sock = s

    conn = _Conn("localhost", timeout=timeout)
    try:
        conn.request("GET", path, headers={"Host": "docker"})
        r = conn.getresponse()
        body = r.read()
        ctype = r.headers.get("Content-Type", "")
        data = json.loads(body) if body and ctype.startswith("application/json") \
            else body.decode(errors="replace")
        return r.status, data
    finally:
        conn.close()


def list_containers():
    status, data = _api("/containers/json?all=1")
    if status != 200:
        return {"error": f"docker API returned {status}: {data}"}
    rows = [{"name": (c.get("Names") or ["?"])[0].lstrip("/"),
             "id": c.get("Id", "")[:12], "image": c.get("Image", ""),
             "state": c.get("State", ""), "status": c.get("Status", "")}
            for c in data]
    return {"running": sum(1 for r in rows if r["state"] == "running"),
            "total": len(rows), "containers": rows}


def inspect_container(name):
    safe, err = _name_or_error(name)
    if err:
        return err
    status, c = _api(f"/containers/{safe}/json")
    if status == 404:
        return {"error": f"no container named '{name}'"}
    if status != 200:
        return {"error": f"docker API returned {status}: {c}"}
    host = c.get("HostConfig") or {}
    state = c.get("State") or {}
    nets = {n: (v or {}).get("IPAddress", "") for n, v in
            ((c.get("NetworkSettings") or {}).get("Networks") or {}).items()}
    return {"name": (c.get("Name") or "").lstrip("/"), "id": c.get("Id", "")[:12],
            "image": (c.get("Config") or {}).get("Image", ""),
            "state": state.get("Status", ""), "started_at": state.get("StartedAt", ""),
            "restart_count": c.get("RestartCount", 0),
            "oom_killed": state.get("OOMKilled", False),
            "exit_code": state.get("ExitCode", 0),
            "network_mode": host.get("NetworkMode", ""),
            "networks": nets or "host network: shares the device's interfaces",
            "restart_policy": (host.get("RestartPolicy") or {}).get("Name", "")}


def container_stats(name):
    safe, err = _name_or_error(name)
    if err:
        return err
    status, s = _api(f"/containers/{safe}/stats?stream=false", timeout=30)
    if status == 404:
        return {"error": f"no container named '{name}'"}
    if status != 200:
        return {"error": f"docker API returned {status}: {s}"}
    cpu, pre = s.get("cpu_stats") or {}, s.get("precpu_stats") or {}
    cpu_d = (cpu.get("cpu_usage") or {}).get("total_usage", 0) - \
            (pre.get("cpu_usage") or {}).get("total_usage", 0)
    sys_d = cpu.get("system_cpu_usage", 0) - pre.get("system_cpu_usage", 0)
    ncpu = cpu.get("online_cpus") or \
        len((cpu.get("cpu_usage") or {}).get("percpu_usage") or []) or 1
    mem = s.get("memory_stats") or {}
    return {"name": name,
            "cpu_percent": round(100.0 * cpu_d / sys_d * ncpu, 1) if sys_d > 0 else 0.0,
            "mem_used_mb": round(mem.get("usage", 0) / (1024 ** 2), 1),
            "mem_limit_mb": round(mem.get("limit", 0) / (1024 ** 2), 1),
            "pids": (s.get("pids_stats") or {}).get("current", 0),
            "note": "on Jetson unified memory, GPU allocations do not appear in "
                    "container mem stats -- get_device_stats shows the real "
                    "device-wide picture"}
