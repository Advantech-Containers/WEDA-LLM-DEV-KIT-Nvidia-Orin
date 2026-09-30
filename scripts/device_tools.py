#!/usr/bin/env python3
# ==============================================================================
# WEDA LLM Suite — generic Linux device readings
# ==============================================================================
# Everything here works on any Linux host: /proc, /sys/class/thermal, ps and
# standard socket interfaces. Hardware-specific readings (GPU, power modes)
# live in platform_jetson.py -- porting the suite to different silicon means
# replacing that file, never this one.
import os
import subprocess
import time


def meminfo():
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                out[k] = int(v.split()[0])
    return out


def mem_summary_mb():
    m = meminfo()
    total = m.get("MemTotal", 0) // 1024
    avail = m.get("MemAvailable", 0) // 1024
    swap_total = m.get("SwapTotal", 0) // 1024
    swap_used = swap_total - m.get("SwapFree", 0) // 1024
    return {"mem_total_mb": total, "mem_available_mb": avail,
            "mem_used_mb": total - avail,
            "swap_total_mb": swap_total, "swap_used_mb": swap_used}


def loadavg():
    with open("/proc/loadavg") as f:
        return f.read().split()[:3]


def cpu_percent(sample_sec=0.15):
    """Per-core + overall CPU percent from two /proc/stat samples -- the same
    source top/htop read, not loadavg (which is a decayed run-queue average,
    not a point-in-time percent)."""
    def read():
        with open("/proc/stat") as f:
            return {p[0]: [int(x) for x in p[1:]]
                    for p in (l.split() for l in f) if p[0].startswith("cpu")}
    t0 = read()
    time.sleep(sample_sec)
    t1 = read()
    pct = {}
    for k in t0:
        prev, cur = t0[k], t1[k]
        prev_idle, cur_idle = prev[3] + prev[4], cur[3] + cur[4]
        dt = sum(cur) - sum(prev)
        di = cur_idle - prev_idle
        pct[k] = round(100.0 * (dt - di) / dt, 1) if dt > 0 else 0.0
    return pct


def thermal_zones(limit=6):
    zones = []
    base = "/sys/class/thermal"
    for z in sorted(os.listdir(base)):
        if z.startswith("thermal_zone"):
            try:
                with open(f"{base}/{z}/temp") as f:
                    zones.append(round(int(f.read().strip()) / 1000.0, 1))
            except (OSError, ValueError):
                continue
    return zones[:limit]


def disk_free_gb(path="/"):
    import shutil
    return round(shutil.disk_usage(path).free / (1024 ** 3), 1)


def uptime_hours():
    return round(float(open("/proc/uptime").read().split()[0]) / 3600.0, 1)


def top_processes(n=5):
    out = subprocess.run(["ps", "-eo", "pid,pcpu,pmem,comm", "--sort=-pcpu"],
                         capture_output=True, text=True, timeout=5)
    lines = out.stdout.strip().splitlines()[1:n + 1]
    return [dict(zip(("pid", "cpu_pct", "mem_pct", "name"), l.split(None, 3)))
            for l in lines]


def drop_caches():
    """Flush reclaimable page cache. Returns (ok, message)."""
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3")
        return True, "page caches flushed"
    except (PermissionError, OSError) as e:
        return False, str(e)


def iface_ips():
    import fcntl
    import socket
    import struct
    ips = {}
    try:
        names = sorted(os.listdir("/sys/class/net"))
    except OSError:
        return ips
    for name in names:
        sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            packed = fcntl.ioctl(sk.fileno(), 0x8915,  # SIOCGIFADDR
                                 struct.pack("256s", name[:15].encode()))
            ips[name] = socket.inet_ntoa(packed[20:24])
        except OSError:
            continue
        finally:
            sk.close()
    return ips


def primary_ip(ips=None):
    """The address a client should use: a physical NIC, never a bridge --
    docker0 comes first alphabetically and is exactly the wrong answer."""
    ips = ips if ips is not None else iface_ips()

    def rank(n):
        if n.startswith(("eth", "en", "wlan", "wl")):
            return 0
        if n == "lo" or n.startswith(("docker", "veth", "br-", "l4tbr", "virbr")):
            return 2
        return 1
    return next((ips[n] for n in sorted(ips, key=rank) if rank(n) < 2), None)


def listening_ports():
    """LISTEN-state TCP ports from /proc/net/tcp{,6} -- host-wide when the
    process shares the host network namespace."""
    ports = set()
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(path) as f:
                next(f)
                for line in f:
                    fields = line.split()
                    if len(fields) > 3 and fields[3] == "0A":
                        ports.add(int(fields[1].rsplit(":", 1)[1], 16))
        except (OSError, ValueError):
            continue
    return sorted(ports)
