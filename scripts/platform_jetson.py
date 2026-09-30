#!/usr/bin/env python3
# ==============================================================================
# WEDA LLM Suite — Jetson platform layer
# ==============================================================================
# Every hardware-specific reading and action lives here, behind a small
# surface: gpu_percent / gpu_freq_mhz / power_modes / current_power_mode /
# set_power_mode. Porting the suite to different silicon means providing a
# platform_<name>.py with these five functions and importing it instead --
# agent_demo.py and the generic readings in device_tools.py never change.
import os
import subprocess

# Jetson names its GPU devfreq node by codename per SoC generation (ga10b =
# Orin/Ampere, gv11b = Xavier/Volta, gp10b = TX2/Pascal, gb10b = Thor/
# Blackwell, or a plain ".gpu" suffix on newer L4T) -- scan by name instead
# of hardcoding one path, the same detection jtop itself uses.
_GPU_CODENAMES = ("ga10b", "gv11b", "gp10b", "gb10b", "gpu")

NVPMODEL_CONF = "/etc/nvpmodel.conf"


def _gpu_devfreq_node():
    base = "/sys/class/devfreq"
    try:
        nodes = os.listdir(base)
    except OSError:
        return None
    for node in nodes:
        if any(code in node for code in _GPU_CODENAMES):
            return f"{base}/{node}"
    return None


def gpu_percent():
    node = _gpu_devfreq_node()
    if not node:
        return None
    # R36 keeps `load` on the platform device, not in the devfreq node itself;
    # the devfreq node's `device` symlink points there, so try both.
    for path in (f"{node}/load", f"{node}/device/load"):
        try:
            raw = int(open(path).read().strip())
            # Jetson GPU load sysfs reports permille (0-1000), not percent
            return round(raw / 10.0, 1)
        except (OSError, ValueError):
            continue
    return None


def gpu_freq_mhz():
    node = _gpu_devfreq_node()
    if not node:
        return None
    try:
        return round(int(open(f"{node}/cur_freq").read().strip()) / 1_000_000, 0)
    except (OSError, ValueError):
        return None


def _nvpmodel_bin():
    """The bind-mounted path is absolute and not guaranteed to be on PATH
    inside this image; fall back to a plain PATH lookup for a JetPack install
    that ships it somewhere else."""
    import shutil
    if os.access("/usr/sbin/nvpmodel", os.X_OK):
        return "/usr/sbin/nvpmodel"
    return shutil.which("nvpmodel") or "nvpmodel"


def power_modes():
    """Jetson power modes are defined in /etc/nvpmodel.conf, not discoverable
    through nvpmodel itself -- parse the '< POWER_MODEL ID=.. NAME=.. >'
    header lines rather than the CPU/GPU clock body under each one."""
    modes = []
    try:
        with open(NVPMODEL_CONF) as f:
            for line in f:
                line = line.strip()
                if line.startswith("<") and "POWER_MODEL" in line:
                    fields = dict(p.split("=", 1) for p in line.strip("<> ").split()[1:]
                                  if "=" in p)
                    if "ID" in fields:
                        modes.append({"id": fields["ID"], "name": fields.get("NAME", "?")})
    except OSError:
        pass
    return modes


def current_power_mode():
    """(mode_id, raw_output) or (None, diagnostic)."""
    try:
        q = subprocess.run([_nvpmodel_bin(), "-q"], capture_output=True,
                           text=True, timeout=5)
    except FileNotFoundError:
        return None, "nvpmodel is not runnable inside this container"
    if q.returncode != 0:
        return None, (q.stderr or q.stdout).strip() or f"nvpmodel -q exited {q.returncode}"
    lines = q.stdout.strip().splitlines()
    return (lines[-1].strip() if lines else None), q.stdout.strip()


def set_power_mode(mode_id):
    """Switch mode and VERIFY by readback -- polls briefly because the status
    file can lag the mode write. Returns (ok, detail)."""
    import time
    try:
        r = subprocess.run([_nvpmodel_bin(), "-m", str(mode_id)],
                           capture_output=True, text=True, timeout=15)
    except FileNotFoundError:
        return False, "nvpmodel is not runnable inside this container -- " \
                      "the power mode must be changed on the host"
    if r.returncode != 0:
        return False, (r.stderr or r.stdout).strip() or f"nvpmodel -m {mode_id} failed"
    actual, raw = None, ""
    for _ in range(4):
        actual, raw = current_power_mode()
        if actual == str(mode_id):
            return True, raw
        time.sleep(0.5)
    return False, (f"nvpmodel accepted the command but the device reports mode "
                   f"'{actual or 'unknown'}', not '{mode_id}' -- the switch did "
                   f"NOT take effect ({raw})")
