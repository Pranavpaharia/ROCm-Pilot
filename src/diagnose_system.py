#!/usr/bin/env python3
"""
ROCm Pilot — Standalone System Diagnostics
============================================
This script is designed to be copied and run directly on a remote AMD GPU
machine.  It has **zero external dependencies** — only Python 3.6+ stdlib
modules are used.

Usage:
    python3 diagnose_system.py            # pretty-printed JSON to stdout
    python3 diagnose_system.py --compact   # single-line JSON (pipe-friendly)

The output is a single JSON object describing the machine's hardware,
OS, ROCm stack, and installed AI/ML Python packages.
"""

import json
import os
import platform
import subprocess
import sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd, timeout=15):
    """Run *cmd* (string) in the shell.  Return stdout or None on failure."""
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout,
        )
        return proc.stdout.strip() if proc.returncode == 0 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Detection routines — each returns a plain JSON-serialisable value
# ---------------------------------------------------------------------------

def _detect_os():
    """OS name and version from /etc/os-release, falling back to platform."""
    info = {
        'name': platform.system(),
        'version': platform.version(),
        'pretty_name': None,
        'id': None,
        'version_id': None,
    }
    try:
        if os.path.isfile('/etc/os-release'):
            with open('/etc/os-release') as fh:
                for line in fh:
                    line = line.strip()
                    if '=' not in line:
                        continue
                    key, _, val = line.partition('=')
                    val = val.strip('"')
                    if key == 'PRETTY_NAME':
                        info['pretty_name'] = val
                    elif key == 'ID':
                        info['id'] = val
                    elif key == 'VERSION_ID':
                        info['version_id'] = val
    except OSError:
        pass
    return info


def _detect_kernel():
    """Kernel version via uname -r."""
    return _run('uname -r') or platform.release()


def _detect_python():
    """Python interpreter version."""
    return {
        'version': platform.python_version(),
        'executable': sys.executable,
        'implementation': platform.python_implementation(),
    }


def _detect_rocm_version():
    """ROCm version from the canonical version file."""
    ver = _run('cat /opt/rocm/.info/version 2>/dev/null')
    if not ver:
        # Fallback: try apt metadata
        apt_out = _run('apt show rocm-core 2>/dev/null | grep -i "^Version:"')
        if apt_out:
            ver = apt_out.split(':', 1)[-1].strip()
    return ver


def _detect_gpus():
    """GPU model, VRAM, and temperature from rocm-smi."""
    gpus = []

    # Try JSON output first (ROCm ≥ 5.x)
    raw = _run('rocm-smi --showproductname --showmeminfo vram --showtemp --json 2>/dev/null')
    if raw:
        try:
            data = json.loads(raw)
            for card_key, card_data in data.items():
                if not isinstance(card_data, dict):
                    continue
                gpu = {'card': card_key}

                # Model name
                for k in ('Card Series', 'Card series', 'card_series'):
                    if k in card_data:
                        gpu['model'] = card_data[k]
                        break

                # VRAM total (bytes → GB)
                for k in ('VRAM Total Memory (B)', 'vram_total'):
                    if k in card_data:
                        try:
                            gpu['vram_gb'] = round(float(card_data[k]) / (1024 ** 3), 1)
                        except (TypeError, ValueError):
                            pass
                        break

                # Temperature (edge)
                for k in ('Temperature (Sensor edge) (C)',
                          'Temperature (edge) (C)', 'temperature_edge'):
                    if k in card_data:
                        try:
                            gpu['temperature_c'] = float(card_data[k])
                        except (TypeError, ValueError):
                            pass
                        break

                gpus.append(gpu)
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: CSV product name
    if not gpus:
        csv_out = _run('rocm-smi --showproductname --csv 2>/dev/null')
        if csv_out:
            for line in csv_out.strip().split('\n')[1:]:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2:
                    gpus.append({'card': parts[0], 'model': parts[1]})

    return gpus


def _detect_gpu_arch():
    """GPU architecture (gfx ID) from rocminfo."""
    archs = []
    raw = _run('rocminfo 2>/dev/null')
    if not raw:
        return archs

    current_name = None
    for line in raw.split('\n'):
        stripped = line.strip()
        if stripped.startswith('Name:') and 'gfx' in stripped:
            arch_id = stripped.split('Name:')[1].strip()
            archs.append({'gfx_id': arch_id, 'marketing_name': current_name})
            current_name = None
        elif stripped.startswith('Marketing Name:'):
            current_name = stripped.split('Marketing Name:')[1].strip()
            if current_name in ('', 'N/A', 'AMD unknown'):
                current_name = None

    return archs


def _detect_pip_packages():
    """Check for common AI/ML pip packages and their versions."""
    packages = [
        'torch', 'tensorflow', 'jax', 'vllm', 'transformers',
    ]
    found = {}
    for pkg in packages:
        ver = _run(
            f'{sys.executable} -c "import {pkg}; print({pkg}.__version__)" 2>/dev/null'
        )
        if ver:
            found[pkg] = ver
    return found


def _detect_pytorch_rocm():
    """Check if PyTorch has ROCm / HIP support."""
    info = {'available': False}
    script = (
        'import torch; '
        'print(torch.cuda.is_available()); '
        'print(torch.version.hip or ""); '
        'print(torch.cuda.device_count()); '
        'print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")'
    )
    raw = _run(f'{sys.executable} -c "{script}" 2>/dev/null')
    if raw:
        parts = raw.split('\n')
        if len(parts) >= 1:
            info['available'] = parts[0].strip() == 'True'
        if len(parts) >= 2 and parts[1].strip():
            info['hip_version'] = parts[1].strip()
        if len(parts) >= 3:
            try:
                info['device_count'] = int(parts[2].strip())
            except ValueError:
                pass
        if len(parts) >= 4 and parts[3].strip():
            info['device_name'] = parts[3].strip()
    return info


def _detect_container():
    """Detect if running inside a Docker / Podman container."""
    # .dockerenv sentinel
    if os.path.exists('/.dockerenv'):
        return {'in_container': True, 'type': 'docker'}

    # cgroup-based detection
    try:
        if os.path.isfile('/proc/1/cgroup'):
            with open('/proc/1/cgroup') as fh:
                content = fh.read()
                for ctype in ('docker', 'podman', 'lxc'):
                    if ctype in content:
                        return {'in_container': True, 'type': ctype}
    except OSError:
        pass

    # Environment variable
    cvar = os.environ.get('container')
    if cvar:
        return {'in_container': True, 'type': cvar}

    return {'in_container': False, 'type': None}


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def diagnose():
    """Run all diagnostics and return a single JSON-serialisable dict."""
    return {
        'os': _detect_os(),
        'kernel': _detect_kernel(),
        'python': _detect_python(),
        'rocm_version': _detect_rocm_version(),
        'gpus': _detect_gpus(),
        'gpu_arch': _detect_gpu_arch(),
        'pip_packages': _detect_pip_packages(),
        'pytorch_rocm': _detect_pytorch_rocm(),
        'container': _detect_container(),
    }


if __name__ == '__main__':
    compact = '--compact' in sys.argv
    result = diagnose()
    if compact:
        print(json.dumps(result))
    else:
        print(json.dumps(result, indent=2))
