"""
AMD Hardware & Software Environment Detector.
Auto-detects GPU model(s), ROCm version, installed frameworks, and OS details.
Supports multi-GPU systems and containerized environments.
"""

import json
import logging
import os
import subprocess
import sys
from typing import Dict, List, Optional

logger = logging.getLogger('rocm_pilot.env')


def _run_cmd(cmd: str, timeout: int = 10) -> Optional[str]:
    """Run a shell command and return its stdout, or None on failure."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


def _detect_container() -> Dict:
    """
    Detect if running inside a Docker/Podman container.
    Returns dict with 'in_container' bool and 'container_type' string.
    """
    info: Dict = {
        'in_container': False,
        'container_type': None,
        'warnings': [],
    }

    try:
        # Check for Docker
        if os.path.exists('/.dockerenv'):
            info['in_container'] = True
            info['container_type'] = 'docker'
            return info

        # Check cgroup for docker/lxc/podman
        if os.path.exists('/proc/1/cgroup'):
            with open('/proc/1/cgroup', 'r') as f:
                cgroup_content = f.read()
                if 'docker' in cgroup_content:
                    info['in_container'] = True
                    info['container_type'] = 'docker'
                    return info
                elif 'podman' in cgroup_content:
                    info['in_container'] = True
                    info['container_type'] = 'podman'
                    return info
                elif 'lxc' in cgroup_content:
                    info['in_container'] = True
                    info['container_type'] = 'lxc'
                    return info

        # Check for container environment variable
        if os.environ.get('container'):
            info['in_container'] = True
            info['container_type'] = os.environ.get('container', 'unknown')
            return info

    except Exception:
        pass

    return info


def detect_gpus() -> List[Dict]:
    """
    Detect all AMD GPU hardware information.
    Returns a list of GPU dicts (one per detected GPU).
    """
    gpus: List[Dict] = []
    detection_errors: List[str] = []

    # ----- Method 1: rocm-smi for multi-GPU -----
    rocm_smi_csv = _run_cmd('rocm-smi --showproductname --csv 2>/dev/null')
    if rocm_smi_csv:
        try:
            lines = rocm_smi_csv.strip().split('\n')
            # Skip header line
            for line in lines[1:]:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2 and parts[0].lower() != 'card':
                    gpu: Dict = {
                        'detected': True,
                        'card_id': parts[0],
                        'model': parts[1] if len(parts) > 1 else 'Unknown',
                        'vram': 'Unknown',
                        'temperature': 'Unknown',
                        'rocm_version': 'Unknown',
                    }
                    # Try to get more info for this specific card
                    card_id = parts[0]
                    gpus.append(gpu)
        except Exception as e:
            detection_errors.append(f"rocm-smi CSV parse error: {e}")
    else:
        # Fallback: basic rocm-smi
        rocm_output = _run_cmd('rocm-smi 2>/dev/null')
        if rocm_output:
            gpu = {
                'detected': True,
                'model': 'Unknown',
                'vram': 'Unknown',
                'temperature': 'Unknown',
                'rocm_version': 'Unknown',
                'raw_rocm_smi': rocm_output,
            }
            gpus.append(gpu)
        else:
            detection_errors.append("rocm-smi not available")

    # ----- Method 2: rocminfo for architecture details -----
    rocminfo_output = _run_cmd(
        'rocminfo 2>/dev/null | grep -E "Name:|Marketing Name:"'
    )
    if rocminfo_output:
        gpu_idx = 0
        for line in rocminfo_output.split('\n'):
            line = line.strip()
            if 'Marketing Name:' in line:
                name = line.split('Marketing Name:')[1].strip()
                if name and name not in ('N/A', '', 'AMD unknown'):
                    if gpu_idx < len(gpus):
                        gpus[gpu_idx]['model'] = name
                    else:
                        gpus.append({
                            'detected': True,
                            'model': name,
                            'vram': 'Unknown',
                            'temperature': 'Unknown',
                            'rocm_version': 'Unknown',
                        })
            elif 'Name:' in line and 'gfx' in line:
                arch = line.split('Name:')[1].strip()
                if gpu_idx < len(gpus):
                    gpus[gpu_idx]['arch'] = arch
                gpu_idx += 1
    else:
        detection_errors.append("rocminfo not available")

    # ----- Method 3: ROCm version -----
    rocm_ver = _run_cmd('cat /opt/rocm/.info/version 2>/dev/null')
    if not rocm_ver:
        rocm_ver = _run_cmd('apt show rocm-core 2>/dev/null | grep Version')
        if rocm_ver:
            rocm_ver = rocm_ver.split(':')[-1].strip()
    if rocm_ver:
        for gpu in gpus:
            gpu['rocm_version'] = rocm_ver
    else:
        detection_errors.append("Could not detect ROCm version")

    # ----- Method 4: PyTorch GPU introspection (multi-GPU) -----
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                # Match PyTorch devices to existing GPU entries or create new ones
                matched = False
                for gpu in gpus:
                    if gpu.get('model') == 'Unknown' or gpu.get('model') == torch.cuda.get_device_name(i):
                        gpu['model'] = torch.cuda.get_device_name(i)
                        props = torch.cuda.get_device_properties(i)
                        gpu['vram'] = f"{props.total_mem / (1024 ** 3):.1f} GB"
                        gpu['pytorch_version'] = torch.__version__
                        gpu['hip_version'] = getattr(torch.version, 'hip', 'N/A')
                        gpu['device_index'] = i
                        matched = True
                        break
                if not matched:
                    props = torch.cuda.get_device_properties(i)
                    gpus.append({
                        'detected': True,
                        'model': torch.cuda.get_device_name(i),
                        'vram': f"{props.total_mem / (1024 ** 3):.1f} GB",
                        'temperature': 'Unknown',
                        'rocm_version': rocm_ver or 'Unknown',
                        'pytorch_version': torch.__version__,
                        'hip_version': getattr(torch.version, 'hip', 'N/A'),
                        'device_index': i,
                    })
    except ImportError:
        detection_errors.append("PyTorch not installed")
    except Exception as e:
        detection_errors.append(f"PyTorch GPU detection error: {e}")

    # If no GPUs found at all, add a placeholder
    if not gpus:
        gpus.append({
            'detected': False,
            'model': 'None detected',
            'vram': 'N/A',
            'temperature': 'N/A',
            'rocm_version': 'Unknown',
        })

    # Attach detection errors to each GPU for visibility
    for gpu in gpus:
        gpu['detection_errors'] = detection_errors.copy()

    return gpus


def detect_software() -> Dict:
    """Detect installed AI/ML software and OS details."""
    software: Dict = {
        'python_version': sys.version.split()[0],
        'os': 'Unknown',
        'frameworks': {},
        'detection_errors': [],
    }

    # OS info
    os_info = _run_cmd('cat /etc/os-release 2>/dev/null | grep PRETTY_NAME')
    if os_info and '=' in os_info:
        software['os'] = os_info.split('=', 1)[1].strip('"')
    else:
        software['detection_errors'].append("Could not detect OS")

    # Probe common AI/ML packages
    probe_packages = [
        'torch', 'tensorflow', 'jax', 'vllm',
        'transformers', 'sentence_transformers',
        'onnxruntime', 'chromadb', 'langchain',
    ]
    for pkg in probe_packages:
        try:
            mod = __import__(pkg)
            software['frameworks'][pkg] = getattr(mod, '__version__', 'installed')
        except ImportError:
            pass

    return software


def detect_gpu_utilization() -> Dict:
    """
    Query real-time GPU utilization, memory usage, and temperature via rocm-smi.

    Returns a dict keyed by GPU ID (e.g. "card0") with sub-keys:
        gpu_id, gpu_use_percent, mem_use_percent, mem_used_mb,
        mem_total_mb, temperature_c.
    Returns an empty dict on failure.
    """
    result: Dict = {}
    raw = _run_cmd('rocm-smi --showuse --showmemuse --showtemp --showmeminfo vram --json 2>/dev/null')
    if not raw:
        logger.debug("rocm-smi utilization query returned no output")
        return result

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse rocm-smi JSON output: %s", exc)
        return result

    for card_key, card_data in data.items():
        if not isinstance(card_data, dict):
            continue
        gpu_id = card_key  # e.g. "card0"

        # Extract GPU usage percentage
        gpu_use = card_data.get('GPU use (%)', card_data.get('GPU Usage (%)', None))
        # Extract memory usage percentage (with VRAM% fallback for VF)
        mem_use = (
            card_data.get('GPU memory use (%)') or
            card_data.get('GPU Memory Usage (%)') or
            card_data.get('GPU Memory Allocated (VRAM%)') or
            None
        )
        # Extract temperature (with junction/memory fallback for VF)
        temperature = (
            card_data.get('Temperature (Sensor edge) (C)') or
            card_data.get('Temperature (edge) (C)') or
            card_data.get('Temperature (Sensor junction) (C)') or
            card_data.get('Temperature (Sensor memory) (C)') or
            None
        )

        # VRAM totals — try common key patterns
        vram_total = card_data.get('VRAM Total Memory (B)',
                                   card_data.get('vram_total', None))
        vram_used = card_data.get('VRAM Total Used Memory (B)',
                                  card_data.get('vram_used', None))

        entry: Dict = {'gpu_id': gpu_id}

        # Safe numeric conversion helper
        def _to_float(val: object) -> Optional[float]:
            if val is None:
                return None
            try:
                return float(val)
            except (TypeError, ValueError):
                return None

        entry['gpu_use_percent'] = _to_float(gpu_use)
        entry['mem_use_percent'] = _to_float(mem_use)
        entry['temperature_c'] = _to_float(temperature)

        # Convert bytes → MB for VRAM
        vram_total_f = _to_float(vram_total)
        vram_used_f = _to_float(vram_used)
        entry['mem_total_mb'] = round(vram_total_f / (1024 * 1024), 1) if vram_total_f is not None else None
        entry['mem_used_mb'] = round(vram_used_f / (1024 * 1024), 1) if vram_used_f is not None else None

        result[gpu_id] = entry

    logger.debug("GPU utilization detected for %d card(s)", len(result))
    return result


def detect_gpu_processes() -> List[Dict]:
    """
    Detect processes currently using AMD GPUs via rocm-smi.

    Returns a list of dicts with keys:
        pid, gpu_id, vram_usage, process_name.
    Returns an empty list on failure.
    """
    processes: List[Dict] = []

    # Use KFD processes information from --showpids (clean tabular format)
    raw = _run_cmd('rocm-smi --showpids 2>/dev/null')
    if not raw:
        # Fall back to --showpidgpus if --showpids is not available
        raw = _run_cmd('rocm-smi --showpidgpus 2>/dev/null')
    if not raw:
        logger.debug("rocm-smi process query returned no output")
        return processes

    # Parse tabular output
    for line in raw.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('=') or line.startswith('-') or line.startswith('KFD'):
            continue
        lower = line.lower()
        if 'pid' in lower and ('process' in lower or 'gpu' in lower):
            continue

        parts = line.split()
        if len(parts) < 1:
            continue

        try:
            pid = int(parts[0])
        except ValueError:
            continue

        # In --showpids, the structure is:
        # PID   PROCESS NAME   GPU(s)   VRAM USED   SDMA USED   CU OCCUPANCY
        # e.g.: 8300   python3   1   17925210112   0   0
        process_name = parts[1] if len(parts) > 1 else 'unknown'
        gpu_id = parts[2] if len(parts) > 2 else 'N/A'
        vram_bytes = parts[3] if len(parts) > 3 else '0'

        # Format VRAM usage to MB for readability
        try:
            vram_mb = round(float(vram_bytes) / (1024 * 1024), 1)
            vram_usage = f"{vram_mb} MB"
        except (ValueError, TypeError):
            vram_usage = vram_bytes

        # Format GPU ID to cardN
        gpu_str = gpu_id
        if gpu_id != 'N/A' and not gpu_id.startswith('card'):
            gpu_str = f"card{gpu_id}"

        processes.append({
            'pid': pid,
            'gpu_id': gpu_str,
            'vram_usage': vram_usage,
            'process_name': process_name,
        })

    logger.debug("Detected %d GPU process(es)", len(processes))
    return processes


def _resolve_process_name(pid: int) -> str:
    """
    Resolve a human-readable process name for a PID.
    Tries /proc/{pid}/comm first, then falls back to ps.
    """
    # Method 1: /proc filesystem (Linux)
    comm_path = f'/proc/{pid}/comm'
    try:
        if os.path.isfile(comm_path):
            with open(comm_path, 'r') as fh:
                name = fh.read().strip()
                if name:
                    return name
    except (OSError, PermissionError):
        pass

    # Method 2: ps command (portable)
    name = _run_cmd(f'ps -p {pid} -o comm= 2>/dev/null')
    if name:
        return name

    return 'unknown'


def format_gpu_monitor(utilization: Dict, processes: List[Dict]) -> str:
    """
    Format GPU utilization and running processes as a readable
    markdown string suitable for display in the web UI.

    Args:
        utilization: dict returned by detect_gpu_utilization().
        processes:   list returned by detect_gpu_processes().

    Returns:
        A multi-line markdown string.
    """
    lines: List[str] = ['## 🖥️ GPU Monitor', '']

    # --- Utilization section ---
    if utilization:
        for card_key in sorted(utilization.keys()):
            info = utilization[card_key]
            gpu_id = info.get('gpu_id', card_key)
            gpu_pct = info.get('gpu_use_percent')
            mem_pct = info.get('mem_use_percent')
            mem_used = info.get('mem_used_mb')
            mem_total = info.get('mem_total_mb')
            temp = info.get('temperature_c')

            lines.append(f'### {gpu_id}')

            # GPU usage bar
            if gpu_pct is not None:
                bar = _progress_bar(gpu_pct)
                lines.append(f'**GPU Usage:** {bar} {gpu_pct:.0f}%')
            else:
                lines.append('**GPU Usage:** N/A')

            # VRAM bar
            if mem_used is not None and mem_total is not None and mem_total > 0:
                vram_pct = (mem_used / mem_total) * 100
                bar = _progress_bar(vram_pct)
                lines.append(
                    f'**VRAM:** {bar} {mem_used:.0f} MB / {mem_total:.0f} MB '
                    f'({vram_pct:.1f}%)'
                )
            elif mem_pct is not None:
                bar = _progress_bar(mem_pct)
                lines.append(f'**VRAM:** {bar} {mem_pct:.0f}%')
            else:
                lines.append('**VRAM:** N/A')

            # Temperature
            if temp is not None:
                temp_emoji = '🔥' if temp >= 80 else '🌡️'
                lines.append(f'**Temp:** {temp_emoji} {temp:.0f} °C')
            else:
                lines.append('**Temp:** N/A')

            lines.append('')
    else:
        lines.append('*No GPU utilization data available.*')
        lines.append('')

    # --- Process table ---
    lines.append('### Running GPU Processes')
    if processes:
        lines.append('')
        lines.append('| PID | GPU | VRAM | Process |')
        lines.append('|-----|-----|------|---------|')
        for proc in processes:
            lines.append(
                f"| {proc['pid']} | {proc['gpu_id']} | "
                f"{proc['vram_usage']} | {proc['process_name']} |"
            )
    else:
        lines.append('*No GPU processes detected.*')

    lines.append('')
    return '\n'.join(lines)


def _progress_bar(pct: float, width: int = 20) -> str:
    """Render a text progress bar with color coding based on usage."""
    pct = max(0.0, min(100.0, pct))
    filled = int(round(width * pct / 100))
    empty = width - filled
    
    if pct < 50:
        color = "#10b981"  # Emerald Green (low usage)
    elif pct < 85:
        color = "#f59e0b"  # Amber/Yellow (medium usage)
    else:
        color = "#ef4444"  # Red (high usage)
        
    filled_str = "█" * filled
    empty_str = "░" * empty
    
    return f'<span style="color: {color};">[{filled_str}{empty_str}]</span>'


def detect_environment() -> Dict:
    """
    Run full environment detection (GPUs + software + container + utilization + processes).
    Returns a dict with:
        - gpus: List[Dict] (one per detected GPU)
        - software: Dict
        - container: Dict
        - gpu_utilization: Dict
        - gpu_processes: List[Dict]
    """
    return {
        'gpus': detect_gpus(),
        'software': detect_software(),
        'container': _detect_container(),
        'gpu_utilization': detect_gpu_utilization(),
        'gpu_processes': detect_gpu_processes(),
    }


def format_env_context(env: Dict) -> str:
    """
    Format the detected environment as a text block for injection
    into the LLM system prompt.
    """
    gpus = env.get('gpus', [])
    sw = env.get('software', {})
    container = env.get('container', {})

    lines = [
        "=== DETECTED ENVIRONMENT ===",
    ]

    # Container info
    if container.get('in_container'):
        ctype = container.get('container_type', 'unknown')
        lines.append(f"⚠️  Running in {ctype} container")
        lines.append("   Note: GPU access may require --device=/dev/kfd --group-add video")
        lines.append("   rocm-smi may have limited functionality inside containers")

    # GPU info (multi-GPU)
    detected_gpus = [g for g in gpus if g.get('detected')]
    if detected_gpus:
        lines.append(f"GPUs Detected: {len(detected_gpus)}")
        for i, gpu in enumerate(detected_gpus):
            prefix = f"  GPU {i}" if len(detected_gpus) > 1 else "  GPU"
            lines.append(f"{prefix} Model: {gpu.get('model', 'Unknown')}")
            if gpu.get('vram') and gpu['vram'] != 'Unknown':
                lines.append(f"{prefix} VRAM: {gpu['vram']}")
            if gpu.get('arch'):
                lines.append(f"{prefix} Architecture: {gpu['arch']}")
            if gpu.get('device_index') is not None:
                lines.append(f"{prefix} PyTorch Index: {gpu['device_index']}")
        # Show ROCm version (same for all GPUs)
        rocm_ver = detected_gpus[0].get('rocm_version', 'Unknown')
        lines.append(f"ROCm Version: {rocm_ver}")
        # Show PyTorch/HIP if available
        pytorch_ver = detected_gpus[0].get('pytorch_version')
        if pytorch_ver:
            lines.append(f"PyTorch Version: {pytorch_ver}")
        hip_ver = detected_gpus[0].get('hip_version')
        if hip_ver:
            lines.append(f"HIP Version: {hip_ver}")
    else:
        lines.append("GPU Detected: No")
        lines.append("  AMD GPU not detected. Check ROCm installation and hardware.")

    # Detection errors
    all_errors = set()
    for gpu in gpus:
        all_errors.update(gpu.get('detection_errors', []))
    all_errors.update(sw.get('detection_errors', []))
    if all_errors:
        lines.append("Detection Notes:")
        for err in sorted(all_errors):
            lines.append(f"  - {err}")

    lines.append(f"Python Version: {sw.get('python_version', 'Unknown')}")
    lines.append(f"OS: {sw.get('os', 'Unknown')}")

    frameworks = sw.get('frameworks', {})
    if frameworks:
        lines.append("Installed Frameworks:")
        for name, ver in sorted(frameworks.items()):
            lines.append(f"  - {name}: {ver}")
    else:
        lines.append("Installed AI/ML Frameworks: None detected")

    lines.append("=== END ENVIRONMENT ===")

    return '\n'.join(lines)


if __name__ == '__main__':
    env = detect_environment()
    print(format_env_context(env))