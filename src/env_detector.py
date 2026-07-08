"""
AMD Hardware & Software Environment Detector.
Auto-detects GPU model(s), ROCm version, installed frameworks, and OS details.
Supports multi-GPU systems and containerized environments.
"""

import os
import subprocess
import sys
from typing import Dict, List, Optional


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


def detect_environment() -> Dict:
    """
    Run full environment detection (GPUs + software + container).
    Returns a dict with:
        - gpus: List[Dict] (one per detected GPU)
        - software: Dict
        - container: Dict
    """
    return {
        'gpus': detect_gpus(),
        'software': detect_software(),
        'container': _detect_container(),
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