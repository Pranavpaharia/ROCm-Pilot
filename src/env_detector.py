"""
AMD Hardware & Software Environment Detector.
Auto-detects GPU model, ROCm version, installed frameworks, and OS details.
"""

import subprocess
import sys
from typing import Dict, Optional


def _run_cmd(cmd: str, timeout: int = 10) -> Optional[str]:
    """Run a shell command and return its stdout, or None on failure."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def detect_gpu() -> Dict:
    """Detect AMD GPU hardware information."""
    gpu_info: Dict = {
        'detected': False,
        'model': 'Unknown',
        'vram': 'Unknown',
        'temperature': 'Unknown',
        'rocm_version': 'Unknown',
    }

    # ----- rocm-smi -----
    rocm_output = _run_cmd('rocm-smi')
    if rocm_output:
        gpu_info['detected'] = True
        gpu_info['raw_rocm_smi'] = rocm_output

    # ----- rocminfo for marketing name & arch -----
    rocminfo_output = _run_cmd(
        'rocminfo 2>/dev/null | grep -E "Name:|Marketing Name:"'
    )
    if rocminfo_output:
        gpu_info['detected'] = True
        for line in rocminfo_output.split('\n'):
            line = line.strip()
            if 'Marketing Name:' in line:
                name = line.split('Marketing Name:')[1].strip()
                if name and name not in ('N/A', ''):
                    gpu_info['model'] = name
            elif 'Name:' in line and 'gfx' in line:
                gpu_info['arch'] = line.split('Name:')[1].strip()

    # ----- ROCm version -----
    rocm_ver = _run_cmd('cat /opt/rocm/.info/version 2>/dev/null')
    if rocm_ver:
        gpu_info['rocm_version'] = rocm_ver
    else:
        rocm_ver = _run_cmd('apt show rocm-core 2>/dev/null | grep Version')
        if rocm_ver:
            gpu_info['rocm_version'] = rocm_ver.split(':')[-1].strip()

    # ----- PyTorch GPU introspection -----
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info['detected'] = True
            gpu_info['model'] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_info['vram'] = f"{props.total_mem / (1024 ** 3):.1f} GB"
            gpu_info['pytorch_version'] = torch.__version__
            gpu_info['hip_version'] = getattr(torch.version, 'hip', 'N/A')
    except ImportError:
        pass

    return gpu_info


def detect_software() -> Dict:
    """Detect installed AI/ML software and OS details."""
    software: Dict = {
        'python_version': sys.version.split()[0],
        'os': 'Unknown',
        'frameworks': {},
    }

    # OS info
    os_info = _run_cmd('cat /etc/os-release 2>/dev/null | grep PRETTY_NAME')
    if os_info and '=' in os_info:
        software['os'] = os_info.split('=', 1)[1].strip('"')

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
    """Run full environment detection (GPU + software)."""
    return {
        'gpu': detect_gpu(),
        'software': detect_software(),
    }


def format_env_context(env: Dict) -> str:
    """
    Format the detected environment as a text block for injection
    into the LLM system prompt.
    """
    gpu = env['gpu']
    sw = env['software']

    lines = [
        "=== DETECTED ENVIRONMENT ===",
        f"GPU Detected: {'Yes' if gpu['detected'] else 'No'}",
    ]

    if gpu['detected']:
        lines.append(f"GPU Model: {gpu['model']}")
        lines.append(f"VRAM: {gpu['vram']}")
        lines.append(f"ROCm Version: {gpu['rocm_version']}")
        if 'arch' in gpu:
            lines.append(f"GPU Architecture: {gpu['arch']}")
        if 'pytorch_version' in gpu:
            lines.append(f"PyTorch Version: {gpu['pytorch_version']}")
        if 'hip_version' in gpu:
            lines.append(f"HIP Version: {gpu['hip_version']}")

    lines.append(f"Python Version: {sw['python_version']}")
    lines.append(f"OS: {sw['os']}")

    if sw['frameworks']:
        lines.append("Installed Frameworks:")
        for name, ver in sorted(sw['frameworks'].items()):
            lines.append(f"  - {name}: {ver}")
    else:
        lines.append("Installed AI/ML Frameworks: None detected")

    lines.append("=== END ENVIRONMENT ===")

    return '\n'.join(lines)


if __name__ == '__main__':
    env = detect_environment()
    print(format_env_context(env))
