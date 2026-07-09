"""
GPU Compatibility module for ROCm-Pilot.
Provides structured lookup functions for AMD GPU architectures,
ROCm versions, and PyTorch compatibility configurations.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("rocm_pilot.gpu_compat")

# Module level cache for database
_GPU_DB_CACHE: Optional[Dict] = None


def load_gpu_database(path: str = 'data/gpu_database.json') -> Dict:
    """
    Loads and validates the GPU compatibility database JSON file.
    Caches the result in a module-level variable.
    """
    global _GPU_DB_CACHE
    if _GPU_DB_CACHE is not None:
        return _GPU_DB_CACHE

    db_path = Path(path)
    if not db_path.exists():
        # Look relative to the workspace root if not found
        db_path = Path(__file__).parent.parent / path

    if not db_path.exists():
        raise FileNotFoundError(f"GPU database not found at {path}")

    with open(db_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Basic validation
    required = ["gpu_architectures", "rocm_versions", "pytorch_rocm_matrix"]
    for field in required:
        if field not in data:
            raise ValueError(f"Invalid database: missing required section '{field}'")

    _GPU_DB_CACHE = data
    logger.info("Loaded GPU compatibility database from %s", db_path)
    return _GPU_DB_CACHE


def lookup_gpu(db: Dict, gfx_id_or_name: str) -> Optional[Dict]:
    """
    Looks up a GPU record by exact gfx ID first, falling back to a case-insensitive
    fuzzy name match (substring) on the GPU model name or family name.
    """
    if not gfx_id_or_name:
        return None

    cleaned = gfx_id_or_name.strip().lower()

    # Exact gfx ID lookup
    gpu_archs = db.get("gpu_architectures", {})
    if cleaned in gpu_archs:
        return gpu_archs[cleaned]

    # Try matching without "gfx" prefix if it was supplied (or vice versa)
    if cleaned.startswith("gfx"):
        no_prefix = cleaned[3:]
        if no_prefix in gpu_archs:
            return gpu_archs[no_prefix]
    else:
        with_prefix = f"gfx{cleaned}"
        if with_prefix in gpu_archs:
            return gpu_archs[with_prefix]

    # Fuzzy product/family name lookup
    for record in gpu_archs.values():
        name = record.get("name", "").lower()
        family = record.get("family", "").lower()
        if cleaned in name or cleaned in family or name in cleaned:
            return record

    return None


def get_compatible_rocm(db: Dict, gfx_id: str) -> List[str]:
    """Returns the list of supported ROCm versions for the given gfx ID."""
    gpu = lookup_gpu(db, gfx_id)
    if not gpu:
        return []
    return gpu.get("rocm_versions", [])


def get_compatible_pytorch(db: Dict, gfx_id: str, rocm_version: Optional[str] = None) -> List[str]:
    """
    Returns compatible PyTorch versions, optionally filtered by a specific ROCm version.
    """
    gpu = lookup_gpu(db, gfx_id)
    if not gpu:
        return []

    gpu_pt_versions = gpu.get("pytorch_versions", [])
    if not rocm_version:
        return gpu_pt_versions

    # Filter by ROCm version support matrix
    supported_pt = []
    matrix = db.get("pytorch_rocm_matrix", {})
    for pt_ver, details in matrix.items():
        if pt_ver in gpu_pt_versions:
            supported_rocm_vers = details.get("rocm_versions", [])
            # Simple match or prefix match (e.g. 6.2 matches 6.2)
            if any(rv == rocm_version or rocm_version.startswith(rv) for rv in supported_rocm_vers):
                supported_pt.append(pt_ver)

    return supported_pt


def get_install_command(db: Dict, pytorch_version: str, rocm_version: Optional[str] = None) -> Optional[str]:
    """
    Returns the exact pip install command for the specified PyTorch and ROCm version.
    """
    matrix = db.get("pytorch_rocm_matrix", {})
    
    # Try direct lookup
    details = matrix.get(pytorch_version)
    if not details:
        # Fuzzy match (e.g., "2.5.1" -> "2.5")
        for k, v in matrix.items():
            if pytorch_version.startswith(k):
                details = v
                break

    if not details:
        return None

    # Check if ROCm version matches
    if rocm_version:
        supported_rocm = details.get("rocm_versions", [])
        if not any(rocm_version.startswith(rv) for rv in supported_rocm):
            logger.warning(
                "PyTorch %s is not officially tested on ROCm %s",
                pytorch_version,
                rocm_version,
            )

    return details.get("pip_install")


def check_compatibility(
    db: Dict,
    gfx_id: str,
    rocm_version: Optional[str] = None,
    pytorch_version: Optional[str] = None,
) -> Dict:
    """
    Evaluates compatibility status and returns status details, warnings, and suggestions.
    """
    gpu = lookup_gpu(db, gfx_id)
    result = {
        "compatible": True,
        "gpu_found": gpu is not None,
        "warnings": [],
        "recommendations": [],
    }

    if not gpu:
        result["compatible"] = False
        result["warnings"].append(f"GPU architecture '{gfx_id}' not found in database.")
        return result

    # Check ROCm support range
    gpu_rocm_versions = gpu.get("rocm_versions", [])
    status = gpu.get("status", "supported")

    if status == "legacy":
        result["warnings"].append(
            f"GPU {gpu['name']} ({gpu['gfx_id']}) is legacy. Official support ended at ROCm {gpu.get('max_rocm')}."
        )
    elif status == "limited":
        result["warnings"].append(
            f"GPU {gpu['name']} ({gpu['gfx_id']}) has limited/unofficial support. You may need HSA_OVERRIDE_GFX_VERSION environment variables."
        )

    if rocm_version:
        # Check if version exists in GPU support list
        # Match prefix to allow 6.2.0 to match 6.2
        is_supported_rocm = any(rocm_version.startswith(rv) or rv.startswith(rocm_version) for rv in gpu_rocm_versions)
        if not is_supported_rocm:
            result["compatible"] = False
            result["warnings"].append(
                f"ROCm {rocm_version} is not officially supported on {gpu['name']} ({gpu['gfx_id']})."
            )
            
            # Formulate recommendation
            if gpu.get("max_rocm"):
                result["recommendations"].append(
                    f"Use ROCm {gpu['max_rocm']} or earlier (supported range: {gpu['min_rocm']} to {gpu['max_rocm']})."
                )
            else:
                result["recommendations"].append(
                    f"Use supported ROCm versions: {', '.join(gpu_rocm_versions)}."
                )

    if pytorch_version:
        gpu_pt_versions = gpu.get("pytorch_versions", [])
        pt_base = ".".join(pytorch_version.split(".")[:2]) # e.g. 2.5.1 -> 2.5
        
        if pt_base not in gpu_pt_versions:
            result["warnings"].append(
                f"PyTorch {pytorch_version} is not verified/recommended for {gpu['name']} ({gpu['gfx_id']})."
            )

        if rocm_version:
            compat_pt = get_compatible_pytorch(db, gfx_id, rocm_version)
            if pt_base not in compat_pt:
                result["compatible"] = False
                result["warnings"].append(
                    f"PyTorch {pytorch_version} is not compatible with ROCm {rocm_version} on this GPU."
                )
                if compat_pt:
                    result["recommendations"].append(
                        f"Recommended PyTorch version on ROCm {rocm_version}: {compat_pt[-1]}"
                    )

    # Suggest HSA_OVERRIDE if RX RDNA2/RDNA3 card needs overrides
    if status == "limited" and gpu["gfx_id"] == "gfx1030":
        result["recommendations"].append(
            "Set environment variable: export HSA_OVERRIDE_GFX_VERSION=10.3.0"
        )
    elif status == "limited" and gpu["gfx_id"].startswith("gfx11"):
         result["recommendations"].append(
            "Set environment variable if execution fails: export HSA_OVERRIDE_GFX_VERSION=11.0.0"
        )

    return result


def format_gpu_report(db: Dict, gfx_id: str) -> str:
    """Formats a structured markdown report for a specific GPU's ROCm compatibility."""
    gpu = lookup_gpu(db, gfx_id)
    if not gpu:
        return f"### GPU Architecture {gfx_id} Not Found"

    status_badge = {
        "current": "🟢 Current / Supported",
        "supported": "🟢 Supported",
        "limited": "🟡 Limited / Community Support",
        "legacy": "🔴 Legacy / End-of-Life"
    }.get(gpu['status'], gpu['status'])

    vram_str = "/".join(str(v) for v in gpu.get("vram_gb", [])) + " GB"

    lines = [
        f"### GPU: {gpu['name']} ({gpu['gfx_id']})",
        f"- **Architecture Family:** {gpu.get('family', 'N/A')}",
        f"- **VRAM Configs:** {vram_str}",
        f"- **Support Status:** {status_badge}",
        f"- **ROCm Version Range:** {gpu.get('min_rocm', 'N/A')} to {gpu.get('max_rocm', 'Latest')}",
        f"- **Supported ROCm Versions:** {', '.join(gpu.get('rocm_versions', []))}",
        f"- **Supported PyTorch Versions:** {', '.join(gpu.get('pytorch_versions', []))}",
    ]

    if gpu.get("notes"):
        lines.append(f"- **Notes:** {gpu['notes']}")

    return "\n".join(lines)


def format_compatibility_matrix(db: Dict) -> str:
    """Generates a structured markdown table showing all GPUs and their supported ROCm/PyTorch configurations."""
    gpu_archs = db.get("gpu_architectures", {})
    
    lines = [
        "### GPU Compatibility Matrix",
        "",
        "| gfx ID | GPU Model | Family | Status | Supported ROCm | PyTorch |",
        "| --- | --- | --- | --- | --- | --- |"
    ]

    for gfx, gpu in sorted(gpu_archs.items()):
        status_symbol = {
            "current": "🟢 Current",
            "supported": "🟢 Yes",
            "limited": "🟡 Limited",
            "legacy": "🔴 EOL"
        }.get(gpu['status'], gpu['status'])
        
        rocm_range = f"{gpu.get('min_rocm')} - {gpu.get('max_rocm')}" if gpu.get('max_rocm') else f"{gpu.get('min_rocm')}+"
        pytorch_range = ", ".join(gpu.get("pytorch_versions", []))
        
        lines.append(
            f"| `{gfx}` | {gpu['name']} | {gpu.get('family', 'N/A')} | {status_symbol} | {rocm_range} | {pytorch_range} |"
        )

    return "\n".join(lines)


def get_gpu_by_detected_name(db: Dict, detected_name: str) -> Optional[Dict]:
    """
    Utility to map raw product name patterns from rocm-smi or PyTorch device name
    to structured GPU config record (e.g. 'AMD Instinct MI300X OAM' -> gfx942).
    """
    if not detected_name:
        return None
    
    cleaned = detected_name.upper()

    # Exact product mappings
    mappings = {
        "MI300": "gfx942",
        "MI250": "gfx90a",
        "MI210": "gfx90a",
        "MI100": "gfx908",
        "RADEON VII": "gfx906",
        "MI50": "gfx906",
        "7900": "gfx1100",
        "7800": "gfx1101",
        "7700": "gfx1101",
        "7600": "gfx1102",
        "9070": "gfx1201",
        "9060": "gfx1200",
    }

    for pattern, gfx_id in mappings.items():
        if pattern in cleaned:
            db_archs = db.get("gpu_architectures", {})
            return db_archs.get(gfx_id)

    # Generic fuzzy matching fallback
    return lookup_gpu(db, detected_name)


if __name__ == '__main__':
    # Quick sanity check
    try:
        db = load_gpu_database()
        print("Successfully loaded database:")
        print(format_compatibility_matrix(db))
        print("\nChecking lookup for gfx1100:")
        res = lookup_gpu(db, "gfx1100")
        print(res)
        print("\nChecking compatibility status:")
        print(check_compatibility(db, "gfx1100", "6.2", "2.5"))
    except Exception as err:
        print(f"Error: {err}")
