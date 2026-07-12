"""
Verification Loop System for ROCm-Pilot.
Structured tools system that replaces free-text `[TOOL:]` approach with proper verification checks.
"""

import subprocess
from typing import Dict, List, Any

class VerificationLoop:
    """Structured verification system with proper tool execution."""
    
    def __init__(self):
        pass

    def run_rocm_smi_check(self) -> Dict[str, Any]:
        """Run rocm-smi verification check."""
        try:
            result = subprocess.run(
                ["rocm-smi", "--show-gpu"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                "tool": "rocm-smi",
                "success": result.returncode == 0,
                "output": result.stdout if result.stdout else "",
                "error": result.stderr if result.stderr else ""
            }
        except Exception as e:
            return {
                "tool": "rocm-smi",
                "success": False,
                "output": "",
                "error": str(e)
            }

    def run_pytorch_hip_check(self) -> Dict[str, Any]:
        """Run PyTorch HIP verification check."""
        try:
            result = subprocess.run(
                ["python3", "-c", "import torch; print('HIP available:', torch.cuda.is_available())"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                "tool": "pytorch_hip",
                "success": result.returncode == 0,
                "output": result.stdout if result.stdout else "",
                "error": result.stderr if result.stderr else ""
            }
        except Exception as e:
            return {
                "tool": "pytorch_hip", 
                "success": False,
                "output": "",
                "error": str(e)
            }

    def run_kernel_check(self) -> Dict[str, Any]:
        """Run kernel version check."""
        try:
            result = subprocess.run(
                ["uname", "-r"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            return {
                "tool": "kernel_check",
                "success": result.returncode == 0,
                "output": result.stdout if result.stdout else "",
                "error": result.stderr if result.stderr else ""
            }
        except Exception as e:
            return {
                "tool": "kernel_check",
                "success": False,
                "output": "",
                "error": str(e)
            }

    def run_verification_suite(self) -> List[Dict[str, Any]]:
        """Run a complete verification suite for ROCm system."""
        results = []
        
        # Run all tools in sequence
        results.append(self.run_rocm_smi_check())
        results.append(self.run_pytorch_hip_check()) 
        results.append(self.run_kernel_check())
        
        return results

    def format_report(self, results: List[Dict[str, Any]]) -> str:
        """Format verification report as readable text."""
        output = []
        output.append("🔍 ROCm System Verification Report")
        output.append("=" * 50)
        
        all_passed = all(r['success'] for r in results)
        
        if all_passed:
            output.append("✅ Overall Status: PASSED")
        else:
            output.append("❌ Overall Status: FAILED")
            
        for result in results:
            output.append(f"\nTool: {result['tool']}")
            if result['success']:
                output.append("  Status: ✅ PASSED")
            else:
                output.append("  Status: ❌ FAILED")
                
            if result['error']:
                output.append(f"  Error: {result['error']}")
                
        return "\n".join(output)

# Export main function for easy use
def run_verification_check() -> str:
    """Main function to execute ROCm verification system."""
    verifier = VerificationLoop()
    results = verifier.run_verification_suite()
    return verifier.format_report(results)

if __name__ == "__main__":
    print("=== ROCm Verification System ===")
    report = run_verification_check()
    print(report)