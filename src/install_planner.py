    def _generate_mi300_plan(self, rocm_version: str) -> List[Dict]:
        """Generate installation plan for MI300 series GPUs."""
        return [
            {
                "step": 1,
                "title": "Verify System Requirements",
                "command": "# Check kernel version\nuname -r | grep '5.15\\|6.0'",
                "verification": "Check if kernel version is 5.15 or higher for ROCm compatibility",
                "type": "check"
            },
            {
                "step": 2,
                "title": "Add ROCm Repository",
                "command": f"""# Add ROCm GPG key
wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
# Add repository
echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/{rocm_version} jammy main" | sudo tee /etc/apt/sources.list.d/rocm.list""",
                "verification": "# Verify repository was added\nls /etc/apt/sources.list.d/rocm.list",
                "type": "setup"
            },
            {
                "step": 3,
                "title": "Update Package Cache",
                "command": "sudo apt update",
                "verification": "# Check that packages were updated\napt list --upgradable 2>/dev/null | grep rocm",
                "type": "setup"
            },
            {
                "step": 4,
                "title": "Install ROCm Development Package",
                "command": f"sudo apt install -y rocm-dev rocm-utils",
                "verification": "# Verify installation with rocminfo\nrocminfo | grep -i rocm",
                "type": "install"
            },
            {
                "step": 5,
                "title": "Verify GPU Access",
                "command": "# Run rocm-smi to check GPU status\nrocm-smi --show-gpu",
                "verification": "# Check that rocm-smi runs without errors\nrocm-smi --show-gpu | head -n 5",
                "type": "verify"
            }
        ]
    
    def _generate_radeon_plan(self, rocm_version: str) -> List[Dict]:
        """Generate installation plan for Radeon GPUs."""
        return [
            {
                "step": 1,
                "title": "Check Kernel Compatibility",
                "command": "# Verify kernel version\nuname -r | grep '5.15\\|6.0'",
                "verification": "# Check kernel requirements\nuname -r",
                "type": "check"
            },
            {
                "step": 2,
                "title": "Install Required Dependencies",
                "command": """# Install basic dependencies
sudo apt update && sudo apt install -y wget gnupg2 ca-certificates""",
                "verification": "# Verify dependencies\nwhich wget",
                "type": "setup"
            },
            {
                "step": 3,
                "title": "Add Radeon ROCm Repository",
                "command": f"""# Add ROCm GPG key
wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
# Add ROCm repository for Radeon
echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/{rocm_version} jammy main" | sudo tee /etc/apt/sources.list.d/rocm.list""",
                "verification": "# Verify repository\nls /etc/apt/sources.list.d/rocm.list",
                "type": "setup"
            },
            {
                "step": 4,
                "title": "Install Radeon ROCm Stack",
                "command": f"sudo apt install -y rocm-dev rocm-utils",
                "verification": "# Verify installation\nrocminfo | grep -i rocm",
                "type": "install"
            }
        ]
        
    def _generate_generic_plan(self, rocm_version: str) -> List[Dict]:
        """Generate generic installation plan."""
        return [
            {
                "step": 1,
                "title": "System Requirements Check",
                "command": "# Ensure kernel is compatible\nuname -r | grep '5.15\\|6.0'",
                "verification": "# Check system info\nlsb_release -a",
                "type": "check"
            },
            {
                "step": 2,
                "title": "Repository Setup",
                "command": f"""# Add ROCm repository key
wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
# Add repository list
echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/{rocm_version} jammy main" | sudo tee /etc/apt/sources.list.d/rocm.list""",
                "verification": "# List repository files\nls /etc/apt/sources.list.d/rocm.list",
                "type": "setup"
            },
            {
                "step": 3,
                "title": "Install ROCm Components",
                "command": f"sudo apt update && sudo apt install -y rocm-dev rocm-utils",
                "verification": "# Verify installation\nrocminfo | grep -i roc",
                "type": "install"
            }
        ]
        
    def generate_install_plan(self, gpu_type: str = "MI300X", rocm_version: str = "6.0") -> Dict[str, Any]:
        """Generate a multi-step installation plan for ROCm."""
        
        # Determine the appropriate plan based on GPU type
        if gpu_type.upper() in ["MI300X", "MI210", "MI250"]:
            plan = self._generate_mi300_plan(rocm_version)
        elif gpu_type.upper() in ["RX7900XTX", "RX7800XT"]:
            plan = self._generate_radeon_plan(rocm_version)
        else:
            plan = self._generate_generic_plan(rocm_version)
            
        return {
            "gpu_type": gpu_type,
            "rocm_version": rocm_version,
            "generated_at": datetime.now().isoformat(),
            "plan_steps": plan
        }