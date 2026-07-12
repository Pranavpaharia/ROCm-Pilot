import json
import urllib.request
import urllib.error
import time
import os

# Configuration
LITELLM_URL = "http://localhost:4000/v1/chat/completions"
MODEL_NAME = "omlx/qwen3-coder-30b"
OUTPUT_FILE = "golden_qa_dataset.json"

CATEGORIES_AND_TOPICS = {
    "ROCm installation procedures": [
        "Ubuntu 22.04 apt installation for ROCm 6.0",
        "Adding ROCm user groups (video, render)",
        "Verifying ROCm installation with rocminfo",
        "Installing ROCm PyTorch via pip",
        "Troubleshooting amdkfd module not loaded",
        "Docker installation for ROCm containers",
        "Uninstalling previous ROCm versions",
        "Kernel headers requirement for ROCm DKMS",
        "Setting up ROCm on RHEL/CentOS 9",
        "Checking GPU initialization with rocm-smi"
    ],
    "Matrix operations and GPU compatibility": [
        "What is rocBLAS and its CUDA equivalent?",
        "Supported datatypes (FP16, BF16, FP8) on MI300X",
        "Matrix multiplication performance tuning on ROCm",
        "Using hipBLAS for portable linear algebra",
        "Memory bandwidth differences in matrix operations",
        "Tensor cores equivalent (Matrix Core) on AMD CDNA",
        "Sparsity support in ROCm matrix operations",
        "rocALUTION for sparse linear algebra",
        "Mixed precision training on ROCm",
        "Profiling matrix operations using rocprof"
    ],
    "vLLM setup and configuration": [
        "Installing vLLM for ROCm (pip install vllm)",
        "Environment variables needed (HIP_VISIBLE_DEVICES)",
        "Setting tensor parallel size in vLLM on ROCm",
        "Handling 'hipErrorNoBinaryForGpu' in vLLM",
        "PagedAttention support on AMD GPUs",
        "Running Llama 3 70B on multiple AMD GPUs",
        "Triton backend compatibility in vLLM for ROCm",
        "Docker container setup for vLLM ROCm",
        "vLLM memory allocation (gpu_memory_utilization) on ROCm",
        "Serving an OpenAI compatible API with vLLM on ROCm"
    ],
    "Radeon vs Instinct GPU comparisons": [
        "Architecture difference (RDNA vs CDNA)",
        "Radeon RX 7900 XTX vs Instinct MI210 for AI",
        "Memory types (GDDR6 vs HBM3) impact on LLMs",
        "ROCm official support matrix (Instinct vs Radeon)",
        "Matrix Core capability differences",
        "Peer-to-peer (P2P) memory access differences",
        "Power limits and cooling (Datacenter vs Consumer)",
        "Virtualization (SR-IOV) support",
        "FP64 (Double Precision) compute differences",
        "Form factors and PCIe configurations"
    ],
    "CUDA to ROCm migration guidance": [
        "What is HIP (Heterogeneous-Compute Interface for Portability)?",
        "Using the HIPIFY tool (hipify-perl vs hipify-clang)",
        "Replacing cudaMalloc with hipMalloc",
        "Replacing CUDA streams with HIP streams",
        "Migrating cuBLAS code to rocBLAS/hipBLAS",
        "Migrating cuDNN code to MIOpen",
        "Handling inline PTX assembly in ROCm",
        "Thread block and grid dimension equivalents",
        "Shared memory allocation in HIP",
        "Migrating CUDA events to HIP events"
    ]
}

def generate_qa_pair(category, topic):
    prompt = f"""
    You are an expert in AMD ROCm, GPU computing, and AI infrastructure.
    Please create a single, highly accurate Q&A pair for a benchmark evaluation dataset.
    
    Category: {category}
    Topic: {topic}
    
    Respond ONLY with a valid JSON object matching this exact structure:
    {{
        "question": "A clear, specific question about this topic.",
        "expected_answer": "A detailed, accurate answer.",
        "ground_truth_facts": ["fact 1", "fact 2", "fact 3"],
        "category": "{category}"
    }}
    """
    
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You output strictly valid JSON and nothing else."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"}
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-ikadmin-secure-admin-password"
    }
    req = urllib.request.Request(LITELLM_URL, data=json.dumps(data).encode("utf-8"), headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode())
            content = result["choices"][0]["message"]["content"]
            # Clean up potential markdown formatting
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
            return json.loads(content.strip())
    except Exception as e:
        print(f"  [!] Error generating for topic '{topic}': {e}")
        return None

def main():
    print("Starting ROCm Golden Q&A Dataset Generation (1 by 1)...")
    
    dataset = []
    
    # Load existing if any
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                content = json.load(f)
                dataset = content.get("dataset", [])
            print(f"Loaded {len(dataset)} existing Q&A pairs.")
        except:
            pass

    existing_questions = set([item["question"] for item in dataset])
    
    total = sum(len(topics) for topics in CATEGORIES_AND_TOPICS.values())
    current = 0
    
    for category, topics in CATEGORIES_AND_TOPICS.items():
        for topic in topics:
            current += 1
            print(f"[{current}/{total}] Generating: {topic} ... ", end="", flush=True)
            
            # Simple deduplication based on topic (we assume if we ran this before, we don't want to re-run)
            # A more robust way is to just generate all 50 if the file is empty.
            if len(dataset) >= current:
                print("Skipping (already exists).")
                continue
                
            qa_pair = generate_qa_pair(category, topic)
            if qa_pair:
                dataset.append(qa_pair)
                # Save progressively
                with open(OUTPUT_FILE, 'w') as f:
                    json.dump({"dataset": dataset}, f, indent=2)
                print("Success.")
            else:
                print("Failed.")
                
            time.sleep(2) # brief pause to let model/proxy recover

    print(f"\nDone! Generated {len(dataset)} Q&A pairs in {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
