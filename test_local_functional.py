#!/usr/bin/env python3
import sys
import logging
from src.agent import RocmPilotAgent

# Configure basic logging to see errors clearly
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def run_functional_test():
    print("=" * 60)
    print("🧪 Running Local Model Functional Test (google/gemma-4-12b-it)")
    print("=" * 60)
    print()

    # Step 1: Initialize the Agent with the local GPU provider
    try:
        print("[1/3] Initializing ROCm-Pilot Agent...")
        agent = RocmPilotAgent(
            provider_type="local_gpu",
            model="google/gemma-4-12b-it",
            auto_detect_env=True,
            auto_approve_tools=True  # Auto-approve so the script doesn't hang on input
        )
        print("✅ Agent initialized successfully!\n")
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize agent or local model provider: {e}")
        logging.exception("Initialization Error traceback:")
        sys.exit(1)

    # Step 2: Define the test prompts
    prompts = [
        "What is ROCm?",
        "How do I install PyTorch on MI300X?",
        "Can you run a diagnostic to check my GPU? Please use the tool."
    ]

    # Step 3: Run the prompts through the agent
    print("[2/3] Executing Test Prompts...")
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Prompt {i}: '{prompt}' ---")
        try:
            # We don't stream here to keep the output concise, we just capture the final response
            response = agent.ask(prompt, stream=False)
            
            if not response or not isinstance(response, str):
                print(f"❌ ERROR: Agent returned an invalid response type: {type(response)}")
                sys.exit(1)
                
            print("✅ Response successfully generated:")
            # Print a snippet of the response to verify it looks coherent
            preview = response[:200].replace('\n', ' ') + "..." if len(response) > 200 else response.replace('\n', ' ')
            print(f"   > {preview}")
            
        except Exception as e:
            print(f"\n❌ ERROR: Prompt {i} failed with an exception: {e}")
            logging.exception(f"Execution Error traceback for Prompt {i}:")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ All functional tests passed successfully! The local model is working perfectly.")
    print("=" * 60)

if __name__ == "__main__":
    run_functional_test()
