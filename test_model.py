import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-7B-Instruct')
    print('Loading model...')
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-7B-Instruct', torch_dtype=torch.float16, device_map='auto')
    print('Success!')
except Exception as e:
    import traceback
    traceback.print_exc()
