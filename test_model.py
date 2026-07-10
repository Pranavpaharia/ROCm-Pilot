import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-4-12b-it')
    print('Loading model...')
    model = AutoModelForCausalLM.from_pretrained('google/gemma-4-12b-it', torch_dtype=torch.float16, device_map='auto')
    print('Success!')
except Exception as e:
    import traceback
    traceback.print_exc()
