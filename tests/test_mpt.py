#!pip install einops
#!pip install triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python

import torch
import transformers

name = 'mosaicml/mpt-7b-chat'

config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
config.attn_config['attn_impl'] = 'triton'
config.init_device = 'cuda:0' # For fast initialization directly on GPU!

model = transformers.AutoModelForCausalLM.from_pretrained(
  name,
  config=config,
  torch_dtype=torch.bfloat16, # Load model weights in bfloat16
  trust_remote_code=True
)
tokenizer = transformers.AutoTokenizer.from_pretrained(name)
prompt = "say something funny"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
inputs = tokenizer(prompt, return_tensors="pt").to(device)
tokens = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.8)
