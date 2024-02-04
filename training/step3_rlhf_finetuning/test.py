import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/cpfs/shared/nlp/xllm2.x-models/ckpt-2.3-chat-best"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)

import deepspeed

ds_model = deepspeed.init_inference(
    model,
    mp_size=1,
    dtype=torch.float16,
    replace_method="auto",
    replace_with_kernel_inject=True
)

text = "This is a sample prompt"
tokens = tokenizer.encode(text, return_tensors='pt').to(ds_model.module.device)
_ = model(tokens)