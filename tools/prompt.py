import torch
import random
from model import *
from tokenizers import Tokenizer



model = torch.load("../modela/test_gpt_model_v3_ver1.pt", map_location='cuda', weights_only=False)


tokenizer = Tokenizer.from_file("../tokenizers/tokenizer.json")
encode = lambda s: tokenizer.encode(s).ids
decode = lambda ids: tokenizer.decode(ids)


prompt = input()
x = torch.tensor([encode(prompt)], dtype=torch.long, device='cuda')

for i in range(7):
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=int(len(prompt) * random.uniform(0.8, 2.5)))

    print("→ gen text:", decode(out[0].tolist()))
