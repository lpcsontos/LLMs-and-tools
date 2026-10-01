import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer

import os

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else "../models/kondi.pt"
MAX_NEW_TOKENS = 200

TOKENIZER_FOR = {
    "test_gpt_model_v3_ver1.pt": "../tokenizers/tokenizer.json",
    "kondi.pt": "../tokenizers/tokenizer_big.json",
}
DEFAULT_TOKENIZER = "../tokenizers/tokenizer_big_v2.json"

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B, T, C)
        q = self.query(x) # (B, T, C)

        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BLM(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # becomes (B, C)

            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)

if isinstance(ckpt, nn.Module):
    model = ckpt
    vocab_size = model.token_embedding_table.num_embeddings
    block_size = model.position_embedding_table.num_embeddings
else:
    vocab_size = ckpt['vocab_size']
    block_size = ckpt['block_size']
    n_embd = ckpt['n_embd']
    n_head = ckpt['n_head']
    n_layer = ckpt['n_layer']
    dropout = ckpt['dropout']
    model = BLM()
    model.load_state_dict(ckpt['model_state_dict'])

model.to(device)
model.eval()

if len(sys.argv) > 2:
    tokenizer_path = sys.argv[2]
else:
    tokenizer_path = TOKENIZER_FOR.get(os.path.basename(CHECKPOINT), DEFAULT_TOKENIZER)

if vocab_size == 65 and len(sys.argv) <= 2:
    # character-level v2 model: same vocabulary as in model_code/v2.py
    with open("../data/tiny.txt", encoding="utf-8") as f:
        chars = sorted(set(f.read()))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda ids: "".join(itos[i] for i in ids)
else:
    tokenizer = Tokenizer.from_file(tokenizer_path)
    encode = lambda s: tokenizer.encode(s).ids
    decode = lambda ids: tokenizer.decode(ids)
    if tokenizer.get_vocab_size(with_added_tokens=True) != vocab_size:
        sys.exit(f"{tokenizer_path} has {tokenizer.get_vocab_size(with_added_tokens=True)} tokens, "
                 f"but the model expects {vocab_size}. Pass the right tokenizer as the second argument.")


print(f"Loaded {CHECKPOINT} on {device}. Type a prompt (empty line to quit).")
while True:
    prompt = input("> ")
    if not prompt:
        break
    x = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=MAX_NEW_TOKENS)
    print("→ gen text:", decode(out[0].tolist()))
