import json, torch
from tokenizers import Tokenizer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from itertools import cycle
from torch.optim.lr_scheduler import ReduceLROnPlateau



tokenizer = Tokenizer.from_file("../tokenizers/tokenizer_big_v2.json")
encode = lambda s: tokenizer.encode(s).ids
decode = lambda ids: tokenizer.decode(ids)

vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)


batch_size = 128
block_size = 384
max_iters = 35000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 300
n_embd = 384
n_head = 8
n_layer = 8
dropout = 0.13

with open("../data/out.txt", encoding="utf-8") as f:
    raw = f.read()
data = torch.tensor(encode(raw), dtype=torch.long)
n = int(0.8 * len(data))
train_data = data[:n]
val_data   = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


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

        if T > block_size:
            raise ValueError(f"Sequence length T={T} exceeds block_size={block_size}")

        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        positions = torch.arange(T, device=device)
        pos_emb = self.position_embedding_table(positions)# (T, C)
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


model = torch.load("../models/test_gpt_model_v3_ver1.pt", map_location='cuda', weights_only=False)


IGNORE = -100

class ChatFinetuneDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, block_size):
        self.examples = []
        pad_id = tokenizer.token_to_id("<pad>")
        eos_id = tokenizer.token_to_id("<eos>")

        for ln in open(jsonl_file, encoding="utf-8"):
            rec = json.loads(ln)
            inst = rec.get("instruction","").strip()
            ctx  = rec.get("context","").strip()
            resp = rec.get("response","").strip()


            prompt = inst
            if ctx:
                prompt += "\n\n" + ctx

            prompt += tokenizer.id_to_token(eos_id)

            prompt += resp + tokenizer.id_to_token(eos_id)
            ids = tokenizer.encode(prompt).ids

            if len(ids) > block_size:
                ids = ids[-block_size:]
            else:
                ids = [pad_id] * (block_size - len(ids)) + ids


            self.examples.append(ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = torch.tensor(self.examples[idx], dtype=torch.long)
        x = ids[:-1]
        y = ids[1:].clone()
        eos_id = tokenizer.token_to_id("<eos>")
        try:
            e = ids.tolist().index(eos_id)
            prompt_len = e + 1
        except ValueError:
            prompt_len = 0

        mask = torch.zeros_like(y, dtype=torch.bool)
        mask[prompt_len:] = True
        y = torch.where(mask, y, torch.tensor(IGNORE, device=y.device))
        return x, y

train_ds = ChatFinetuneDataset("../data/data.jsonl", tokenizer, block_size)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=learning_rate*0.5,
    total_steps=max_iters,
    pct_start=0.3,
    anneal_strategy='cos',
)

scheduler_plateau = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=4,
    threshold=1e-4,
    cooldown=2,
    min_lr=1e-6,
)


model.train()
total_loss = 0.0
train_iter = cycle(train_loader)

for iter in range(1, max_iters + 1):
    xb, yb = next(train_iter)
    xb, yb = xb.to(device), yb.to(device)

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    total_loss += loss.item()

    if iter % eval_interval == 0 or iter == max_iters:
        losses = estimate_loss()
        exp_lr = scheduler.get_last_lr()[0]
        avg = total_loss / iter
        print(f"[{iter}/{max_iters}] "
              f"lr={exp_lr:.3e}  "
              f"train-loss≈{avg:.4f}  "
              f"val-loss={losses['val']:.4f}")
        scheduler_plateau.step(losses["val"])
        


torch.save(model, '../models/kondi.pt')

load_model = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': loss,
    'batch_size': batch_size,
    'block_size': block_size,
    'max_iters': max_iters,
    'eval_interval': eval_interval,
    'learning_rate': learning_rate,
    'device': device,
    'eval_iters': eval_iters,
    'n_embd': n_embd,
    'n_head': n_head,
    'n_layer': n_layer,
    'dropout': dropout,
    'vocab_size': vocab_size,
    'stoi': tokenizer.get_vocab(),   
    'itos': {v: k for k, v in tokenizer.get_vocab().items()},
}

torch.save(load_model, '../models/kondi_ver2.pth')
