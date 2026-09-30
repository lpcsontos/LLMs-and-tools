# LLMs and tools

A small GPT-style language model written from scratch in PyTorch, together with the tools around it: a BPE tokenizer, a data preparation script and an instruction fine-tuning script.

The project started from Andrej Karpathy's ["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY) lecture. The first version (`v2.py`) follows it closely; later versions extend it with a trained BPE tokenizer, a much larger corpus, mixed-precision training, learning-rate scheduling, early stopping, checkpointing and a supervised fine-tuning step.

## Architecture

A decoder-only transformer with pre-LayerNorm blocks:

| Parameter | v2 (character-level) | v3 (BPE) |
|---|---|---|
| Tokenizer | characters | byte-level BPE, 8,000 tokens |
| Layers | 8 | 8 |
| Attention heads | 8 | 8 |
| Embedding size | 384 | 384 |
| Context length | 256 tokens | 384 tokens |
| Dropout | 0.10 | 0.15 |
| Parameters | ~14 M | ~20.5 M |

Each block has causal multi-head self-attention and a feed-forward layer (4× expansion, ReLU), both with residual connections. Token and learned position embeddings are summed at the input, and a final LayerNorm and linear head produce the next-token logits.

## Repository structure

```
data/
  tiny.txt            Tiny Shakespeare (~1 MB)
  out.txt             training corpus: Tiny Shakespeare + 300+ books (~92 MB)
  lmsys.jsonl         instruction/response pairs for fine-tuning (placeholder data)
model_code/
  v2.py               character-level model, trained on Tiny Shakespeare
  v3.py               BPE model, trained on the full corpus
tokenizers/
  tokenizer*.json     trained BPE tokenizers (Hugging Face `tokenizers` format)
models/
  *.pt                saved model checkpoints
tools/
  tok_gen.py          trains a BPE tokenizer on the corpus
  data_concat.py      converts chat data into instruction/response JSONL and merges files
  kondicionizer.py    instruction fine-tuning of a pretrained model
  prompt.py, prompt2.py   generate text from a trained model
```

## Training details

**Tokenizer.** Byte-level BPE trained with the Hugging Face `tokenizers` library on the full corpus, with a vocabulary of 8,000 and the special tokens `<pad>`, `<unk>`, `<sos>` and `<eos>`.

**Pretraining (v3).** Random 384-token windows from a 90/10 train/validation split, batch size 128, AdamW (lr 3e-4, weight decay 0.01), a OneCycle cosine schedule, gradient clipping at 1.0 and automatic mixed precision. Validation loss is checked every 500 steps; the best checkpoint is saved, and training stops early after 10 evaluations without improvement.

**Fine-tuning.** `kondicionizer.py` loads the pretrained model and trains it on `instruction → response` pairs. The prompt and response are joined with `<eos>` tokens, and the loss is masked (`-100`) on the prompt, so the model only learns to predict the response.

**Hardware.** Everything was trained on a single NVIDIA RTX 4070 Super. The v3 model trained for roughly 4–5 days continuously.

## Results

The model learned to produce text with realistic structure and vocabulary, and at times it wrote coherent (and sometimes quite funny) passages. At ~20 M parameters and without large-scale instruction data, however, it is not usable as an assistant: its output often drifts off topic or loses meaning. The main value of the project was understanding every step of the pipeline, from tokenization to fine-tuning, by building it by hand.

## Running it

Requirements: Python 3.10+, a CUDA-capable GPU, and

```
pip install torch tokenizers datasets tqdm
```

The scripts use relative paths, so run them from their own folder:

```
cd model_code
python v2.py          # character-level model on Tiny Shakespeare
python v3.py          # tokenizer + BPE model (expects out.txt in the working directory)

cd ../tools
python kondicionizer.py   # fine-tune a pretrained checkpoint
python prompt2.py         # type a prompt, get generated text
```

## Possible improvements

- Move hyperparameters and the model class into a shared module and config file instead of repeating them in every script
- Use `torch.nn.functional.scaled_dot_product_attention` for faster attention
- Evaluate the fine-tuning on a held-out instruction set instead of the pretraining data
- Host model weights on the Hugging Face Hub instead of in the repository
