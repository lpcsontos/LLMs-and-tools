from datasets import load_dataset
import json, random, glob
from tqdm import tqdm

ds = load_dataset("../data/lmsys-chat-1m/data", split="train")

out = []
for convo in ds:
    msgs = convo["conversation"]
    if len(msgs) >= 2:
        for i in range(len(msgs)-1):
            u = msgs[i]
            a = msgs[i+1]
            if u.get("role") == "user" and a.get("role") == "assistant":
                inst = u.get("content", "").strip()
                resp = a.get("content", "").strip()
                if inst and resp:
                    out.append({"instruction": inst,
                                "context": "",
                                "response": resp})

random.shuffle(out)


out = [{"instruction": f"{i}", "context": "", "response": f"{i}"} for i in range(1, 100001)]

with open("../data/lmsys.jsonl", "w", encoding="utf-8") as f:
    for rec in tqdm(out, desc="Write lmsys.jsonl", unit="rekord"):
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


files = glob.glob("../data/*.jsonl")
recs = []
for fname in tqdm(files, desc="Read files", unit="file"):
    with open(fname, encoding="utf-8") as f:
        for ln in f:
            recs.append(json.loads(ln))


random.shuffle(recs)

with open("../data/data.jsonl", "w", encoding="utf-8") as fout:
    for r in tqdm(recs, desc="Write data.jsonl", unit="rekord"):
        fout.write(json.dumps(r, ensure_ascii=False) + "\n")

