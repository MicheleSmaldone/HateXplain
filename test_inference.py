#!/usr/bin/env python
"""
Quick inference script for HateXplain BERT models
(softmax or sparsemax attention).

$ python bert_inference.py \
    --ckpt checkpoints/bert_softmax.pt \
    --variant softmax \
    --k 5 \
    --n 10
"""

import argparse, json, torch, numpy as np
from pathlib import Path
from transformers import BertTokenizer
# --- your own model class --------------------------
from Models.bertModels import SC_weighted_BERT          # <-- EDIT if name differs
# ----------------------------------------------------

CLS_ID = 101   # bert-base-uncased CLS

def softmax(x):               # cheap-and-cheerful
    e = np.exp(x - np.max(x))
    return e / e.sum()

@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- load model & tokenizer --------------------------
    print(f"Loading {args.variant} model from {args.ckpt} …")
    model = SC_weighted_BERT.from_pretrained(
        args.ckpt,
        num_labels=3,
        output_attentions=True,        # we want att. for viz
        params={"type_attention": args.variant}
    )
    model.to(device).eval()
    tok = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=False)

    class_names = np.load("Data/classes.npy")           # hate | normal | offensive

    # ---------- sample N test posts -----------------------------
    data = json.load(open("Data/dataset.json"))
    test_ids = set(json.load(open("Data/post_id_divisions.json"))["test"])
    samples = [data[pid] for pid in test_ids if pid in data][:args.n]

    print(f"\nRunning inference on {len(samples)} test posts (variant={args.variant})\n")

    for i, rec in enumerate(samples, 1):
        text_ids = rec["post_tokens"]
        text = tok.decode(text_ids, skip_special_tokens=True)

        ids = torch.tensor([text_ids]).to(device)
        mask = torch.ones_like(ids)          # full mask – tokens already padded in data
        dummy_att = torch.zeros_like(ids)    # not used at inference

        logits, _, att_dict = model(
            ids, attention_vals=dummy_att, attention_mask=mask,
            labels=None, device=device
        )
        probs = softmax(logits.squeeze(0).cpu().numpy())
        pred = class_names[probs.argmax()]

        # ------- fetch CLS attention from top layer ---------------
        att_last = att_dict[-1]              # list(layer)→tensor(batch,h,seq,seq)
        att_cls = att_last[0].mean(0)[0]     # (h,seq,seq) → avg-head over (seq)
        top_idx = att_cls[1:].topk(args.k).indices.cpu().tolist()  # drop CLS itself

        # human-readable sentence w/ rationale marks
        toks = tok.convert_ids_to_tokens(text_ids)
        marked = [f"[{t}]" if j in top_idx else t for j, t in enumerate(toks)]
        print(f"--- sample {i} ------------------------------")
        print("text :", " ".join(marked))
        print("true :", rec["annotators"][0]["label"])        # majority later? keep simple
        print("pred :", pred, " | probs =", dict(zip(class_names, probs.round(3))))
        print()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",   required=True, help="Path to *.pt directory or HF-style folder")
    p.add_argument("--variant", choices=["softmax", "sparsemax"], required=True,
                   help="Type of attention the checkpoint was trained with")
    p.add_argument("-n", type=int, default=5, help="#examples to run")
    p.add_argument("-k", type=int, default=5, help="Top-k tokens to highlight per example")
    args = p.parse_args()
    main(args)
