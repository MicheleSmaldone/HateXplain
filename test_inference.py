#!/usr/bin/env python3
"""
Run a few sentences through the SC_weighted_BERT - Sparsemax model.
No command-line arguments required.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# simul_1_softmax_bert-base-uncased_11_6_3_0.001
# simul_1_bert-base-uncased_11_6_3_0.001 (sparsemax)
MODEL_DIR = "./Saved/simul_1_bert-base-uncased_11_6_3_0.001"   # <- change if needed
EXAMPLES  = [
    "I hate you niggerish slut.",
    "What a beautiful day!, I'm happy",
    "Meh, that movie was just okay. Nothing fancy and the director could have tought something more profound",
    "You are just a nigger",
    "I love you",
    "cheers to all my bitcoin retarded bros",
    "you are a total retarded for having this opinion"
]

def main():
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True   # lets HF find your custom SC_weighted_BERT class
    ).to(device).eval()

    batch = tokenizer(EXAMPLES,
                      return_tensors="pt",
                      padding=True,
                      truncation=True).to(device)

    with torch.no_grad():
        logits = model(**batch).logits                     # [B, 3]

    # If you *really* want sparsemax probs, install torch-sparsemax; argmax is enough for labels
    probs = torch.softmax(logits, dim=-1)                 # works for ranking
    preds = probs.argmax(dim=-1)                          # [B]

    id2label = model.config.id2label                      # keys are **ints**
    for sent, idx, pr in zip(EXAMPLES, preds, probs):
        print(f"{id2label[idx.item()]:<8}  {pr.tolist()}  |  {sent}")

if __name__ == "__main__":
    main()
