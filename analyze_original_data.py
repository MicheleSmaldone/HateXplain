# #!/usr/bin/env python
# import json
# import numpy as np
# import argparse
# from collections import Counter, defaultdict

# def majority_label(annots):
#     labels = [a['label'] for a in annots]
#     return Counter(labels).most_common(1)[0][0]

# def flatten(l):
#     return [item for sublist in l for item in sublist]

# def print_counter(c, title, topn=None):
#     print(f"\n{title}:")
#     for i,(k,v) in enumerate(c.most_common(topn)):
#         print(f"  {k:>15} : {v}")
#     if topn and len(c)>topn:
#         print(f"  ... (+{len(c)-topn} more)")

# def main():
#     p = argparse.ArgumentParser(
#         description="Print dataset structure & sanity‐check stats"
#     )
#     p.add_argument("--dataset-json",   default="Data/dataset.json",
#                    help="Mapping post_id→record")
#     p.add_argument("--divisions-json", default="Data/post_id_divisions.json",
#                    help="Train/val/test splits")
#     p.add_argument("--classes-npy",    default="Data/classes.npy",
#                    help="Index→class mapping")
#     args = p.parse_args()

#     # load
#     data      = json.load(open(args.dataset_json))
#     divs      = json.load(open(args.divisions_json))
#     classes   = np.load(args.classes_npy)

#     print(f"\nLoaded {len(data)} examples from {args.dataset_json}")
#     print(f"Class mapping ({len(classes)}): {classes.tolist()}")

#     # gather stats
#     tok_lens        = []
#     annot_counts    = []
#     gold_labels     = []
#     has_rationale   = 0
#     rat_lens        = []

#     for pid, rec in data.items():
#         toks = rec.get("post_tokens", [])
#         tok_lens.append(len(toks))
#         annots = rec.get("annotators", [])
#         annot_counts.append(len(annots))
#         gold_labels.append(majority_label(annots))
#         # combine all annotator masks
#         gold_rats = rec.get("rationales", [])
#         if gold_rats and any(flatten(gold_rats)):
#             has_rationale += 1
#             # flatten OR‐mask
#             or_mask = [int(any(col)) for col in zip(*gold_rats)]
#             rat_lens.append(sum(or_mask))

#     # token length stats
#     print("\npost_tokens length:")
#     print(f"  min {np.min(tok_lens)}, mean {np.mean(tok_lens):.1f}, "
#           f"median {np.median(tok_lens)}, max {np.max(tok_lens)}")

#     # annotator count
#     print_counter(Counter(annot_counts), 
#                   "Number of annotators per example", topn=5)

#     # gold label distribution
#     print_counter(Counter(gold_labels), 
#                   "Majority-vote label distribution")

#     # rationale coverage
#     print(f"\nExamples with ≥1 gold rationale span: {has_rationale} "
#           f"({has_rationale/len(data)*100:.1f}%)")
#     if rat_lens:
#         print(f"  rationale‐span lengths -- "
#               f"min {np.min(rat_lens)}, mean {np.mean(rat_lens):.1f}, "
#               f"median {np.median(rat_lens)}, max {np.max(rat_lens)}")

#     # split sizes & label balance
#     for split in ["train","val","test"]:
#         ids = set(divs.get(split, []))
#         inter = ids & set(data.keys())
#         labs = [majority_label(data[i]["annotators"]) for i in inter]
#         print(f"\n{split.upper():5} → {len(inter)} examples")
#         print_counter(Counter(labs), f"  {split} label distribution")

# if __name__=="__main__":
#     main()
#!/usr/bin/env python
import json
import numpy as np
import argparse
from collections import Counter, defaultdict

def majority_label(annots):
    labels = [a['label'] for a in annots]
    return Counter(labels).most_common(1)[0][0]

def flatten(l):
    return [item for sublist in l for item in sublist]

def print_counter(c, title, topn=None):
    print(f"\n{title}:")
    for i, (k, v) in enumerate(c.most_common(topn)):
        print(f"  {k:>15} : {v}")
    if topn and len(c) > topn:
        print(f"  ... (+{len(c) - topn} more)")


def main():
    p = argparse.ArgumentParser(
        description="Print dataset structure & sanity‐check stats"
    )
    p.add_argument("--dataset-json",   default="Data/dataset.json",
                   help="Mapping post_id→record")
    p.add_argument("--divisions-json", default="Data/post_id_divisions.json",
                   help="Train/val/test splits")
    p.add_argument("--classes-npy",    default="Data/classes.npy",
                   help="Index→class mapping")
    args = p.parse_args()

    # load
    data      = json.load(open(args.dataset_json))
    divs      = json.load(open(args.divisions_json))
    classes   = np.load(args.classes_npy)

    # Structure overview
    print(f"\n=== DATA STRUCTURE OVERVIEW ===")
    print(f"Dataset JSON: {args.dataset_json}")
    print(f"  Total examples: {len(data)}")
    print(f"  Example record keys: {list(next(iter(data.values())).keys())}")

    print(f"\nPost-ID Divisions: {args.divisions_json}")
    print(f"  Splits available: {list(divs.keys())}")
    for split in divs:
        print(f"    {split}: {len(divs[split])} IDs (sample: {divs[split][:3]})")

    print(f"\nClasses mapping (.npy): {args.classes_npy}")
    print(f"  dtype: {classes.dtype}, shape: {classes.shape}")
    print(f"  classes: {classes.tolist()}")

    # gather stats
    tok_lens        = []
    annot_counts    = []
    gold_labels     = []
    has_rationale   = 0
    rat_lens        = []

    for pid, rec in data.items():
        toks = rec.get("post_tokens", [])
        tok_lens.append(len(toks))
        annots = rec.get("annotators", [])
        annot_counts.append(len(annots))
        gold = majority_label(annots) if annots else None
        gold_labels.append(gold)
        # combine all annotator masks
        gold_rats = rec.get("rationales", [])
        if gold_rats and any(flatten(gold_rats)):
            has_rationale += 1
            # flatten OR‐mask
            or_mask = [int(any(col)) for col in zip(*gold_rats)]
            rat_lens.append(sum(or_mask))

    # token length stats
    print(f"\n=== Token Statistics ===")
    print(f"post_tokens length: min={np.min(tok_lens)}, mean={np.mean(tok_lens):.1f}, median={np.median(tok_lens)}, max={np.max(tok_lens)}")

    # annotator count
    print_counter(Counter(annot_counts), 
                  "Number of annotators per example", topn=5)

    # gold label distribution
    print_counter(Counter(filter(None, gold_labels)), 
                  "Majority-vote label distribution")

    # rationale coverage
    print(f"\nExamples with ≥1 gold rationale span: {has_rationale} ({has_rationale/len(data)*100:.1f}%)")
    if rat_lens:
        print(f"  rationale‐span lengths -- min={np.min(rat_lens)}, mean={np.mean(rat_lens):.1f}, median={np.median(rat_lens)}, max={np.max(rat_lens)}")

    # split sizes & label balance
    print(f"\n=== Split Counts & Label Balance ===")
    for split in ["train","val","test"]:
        ids = set(divs.get(split, []))
        inter = ids & set(data.keys())
        labs = [majority_label(data[i]["annotators"]) for i in inter if data.get(i)]
        print(f"{split.upper():5} → {len(inter)} examples")
        print_counter(Counter(labs), f"  {split} label distribution")

if __name__=="__main__":
    main()
