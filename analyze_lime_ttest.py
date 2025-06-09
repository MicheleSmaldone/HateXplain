#!/usr/bin/env python
import argparse
import json
from collections import defaultdict

import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import precision_recall_fscore_support

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_lime(path):
    with open(path, 'r') as f:
        for line in f:
            yield json.loads(line)

def majority_label(annots):
    labels = [a['label'] for a in annots]
    return max(set(labels), key=labels.count)

def combine_gold_rationales(rationale_lists):
    if not rationale_lists:
        return []
    # OR across annotator masks
    return [int(any(col)) for col in zip(*rationale_lists)]

def mask_from_spans(spans, length):
    m = [0] * length
    for s in spans:
        for i in range(s['start_token'], min(s['end_token'], length)):
            m[i] = 1
    return m

def extract_metrics(lime_path, data, divs):
    test_ids = set(divs['test'])
    suff, comp = [], []
    precs, recs, f1s = [], [], []

    for rec in load_lime(lime_path):
        pid = rec["annotation_id"]
        if pid not in test_ids:
            continue
        record = data[pid]

        # gold label index
        gold_label = majority_label(record["annotators"])

        # sufficiency/comprehensiveness drops
        orig_conf = rec["classification_scores"][gold_label]
        suff_conf = rec["sufficiency_classification_scores"][gold_label]
        comp_conf = rec["comprehensiveness_classification_scores"][gold_label]
        suff.append(orig_conf - suff_conf)
        comp.append(orig_conf - comp_conf)

        # gold rationale mask (from dataset.json)
        gold_mask = combine_gold_rationales(record["rationales"])

        # predicted mask from LIME spans!
        spans = rec["rationales"][0]["hard_rationale_predictions"]
        pred_mask = mask_from_spans(spans, len(gold_mask))

        p, r, f1, _ = precision_recall_fscore_support(
            gold_mask, pred_mask, average='binary', zero_division=0
        )
        precs.append(p)
        recs.append(r)
        f1s.append(f1)

    return {
        "suff": np.array(suff),
        "comp": np.array(comp),
        "prec": np.array(precs),
        "rec": np.array(recs),
        "f1": np.array(f1s),
    }

def run_tests(name, a, b):
    d = a - b
    mean_a, std_a = a.mean(), a.std(ddof=1)
    mean_b, std_b = b.mean(), b.std(ddof=1)
    mean_d, std_d = d.mean(), d.std(ddof=1)

    t_stat, p_t = ttest_rel(a, b)
    w_stat, p_w = wilcoxon(a, b)
    cohen_d = mean_d / std_d if std_d > 0 else float('nan')

    print(f"\n--- {name} ---")
    print(f"softmax   mean={mean_a:.4f}, std={std_a:.4f}")
    print(f"sparsemax mean={mean_b:.4f}, std={std_b:.4f}")
    print(f"mean Δ    = {mean_d:.4f}   (soft−sparse)")
    print(f"paired t  = {t_stat:.3f}, p(t)={p_t:.3e}")
    print(f"Wilcoxon  = {w_stat:.3f}, p(w)={p_w:.3e}")
    print(f"Cohen’s d = {cohen_d:.3f}")

def main():
    p = argparse.ArgumentParser(
        description="Paired t-test & Wilcoxon on LIME suff/comp/rationale metrics"
    )
    p.add_argument("--dataset-json",   required=True,
                   help="path to Data/dataset.json")
    p.add_argument("--divisions-json", required=True,
                   help="path to Data/post_id_divisions.json")
    p.add_argument("--soft-lime",      required=True,
                   help="LIME JSONL from softmax model")
    p.add_argument("--sparse-lime",    required=True,
                   help="LIME JSONL from sparsemax model")
    args = p.parse_args()

    data = load_json(args.dataset_json)
    divs = load_json(args.divisions_json)

    met_soft   = extract_metrics(args.soft_lime,   data, divs)
    met_sparse = extract_metrics(args.sparse_lime, data, divs)

    n = len(met_soft["suff"])
    assert n == len(met_sparse["suff"]), \
        f"Mismatch in paired sample count: {n} vs {len(met_sparse['suff'])}"
    print(f"→ Found {n} paired test samples.\n")

    run_tests("Sufficiency drop",      met_soft["suff"], met_sparse["suff"])
    run_tests("Comprehensiveness drop", met_soft["comp"], met_sparse["comp"])
    run_tests("Rationale precision",    met_soft["prec"], met_sparse["prec"])
    run_tests("Rationale recall",       met_soft["rec"],  met_sparse["rec"])
    run_tests("Rationale F1",           met_soft["f1"],   met_sparse["f1"])

if __name__ == "__main__":
    main()
