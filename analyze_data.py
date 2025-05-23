# # import pickle
# # data = pickle.load(open("Data/Total_data_bert_softmax_1_128_3/train_data.pickle","rb"))
# # print(type(data), len(data))
# # print(data[0])




# # import os
# # import pickle
# # import pandas as pd
# # import matplotlib.pyplot as plt

# # # 1) CONFIGURATION
# # DATA_DIR = "Data"
# # SPLIT_DIR = "Total_data_bert_sparsemax_1_128_3"   # change as needed
# # SPLITS = ["train_data.pickle", "val_data.pickle", "test_data.pickle"]

# # # 2) LOADING FUNCTION
# # def load_split(split_fname):
# #     path = os.path.join(DATA_DIR, SPLIT_DIR, split_fname)
# #     with open(path, "rb") as f:
# #         return pickle.load(f)

# # # 3) BUILD A DATAFRAME
# # records = []
# # for split in SPLITS:
# #     data = load_split(split)
# #     for tokens, scores, label in data:
# #         records.append({
# #             "split": split.replace("_data.pickle",""),
# #             "n_tokens": len(tokens),
# #             "label": label,
# #             # you can store tokens and scores if you like:
# #             # "tokens": tokens,
# #             # "scores": scores,
# #         })

# # df = pd.DataFrame(records)

# # # 4) QUICK STATS
# # print(df.groupby("split")["n_tokens"].describe())
# # print(df["label"].value_counts())

# # # 5) PLOTTING
# # plt.figure(figsize=(6,4))
# # df["label"].value_counts().plot.bar()
# # plt.title("Overall Label Distribution")
# # plt.ylabel("Count")
# # plt.tight_layout()
# # plt.show()

# # plt.figure(figsize=(6,4))
# # df["n_tokens"].hist(bins=50)
# # plt.title("Post Length Distribution (tokens)")
# # plt.xlabel("Number of BERT tokens")
# # plt.tight_layout()
# # plt.show()

# # # 6) OPTIONAL: DECODE TOKENS TO TEXT
# # # If you tell me which HuggingFace BERT model / tokenizer you used
# # # (e.g. "bert-base-uncased"), I can add:
# # #
# # # from transformers import AutoTokenizer
# # # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# # # df["text"] = df["tokens"].apply(lambda toks: tokenizer.decode(toks))


# ## SOFTMAX

# #         count       mean        std  min   25%   50%   75%    max
# # split                                                             
# # test    1924.0  28.437110  15.349955  5.0  16.0  25.0  40.0   81.0
# # train  15383.0  28.762335  15.415582  4.0  16.0  26.0  40.0  127.0
# # val     1922.0  28.746618  15.404955  5.0  16.0  25.0  40.0   88.0
# # label
# # normal        7814
# # hatespeech    5935
# # offensive     5480
# # Name: count, dtype: int64


# # import os
# # import pickle
# # import pandas as pd
# # import matplotlib.pyplot as plt

# # # —— CONFIG ——
# # DATA_DIR = "Data"
# # SOFT_DIR = "Total_data_bert_softmax_1_128_3"
# # SPARSE_DIR = "Total_data_bert_sparsemax_1_128_3"
# # SPLITS     = ["train_data.pickle", "val_data.pickle", "test_data.pickle"]

# # def load_labels(dir_name, fname):
# #     """Load [label, label, …] from the given folder+pickle."""
# #     path = os.path.join(DATA_DIR, dir_name, fname)
# #     with open(path, "rb") as f:
# #         data = pickle.load(f)
# #     # each element is (tokens, scores, label)
# #     return [item[2] for item in data]

# # # 1) BUILD A LONG DATAFRAME
# # rows = []
# # for split_fname in SPLITS:
# #     split_name = split_fname.replace("_data.pickle","")
# #     soft_labels   = load_labels(SOFT_DIR,   split_fname)
# #     sparse_labels = load_labels(SPARSE_DIR, split_fname)
# #     assert len(soft_labels) == len(sparse_labels), \
# #         f"length mismatch in {split_name}"
# #     for soft, sparse in zip(soft_labels, sparse_labels):
# #         rows.append({
# #             "split": split_name,
# #             "soft_label": soft,
# #             "sparse_label": sparse,
# #             "changed": soft != sparse
# #         })

# # df = pd.DataFrame(rows)

# # # 2) CROSTAB: soft→sparse
# # ct = pd.crosstab(df["soft_label"], df["sparse_label"])
# # print("=== Label transition counts (soft→sparse) ===\n")
# # print(ct, "\n")

# # # 3) OVERALL CHANGE RATE
# # overall_change = df["changed"].mean()
# # print(f"Overall fraction of posts whose label changed: {overall_change:.2%}\n")

# # # 4) PER-SPLIT CHANGE RATES
# # print("Per-split change rates:")
# # print(df.groupby("split")["changed"].mean().apply(lambda x: f"{x:.2%}"))

# # # 5) PLOTTING (optional)
# # # 5a) Heatmap of crosstab
# # plt.figure(figsize=(6,5))
# # plt.imshow(ct, interpolation="nearest", aspect="auto")
# # plt.xticks(range(len(ct.columns)), ct.columns, rotation=45)
# # plt.yticks(range(len(ct.index)), ct.index)
# # plt.colorbar(label="Count")
# # plt.title("Softmax→Sparsemax label transitions")
# # plt.xlabel("sparse_label")
# # plt.ylabel("soft_label")
# # plt.tight_layout()
# # plt.show()

# # # 5b) Bar chart of change rate by split
# # plt.figure(figsize=(5,3))
# # (df.groupby("split")["changed"].mean()*100).plot.bar()
# # plt.ylabel("Change rate (%)")
# # plt.title("Label-change rate per split")
# # plt.tight_layout()
# # plt.show()

# # === Label transition counts (soft→sparse) ===

# # sparse_label  hatespeech  normal  offensive
# # soft_label                                 
# # hatespeech          5935       0          0
# # normal                 0    7814          0
# # offensive              0       0       5480 

# # Overall fraction of posts whose label changed: 0.00%

# # Per-split change rates:
# # split
# # test     0.00%
# # train    0.00%
# # val      0.00%
# # Name: changed, dtype: object

# import pickle

# DATA_DIR = "Data"
# SOFT_DIR   = "Total_data_bert_softmax_1_128_3"
# SPARSE_DIR = "Total_data_bert_sparsemax_1_128_3"
# SPLIT      = "train_data.pickle"   # or val_data.pickle / test_data.pickle

# # Load both
# soft = pickle.load(open(f"{DATA_DIR}/{SOFT_DIR}/{SPLIT}", "rb"))
# sparse = pickle.load(open(f"{DATA_DIR}/{SPARSE_DIR}/{SPLIT}", "rb"))

# print(f"Total records in this split: {len(soft)} (softmax), {len(sparse)} (sparsemax)\n")

# # How each record is structured
# print("One record looks like:")
# print("  (tokens_list, score_list, label_string)")
# print("Example:")
# print(soft[0])
# print("------------------")
# print(sparse[0])
# print()

# # Print first 5 labels side by side
# print("Index | softmax_label  | sparsemax_label")
# print("------|----------------|-----------------")
# for i in range(5):
#     lbl_soft   = soft[i][2]
#     lbl_sparse = sparse[i][2]
#     print(f"{i:5d} | {lbl_soft:14s} | {lbl_sparse:15s}")


# i = 2  
# print("softmax attention:", soft[i][1][:50])
# print("sparsemax attention:", sparse[i][1][:50])

# diffs = []
# for i, ((_, soft_scores, _), (_, spar_scores, _)) in enumerate(zip(soft, sparse)):
#     # compare floats up to some tolerance
#     if any(abs(a - b) > 1e-6 for a, b in zip(soft_scores, spar_scores)):
#         diffs.append(i)
# print("Found", len(diffs), "posts with different attentions.")
# if diffs:
#     print("First diff at index", diffs[0])

# # Total records in this split: 15383 (softmax), 15383 (sparsemax)

# # One record looks like:
# #   (tokens_list, score_list, label_string)
# # Example:
# # ([101, 2057, 3685, 3613, 4214, 9731, 10469, 2015, 2065, 1996, 2916, 1997, 2035, 24185, 22984, 2078, 4995, 2102, 8280, 2748, 2000, 1037, 4424, 18421, 2270, 2862, 2021, 2097, 1037, 9099, 11690, 22437, 1998, 19483, 24185, 22984, 2078, 2022, 2583, 2000, 4607, 2037, 2592, 2006, 1996, 7316, 7123, 5907, 7057, 102], [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02], 'normal')
# # ------------------
# # ([101, 2057, 3685, 3613, 4214, 9731, 10469, 2015, 2065, 1996, 2916, 1997, 2035, 24185, 22984, 2078, 4995, 2102, 8280, 2748, 2000, 1037, 4424, 18421, 2270, 2862, 2021, 2097, 1037, 9099, 11690, 22437, 1998, 19483, 24185, 22984, 2078, 2022, 2583, 2000, 4607, 2037, 2592, 2006, 1996, 7316, 7123, 5907, 7057, 102], [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02], 'normal')

# # Index | softmax_label  | sparsemax_label
# # ------|----------------|-----------------
# #     0 | normal         | normal         
# #     1 | normal         | normal         
# #     2 | hatespeech     | hatespeech     
# #     3 | hatespeech     | hatespeech     
# #     4 | offensive      | offensive     


import os, pickle, numpy as np

DATA_DIR   = "Data"
SOFT_DIR   = "Total_data_bert_softmax_1_128_3"
SPARSE_DIR = "Total_data_bert_sparsemax_1_128_3"
SPLIT      = "train_data.pickle"

def load_scores(folder):
    with open(os.path.join(DATA_DIR, folder, SPLIT), "rb") as f:
        data = pickle.load(f)
    # extract only the attention vectors
    return [scores for _, scores, _ in data]

soft_scores   = load_scores(SOFT_DIR)
sparse_scores = load_scores(SPARSE_DIR)

def summarize(scores_list):
    n_vecs = len(scores_list)
    lengths = [len(s) for s in scores_list]
    # number of nonzeros per vector
    nonzeros = [np.count_nonzero(s) for s in scores_list]
    # top-1 mass
    top1 = [np.max(s) for s in scores_list]
    # fraction of mass in top-3
    mass_top3 = []
    for s in scores_list:
        i = np.argsort(s)[-3:]
        mass_top3.append(np.sum(np.array(s)[i]))
    return {
        "n_posts": n_vecs,
        "avg_length": np.mean(lengths),
        "avg_nonzero": np.mean(nonzeros),
        "avg_top1": np.mean(top1),
        "avg_mass_top3": np.mean(mass_top3),
    }

soft_summary   = summarize(soft_scores)
sparse_summary = summarize(sparse_scores)

print("Metric          |   softmax   |   sparsemax")
print("-"*43)
for k in soft_summary:
    print(f"{k:15s} | {soft_summary[k]:10.3f} | {sparse_summary[k]:10.3f}")
