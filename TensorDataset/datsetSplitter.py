import sys
import os
from tqdm import tqdm
import numpy as np
sys.path.append('../')
from torch.utils.data import Dataset
import pandas as pd
from Preprocess.dataCollect import collect_data, set_name
from sklearn.model_selection import train_test_split
from os import path
from gensim.models import KeyedVectors
import pickle
import json

class Vocab_own():
    def __init__(self, dataframe, model):
        self.itos = {}
        self.stoi = {}
        self.vocab = {}
        self.embeddings = []
        self.dataframe = dataframe
        self.model = model

    def load_embeddings(self, word):
        try:
            return self.model[word], word
        except KeyError:
            return self.model['unk'], 'unk'

    def create_vocab(self):
        count = 1
        for _, row in tqdm(self.dataframe.iterrows(), total=len(self.dataframe)):
            for word in row['Text']:
                vector, token = self.load_embeddings(word)
                if token in self.vocab:
                    self.vocab[token] += 1
                else:
                    self.vocab[token] = 1
                    self.stoi[token] = count
                    self.itos[count] = token
                    self.embeddings.append(vector)
                    count += 1

        # add padding token
        self.vocab['<pad>'] = 1
        self.stoi['<pad>'] = 0
        self.itos[0] = '<pad>'
        self.embeddings.append(np.zeros((300,), dtype=float))
        self.embeddings = np.array(self.embeddings)
        print("Embeddings matrix shape:", self.embeddings.shape)


def encodeData(dataframe, vocab, params):
    tuple_new_data = []
    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        if params['bert_tokens']:
            tuple_new_data.append((row['Text'], row['Attention'], row['Label']))
        else:
            token_ids = []
            for word in row['Text']:
                idx = vocab.stoi.get(word, vocab.stoi['unk'])
                token_ids.append(idx)
            tuple_new_data.append((token_ids, row['Attention'], row['Label']))

    # only dump vocab if it exists (i.e., non-BERT branch)
    if vocab is not None:
        with open('stoi.json', 'w') as f:
            json.dump(vocab.stoi, f)
        with open('itos.json', 'w') as f:
            json.dump(vocab.itos, f)

    return tuple_new_data


def createDatasetSplit(params):
    print(">>> createDatasetSplit params['bert_tokens'] =", params['bert_tokens'])

    # # ────── Hack: force softmax pickle for sparsemax runs ──────
    # if params.get('type_attention') == 'sparsemax':
    #     print("⚠️  Overriding type_attention sparsemax → softmax so we can load the existing pickle")
    #     params['type_attention'] = 'softmax'
    # # ────────────────────────────────────────────────────────────

    filename = set_name(params)
    print("----------------------------------------")
    print(">> Using data pickle:", filename)

    if not path.exists(filename):
        dataset = collect_data(params)

    # if precomputed folder exists, load picks
    cache_dir = filename[:-7]
    print(">> Loading cached splits from:", cache_dir)
    print("----------------------------------------")

    if path.exists(cache_dir):
        with open(os.path.join(cache_dir, 'train_data.pickle'), 'rb') as f:
            X_train = pickle.load(f)
        with open(os.path.join(cache_dir, 'val_data.pickle'), 'rb') as f:
            X_val = pickle.load(f)
        with open(os.path.join(cache_dir, 'test_data.pickle'), 'rb') as f:
            X_test = pickle.load(f)

        vocab_own = None
        if not params['bert_tokens']:
            with open(os.path.join(cache_dir, 'vocab_own.pickle'), 'rb') as f:
                vocab_own = pickle.load(f)
    else:
        # build from scratch
        if not params['bert_tokens']:
            word2vecmodel1 = KeyedVectors.load("Data/word2vec.model")
            _ = word2vecmodel1['easy']  # sanity check

        dataset = pd.read_pickle(filename)
        with open('Data/post_id_divisions.json', 'r') as fp:
            post_id_dict = json.load(fp)

        X_train = dataset[dataset['Post_id'].isin(post_id_dict['train'])]
        X_val   = dataset[dataset['Post_id'].isin(post_id_dict['val'])]
        X_test  = dataset[dataset['Post_id'].isin(post_id_dict['test'])]

        if params['bert_tokens']:
            vocab_own = None
        else:
            vocab_own = Vocab_own(X_train, word2vecmodel1)
            vocab_own.create_vocab()

        X_train = encodeData(X_train, vocab_own, params)
        X_val   = encodeData(X_val,   vocab_own, params)
        X_test  = encodeData(X_test,  vocab_own, params)

        print("total dataset size:", len(X_train) + len(X_val) + len(X_test))

        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, 'train_data.pickle'), 'wb') as f:
            pickle.dump(X_train, f)
        with open(os.path.join(cache_dir, 'val_data.pickle'), 'wb') as f:
            pickle.dump(X_val, f)
        with open(os.path.join(cache_dir, 'test_data.pickle'), 'wb') as f:
            pickle.dump(X_test, f)

        if not params['bert_tokens']:
            with open(os.path.join(cache_dir, 'vocab_own.pickle'), 'wb') as f:
                pickle.dump(vocab_own, f)

    if not params['bert_tokens']:
        return X_train, X_val, X_test, vocab_own
    else:
        return X_train, X_val, X_test
