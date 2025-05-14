#!/usr/bin/env bash
set -euo pipefail

# 1) Make sure our directories exist
mkdir -p Saved explanations_dicts Data

# 2) Download & unzip GloVe
wget http://nlp.stanford.edu/data/glove.42B.300d.zip -P Data/
unzip Data/glove.42B.300d.zip -d Data/
rm Data/glove.42B.300d.zip

# 3) Convert to word2vec, save model, then free memory
python3 - << 'PYCODE'
import warnings, gc
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

warnings.filterwarnings('ignore')
glove2word2vec('Data/glove.42B.300d.txt', 'Data/glove.42B.300d_w2v.txt')

model = KeyedVectors.load_word2vec_format(
    'Data/glove.42B.300d_w2v.txt', 
    binary=False
)
model.save("Data/word2vec.model")

# clean up
del model
gc.collect()
PYCODE

# 4) Remove intermediate files
rm Data/glove.42B.300d.txt Data/glove.42B.300d_w2v.txt

echo "✅ GloVe downloaded, converted to word2vec, and model saved at Data/word2vec.model"

# RUN in terminal: 
# chmod +x prepare_glove.sh (Change the file permission to execution)
# ./prepare_glove.sh

