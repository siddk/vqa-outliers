#!/usr/bin/env bash
# glove.sh
#   Download GloVe Vectors for pretrained language models (in BUTD, Logistic Regression, etc.).

# Setup
mkdir -p data

# Get GloVe Embeddings
wget -P data http://nlp.stanford.edu/data/glove.6B.zip
unzip data/glove.6B.zip -d data/GloVe
rm data/glove.6B.zip
