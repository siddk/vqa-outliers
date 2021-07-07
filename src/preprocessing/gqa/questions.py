"""
questions.py

Read and preprocess all questions in the GQA Dataset, and create dictionary (for embeddings). Additionally
load and process GloVe embeddings.

Reference: https://github.com/hengyuan-hu/bottom-up-attention-vqa/blob/master/tools/create_dictionary.py
"""
import json
import os
import pickle

import numpy as np


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx, self.idx2word = word2idx, idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        # Lowercase, strip commas and ?, and add space between nouns and 's
        sentence = sentence.lower().replace(",", "").replace(".", "").replace("?", "").replace("'s", " 's")
        words, tokens = sentence.split(), []

        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx.get(w, self.padding_idx))

        return tokens

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def gqa_create_dictionary_glove(
    gqa_q="data/GQA-Questions", glove="data/GloVe/glove.6B.300d.txt", cache="data/GQA-Cache"
):
    """Create Dictionary from GQA Question Files, and Initialize GloVe Embeddings from File"""

    # Create File Paths and Load from Disk (if cached)
    dfile, gfile = os.path.join(cache, "dictionary.pkl"), os.path.join(cache, "glove.npy")
    if os.path.exists(dfile) and os.path.exists(gfile):
        with open(dfile, "rb") as f:
            dictionary = pickle.load(f)

        weights = np.load(gfile)
        return dictionary, weights

    elif not os.path.exists(cache):
        os.makedirs(cache)

    dictionary = Dictionary()
    questions = [
        "train_balanced_questions.json",
        "val_balanced_questions.json",
        "testdev_balanced_questions.json",
        "test_balanced_questions.json",
    ]

    # Iterate through Question in Question Files and update Vocabulary
    print("\t[*] Creating Dictionary from GQA Questions...")
    for qfile in questions:
        qpath = os.path.join(gqa_q, qfile)
        with open(qpath, "r") as f:
            examples = json.load(f)

        for ex_key in examples:
            ex = examples[ex_key]
            dictionary.tokenize(ex["question"], add_word=True)

    # Load GloVe Embeddings
    print("\t[*] Loading GloVe Embeddings...")
    with open(glove, "r") as f:
        entries = f.readlines()

    # Assert that we're using the 300-Dimensional GloVe Embeddings
    assert len(entries[0].split()) - 1 == 300, "ERROR - Not using 300-dimensional GloVe Embeddings!"

    # Create Embedding Weights
    weights = np.zeros((len(dictionary.idx2word), 300), dtype=np.float32)

    # Populate Embedding Weights
    for entry in entries:
        word_vec = entry.split()
        word, vec = word_vec[0], list(map(float, word_vec[1:]))
        if word in dictionary.word2idx:
            weights[dictionary.word2idx[word]] = vec

    # Dump Dictionary and Weights to file
    with open(dfile, "wb") as f:
        pickle.dump(dictionary, f)
    np.save(gfile, weights)

    # Return Dictionary and Weights
    return dictionary, weights
