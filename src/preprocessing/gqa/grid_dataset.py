"""
grid_dataset.py

Core script defining GQA Grid Dataset Class --> as well as utilities for loading image features from HDF5 files and
tensorizing data.
"""
import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def create_entry(example, qid, img2idx, ans2label, idx=None):
    # Get Valid Image ID
    img_id = example["imageId"]
    assert img_id in img2idx, "Image ID not in Index!"

    entry = {
        "question_id": qid,
        "image_id": img_id,
        "image": img2idx[img_id],
        "question": example["question"],
        "answer": ans2label[example["answer"].lower()],
        # Index in Sorted GQA JSON
        "idx": idx,
    }

    return True, entry


def load_dataset(img2idx, ans2label, gqa_q="data/GQA-Questions", split="all", mode="train"):
    """Load Dataset Entries"""
    question_path = os.path.join(gqa_q, "%s_balanced_questions.json" % mode)
    with open(question_path, "r") as f:
        examples = json.load(f)

    print("\t[*] Create GQA %s Entries..." % mode)
    entries, indices = [], []
    for idx, ex_key in enumerate(sorted(examples)):
        in_dataset, entry = create_entry(examples[ex_key], ex_key, img2idx, ans2label, idx=idx)
        if in_dataset:
            entries.append(entry)
            indices.append(idx)

    return entries, indices


class GQAGridDataset(Dataset):
    def __init__(
        self,
        dictionary,
        ans2label,
        label2ans,
        img2idx,
        gqa_q="data/GQA-Questions",
        gqa_g="data/GQA-Spatials",
        cache="data/GQA-Cache",
        split="all",
        mode="train",
    ):
        super(GQAGridDataset, self).__init__()
        self.ans2label, self.label2ans, self.num_ans_candidates = ans2label, label2ans, len(ans2label)
        self.dictionary, self.img2idx = dictionary, img2idx

        # Load HDF5 Image Features
        self.dim = 2048
        self.hf = h5py.File(os.path.join(gqa_g, "gqa101.hdf5"), "r")
        self.features = self.hf.get("pool")

        # Create the Dataset Entries by Iterating through the Data
        self.entries, self.indices = load_dataset(self.img2idx, ans2label, gqa_q=gqa_q, split=split, mode=mode)

        self.tokenize()
        self.tensorize()

    def tokenize(self, max_length=40):
        """Tokenize and Front-Pad the Questions in the Dataset"""
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry["question"], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            assert len(tokens) == max_length, "Tokenized & Padded Question != Max Length!"
            entry["q_token"] = tokens

    def tensorize(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry["q_token"]))
            entry["q_token"] = question

    def __getitem__(self, index):
        entry = self.entries[index]

        # Get Features
        features = torch.from_numpy(np.array(self.features[entry["image"]]))
        question = entry["q_token"]
        target = entry["answer"]
        idx = entry["idx"]

        return features, question, target, idx

    def __len__(self):
        return len(self.entries)


# --- GQAGridIndexDataset
def load_index_dataset(img2idx, ans2label, gqa_q="data/GQA-Questions", split="all", mode="train", indices=None):
    """Load Dataset Entries"""
    question_path = os.path.join(gqa_q, "%s_balanced_questions.json" % mode)
    with open(question_path, "r") as f:
        examples = json.load(f)

    print("\t[*] Creating GQA '%s' Active Learning Entries..." % split)
    entries, selected_indices, set_indices = [], [], set(indices)
    for idx, ex_key in enumerate(sorted(examples)):
        if indices is not None and idx in set_indices:
            in_dataset, entry = create_entry(examples[ex_key], ex_key, img2idx, ans2label, idx=idx)
            assert in_dataset, "Something went horribly wrong w/ active learning example selection"
            entries.append(entry)
            selected_indices.append(idx)

    return entries, selected_indices


class GQAGridIndexDataset(Dataset):
    def __init__(
        self,
        dictionary,
        ans2label,
        label2ans,
        img2idx,
        indices,
        gqa_q="data/GQA-Questions",
        gqa_g="data/GQA-Spatials",
        cache="data/GQA-Cache",
        split="all",
        mode="train",
    ):
        super(GQAGridIndexDataset, self).__init__()
        self.ans2label, self.label2ans, self.num_ans_candidates = ans2label, label2ans, len(ans2label)
        self.dictionary, self.img2idx = dictionary, img2idx

        # Load HDF5 Image Features (jk not really)
        self.dim = 2048
        self.features = None

        # Create the Dataset Entries by Limiting the Indices
        self.entries, self.indices = load_index_dataset(
            self.img2idx, ans2label, gqa_q=gqa_q, split=split, mode=mode, indices=indices
        )

        self.tokenize()
        self.tensorize()

    def tokenize(self, max_length=40):
        """Tokenize and Front-Pad the Questions in the Dataset"""
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry["question"], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            assert len(tokens) == max_length, "Tokenized & Padded Question != Max Length!"
            entry["q_token"] = tokens

    def tensorize(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry["q_token"]))
            entry["q_token"] = question

    def __getitem__(self, index):
        entry = self.entries[index]

        # Get Features
        features = torch.from_numpy(np.array(self.features[entry["image"]]))
        question = entry["q_token"]
        target = entry["answer"]
        idx = entry["idx"]

        return features, question, target, idx

    def __len__(self):
        return len(self.entries)
