"""
obj_dataset.py

Core script defining GQA Object Dataset Class --> as well as utilities for loading image features from HDF5 files and
tensorizing data.

Reference: https://github.com/hengyuan-hu/bottom-up-attention-vqa/blob/master/dataset.py
"""
import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import LxmertConfig, LxmertTokenizerFast


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

    print("\t[*] Creating GQA %s Entries..." % mode)
    entries, indices = [], []
    for idx, ex_key in enumerate(sorted(examples)):
        in_dataset, entry = create_entry(examples[ex_key], ex_key, img2idx, ans2label, idx=idx)
        if in_dataset:
            entries.append(entry)
            indices.append(idx)

    return entries, indices


class GQAObjectDataset(Dataset):
    def __init__(
        self,
        dictionary,
        ans2label,
        label2ans,
        img2idx,
        gqa_q="data/GQA-Questions",
        cache="data/GQA-Cache",
        split="all",
        mode="train",
        lxmert=False,
        lxmert_cache="data/LXMERT",
    ):
        super(GQAObjectDataset, self).__init__()
        self.ans2label, self.label2ans, self.num_ans_candidates = ans2label, label2ans, len(ans2label)
        self.dictionary, self.img2idx = dictionary, img2idx
        self.lxmert, self.lxmert_cache = lxmert, lxmert_cache

        # Set Appropriate Prefix
        prefix = "trainval" if mode in "train" else "testdev"

        # Load HDF5 Image Features
        self.v_dim, self.s_dim = 2048, 6
        self.hf = h5py.File(os.path.join(cache, "%s36.hdf5" % prefix), "r")
        self.features = self.hf.get("image_features")
        self.spatials = self.hf.get("spatial_features")

        # Create the Dataset Entries by Iterating though the Data
        self.entries, self.indices = load_dataset(self.img2idx, ans2label, gqa_q=gqa_q, split=split, mode=mode)

        # If LXMERT, create appropriate config & tokenizer...
        if self.lxmert:
            # Create QA Config and set Number of Answers Appropriately
            self.lxmert_config = LxmertConfig.from_pretrained("unc-nlp/lxmert-base-uncased", cache_dir=self.lxmert_cache)
            self.lxmert_config.num_qa_labels = self.num_ans_candidates

            # Create Tokenizer (Rust Fast Tokenizer)
            self.lxmert_tokenizer = LxmertTokenizerFast.from_pretrained(
                "unc-nlp/lxmert-base-uncased", cache_dir=self.lxmert_cache
            )

        self.tokenize()
        self.tensorize()

    def tokenize(self, max_length=40):
        """Tokenize and Front-Pad the Questions in the Dataset"""

        # Regular Processing
        if not self.lxmert:
            for entry in self.entries:
                tokens = self.dictionary.tokenize(entry["question"], False)
                tokens = tokens[:max_length]
                if len(tokens) < max_length:
                    # Note here we pad in front of the sentence
                    padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                    tokens = padding + tokens
                assert len(tokens) == max_length, "Tokenized & Padded Question != Max Length!"
                entry["q_token"] = tokens

        # LXMERT Processing
        else:
            for entry in self.entries:
                inputs = self.lxmert_tokenizer(
                    entry["question"],
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_token_type_ids=True,
                    return_attention_mask=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
                entry["lxmert-inputs"] = inputs

    def tensorize(self):
        # Regular Processing
        if not self.lxmert:
            for entry in self.entries:
                question = torch.from_numpy(np.array(entry["q_token"]))
                entry["q_token"] = question

    def __getitem__(self, index):
        entry = self.entries[index]

        # Get Features
        features = torch.from_numpy(np.array(self.features[entry["image"]]))
        spatials = torch.from_numpy(np.array(self.spatials[entry["image"]]))
        if self.lxmert:
            spatials = spatials[:, :4]
        target = entry["answer"]
        idx = entry["idx"]

        # Regular Processing
        if not self.lxmert:
            question = entry["q_token"]
            return features, spatials, question, target, idx

        else:
            linputs = entry["lxmert-inputs"]
            return linputs.input_ids, linputs.attention_mask, features, spatials, linputs.token_type_ids, target, idx

    def __len__(self):
        return len(self.entries)


# --- GQAObjectIndexDataset
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
            assert in_dataset, "Something went horribly wrong w/ active learning example selection..."
            entries.append(entry)
            selected_indices.append(idx)

    return entries, selected_indices


class GQAObjectIndexDataset(Dataset):
    def __init__(
        self,
        dictionary,
        ans2label,
        label2ans,
        img2idx,
        indices,
        gqa_q="data/GQA-Questions",
        cache="data/GQA-Cache",
        split="all",
        mode="train",
        lxmert=False,
        lxmert_cache="data/LXMERT",
    ):
        super(GQAObjectIndexDataset, self).__init__()
        self.ans2label, self.label2ans, self.num_ans_candidates = ans2label, label2ans, len(ans2label)
        self.dictionary, self.img2idx = dictionary, img2idx
        self.lxmert, self.lxmert_cache = lxmert, lxmert_cache

        # Load HDF5 Image Features (jk not really)
        self.v_dim, self.s_dim = 2048, 6
        self.features, self.spatials = None, None

        # Create the Dataset Entries by Limiting the Indices
        self.entries, self.indices = load_index_dataset(
            self.img2idx, ans2label, gqa_q=gqa_q, split=split, mode=mode, indices=indices
        )

        # If LXMERT, create appropriate config & tokenizer...
        if self.lxmert:
            # Create QA Config and set Number of Answers Appropriately
            self.lxmert_config = LxmertConfig.from_pretrained("unc-nlp/lxmert-base-uncased", cache_dir=self.lxmert_cache)
            self.lxmert_config.num_qa_labels = self.num_ans_candidates

            # Create Tokenizer (Rust Fast Tokenizer)
            self.lxmert_tokenizer = LxmertTokenizerFast.from_pretrained(
                "unc-nlp/lxmert-base-uncased", cache_dir=self.lxmert_cache
            )

        self.tokenize()
        self.tensorize()

    def tokenize(self, max_length=40):
        """Tokenize and Front-Pad the Questions in the Dataset"""

        # Regular Processing
        if not self.lxmert:
            for entry in self.entries:
                tokens = self.dictionary.tokenize(entry["question"], False)
                tokens = tokens[:max_length]
                if len(tokens) < max_length:
                    # Note here we pad in front of the sentence
                    padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                    tokens = padding + tokens
                assert len(tokens) == max_length, "Tokenized & Padded Question != Max Length!"
                entry["q_token"] = tokens

        # LXMERT Processing
        else:
            for entry in self.entries:
                inputs = self.lxmert_tokenizer(
                    entry["question"],
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_token_type_ids=True,
                    return_attention_mask=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
                entry["lxmert-inputs"] = inputs

    def tensorize(self):
        # Regular Processing
        if not self.lxmert:
            for entry in self.entries:
                question = torch.from_numpy(np.array(entry["q_token"]))
                entry["q_token"] = question

    def __getitem__(self, index):
        entry = self.entries[index]

        # Get Features
        features = torch.from_numpy(np.array(self.features[entry["image"]]))
        spatials = torch.from_numpy(np.array(self.spatials[entry["image"]]))[:, :4]
        target = entry["answer"]
        idx = entry["idx"]

        # Regular Processing
        if not self.lxmert:
            question = entry["q_token"]
            return features, spatials, question, target, idx

        else:
            linputs = entry["lxmert-inputs"]
            return linputs.input_ids, linputs.attention_mask, features, spatials, linputs.token_type_ids, target, idx

    def __len__(self):
        return len(self.entries)
