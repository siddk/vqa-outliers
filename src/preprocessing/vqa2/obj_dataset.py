"""
obj_dataset.py

Core script defining VQA-2 Object Dataset Class --> as well as utilities for loading image features from HDF5 files and
tensorizing data.

Reference: https://github.com/hengyuan-hu/bottom-up-attention-vqa/blob/master/dataset.py
"""
from torch.utils.data import Dataset
from transformers import LxmertConfig, LxmertTokenizerFast

import h5py
import json
import numpy as np
import os
import torch


def create_entry(img, question, answer, ans2label, idx=None):
    # Check if answer in ans2label
    if answer["multiple_choice_answer"] in ans2label:
        entry = {
            "question_id": question["question_id"],
            "image_id": question["image_id"],
            "image": img,
            "question": question["question"],
            "answer": ans2label[answer["multiple_choice_answer"]],
            # Index in Sorted VQA-2 Questions JSON
            "idx": idx,
        }

        return True, entry

    else:
        return False, None


def load_dataset(img2idx, ans2label, moderated_indices=None, vqa2_q="data/VQA2-Questions", split="all", mode="train"):
    """Load Dataset Entries"""
    question_path = os.path.join(vqa2_q, "v2_OpenEnded_mscoco_%s2014_questions.json" % mode)
    answer_path = os.path.join(vqa2_q, "v2_mscoco_%s2014_annotations.json" % mode)
    with open(question_path, "r") as f:
        questions = sorted(json.load(f)["questions"], key=lambda x: x["question_id"])
    with open(answer_path, "r") as f:
        answers = sorted(json.load(f)["annotations"], key=lambda x: x["question_id"])
    assert len(questions) == len(answers), "Number of Questions != Number of Answers!"

    # VQA-2 Moderation!
    if moderated_indices is not None:
        print("\t[*] Creating VQA-2 '%s' Entries..." % split)
        entries, indices, moderated = [], [], set(moderated_indices)
        for idx, (question, answer) in enumerate(zip(questions, answers)):
            if idx in moderated:
                assert question["question_id"] == answer["question_id"], "Question ID != Answer ID!"
                assert question["image_id"] == answer["image_id"], "Question Image ID != Answer Image ID"
                in_dataset, entry = create_entry(
                    img2idx["COCO_%s2014_%012d" % (mode, question["image_id"])], question, answer, ans2label, idx=idx
                )
                if in_dataset:
                    entries.append(entry)
                    indices.append(idx)

        return entries, indices

    # Normal VQA-2
    else:
        print("\t[*] Creating VQA-2 '%s' Entries..." % split)
        entries, indices = [], []
        for idx, (question, answer) in enumerate(zip(questions, answers)):
            assert question["question_id"] == answer["question_id"], "Question ID != Answer ID!"
            assert question["image_id"] == answer["image_id"], "Question Image ID != Answer Image ID"
            in_dataset, entry = create_entry(
                img2idx["COCO_%s2014_%012d" % (mode, question["image_id"])], question, answer, ans2label, idx=idx
            )
            if in_dataset:
                entries.append(entry)
                indices.append(idx)

        return entries, indices


class VQAObjectDataset(Dataset):
    def __init__(
        self,
        dictionary,
        ans2label,
        label2ans,
        img2idx,
        moderated_indices=None,
        vqa2_q="data/VQA2-Questions",
        cache="data/VQA2-Cache",
        split="all",
        mode="train",
        lxmert=False,
        lxmert_cache="data/LXMERT",
    ):
        super(VQAObjectDataset, self).__init__()
        self.ans2label, self.label2ans, self.num_ans_candidates = ans2label, label2ans, len(ans2label)
        self.dictionary, self.img2idx = dictionary, img2idx
        self.lxmert, self.lxmert_cache = lxmert, lxmert_cache

        # Load HDF5 Image Features
        self.v_dim, self.s_dim = 2048, 6
        with h5py.File(os.path.join(cache, "%s36.hdf5" % mode), "r") as hf:
            self.features = np.array(hf.get("image_features"))
            self.spatials = np.array(hf.get("spatial_features"))

        # Create the Dataset Entries by Iterating through the Data
        self.entries, self.indices = load_dataset(
            self.img2idx, ans2label, moderated_indices, vqa2_q=vqa2_q, split=split, mode=mode
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

        # Tokenize and Tensorize!
        self.tokenize()
        self.tensorize()

    def tokenize(self, max_length=14):
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
        self.features = torch.from_numpy(self.features)
        self.spatials = torch.from_numpy(self.spatials)

        # Regular Processing
        if not self.lxmert:
            for entry in self.entries:
                question = torch.from_numpy(np.array(entry["q_token"]))
                entry["q_token"] = question

        else:
            # Truncate Spatials to be N x 36 x 4 (instead of 6)
            self.spatials = self.spatials[:, :, :4]

    def __getitem__(self, index):
        entry = self.entries[index]

        # Get Features
        features = self.features[entry["image"]]
        spatials = self.spatials[entry["image"]]
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


# --- VQAObjectIndexDataset
def load_index_dataset(img2idx, ans2label, vqa2_q="data/VQA2-Questions", split="all", mode="train", indices=None):
    """Load Dataset Entries"""
    question_path = os.path.join(vqa2_q, "v2_OpenEnded_mscoco_%s2014_questions.json" % mode)
    answer_path = os.path.join(vqa2_q, "v2_mscoco_%s2014_annotations.json" % mode)
    with open(question_path, "r") as f:
        questions = sorted(json.load(f)["questions"], key=lambda x: x["question_id"])
    with open(answer_path, "r") as f:
        answers = sorted(json.load(f)["annotations"], key=lambda x: x["question_id"])
    assert len(questions) == len(answers), "Number of Questions != Number of Answers!"

    print("\t[*] Creating VQA-2 '%s' Active Learning Entries..." % split)
    entries, selected_indices, set_indices = [], [], set(indices)
    for idx, (question, answer) in enumerate(zip(questions, answers)):
        assert question["question_id"] == answer["question_id"], "Question ID != Answer ID!"
        assert question["image_id"] == answer["image_id"], "Question Image ID != Answer Image ID"
        if indices is not None and idx in set_indices:
            in_dataset, entry = create_entry(
                img2idx["COCO_%s2014_%012d" % (mode, question["image_id"])], question, answer, ans2label, idx=idx
            )
            assert in_dataset, "Something went horribly wrong w/ active learning example selection..."
            entries.append(entry)
            selected_indices.append(idx)

    return entries, selected_indices


class VQAObjectIndexDataset(Dataset):
    def __init__(
        self,
        dictionary,
        ans2label,
        label2ans,
        img2idx,
        indices,
        vqa2_q="data/VQA2-Questions",
        cache="data/VQA2-Cache",
        split="all",
        mode="train",
        lxmert=False,
        lxmert_cache="data/LXMERT",
    ):
        super(VQAObjectIndexDataset, self).__init__()
        self.ans2label, self.label2ans, self.num_ans_candidates = ans2label, label2ans, len(ans2label)
        self.dictionary, self.img2idx = dictionary, img2idx
        self.lxmert, self.lxmert_cache = lxmert, lxmert_cache

        # Load HDF5 Image Features
        self.v_dim, self.s_dim = 2048, 6
        self.features, self.spatials = None, None

        # Create the Dataset Entries by Limiting the Indices
        self.entries, self.indices = load_index_dataset(
            self.img2idx, ans2label, vqa2_q=vqa2_q, split=split, mode=mode, indices=indices
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

        # Tokenize and Tensorize!
        self.tokenize()
        self.tensorize()

    def tokenize(self, max_length=14):
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
        features = self.features[entry["image"]]
        spatials = self.spatials[entry["image"]]
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
