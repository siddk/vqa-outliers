"""
grid_dataset.py

Core script defining VQA-2 Grid Dataset Class --> as well as utilities for loading image features from HDF5 files and
tensorizing data.
"""
import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


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
                img2idx["COCO_%s2014_%012d.jpg" % (mode, question["image_id"])], question, answer, ans2label, idx=idx
            )
            if in_dataset:
                entries.append(entry)
                indices.append(idx)

        return entries, indices


class VQAGridDataset(Dataset):
    def __init__(
        self,
        dictionary,
        ans2label,
        label2ans,
        img2idx,
        moderated_indices=None,
        vqa2_q="data/VQA2-Questions",
        vqa2_g="data/VQA2-Spatials",
        split="all",
        mode="train",
        mtype="glreg",
    ):
        super(VQAGridDataset, self).__init__()
        self.ans2label, self.label2ans, self.num_ans_candidates = ans2label, label2ans, len(ans2label)
        self.dictionary, self.img2idx = dictionary, img2idx

        # Load HDF5 Image Features
        self.dim = 2048
        with h5py.File(os.path.join(vqa2_g, "%s101.hdf5" % mode), "r") as hf:
            self.features = np.array(hf.get("pool"))

        # Create the Dataset Entries by Iterating through the Data
        self.entries, self.indices = load_dataset(
            self.img2idx, ans2label, moderated_indices, vqa2_q=vqa2_q, split=split, mode=mode
        )

        self.tokenize()
        self.tensorize()

    def tokenize(self, max_length=14):
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
        self.features = torch.from_numpy(self.features)
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry["q_token"]))
            entry["q_token"] = question

    def __getitem__(self, index):
        entry = self.entries[index]

        # Get Features
        features = self.features[entry["image"]]
        question = entry["q_token"]
        target = entry["answer"]
        idx = entry["idx"]

        return features, question, target, idx

    def __len__(self):
        return len(self.entries)


# --- VQAGridIndexDataset
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
                img2idx["COCO_%s2014_%012d.jpg" % (mode, question["image_id"])], question, answer, ans2label, idx=idx
            )
            assert in_dataset, "Something went horribly wrong w/ active learning example selection..."
            entries.append(entry)
            selected_indices.append(idx)

    return entries, selected_indices


class VQAGridIndexDataset(Dataset):
    def __init__(
        self,
        dictionary,
        ans2label,
        label2ans,
        img2idx,
        indices,
        vqa2_q="data/VQA2-Questions",
        vqa2_g="data/VQA2-Spatials",
        split="all",
        mode="train",
        mtype="glreg",
    ):
        super(VQAGridIndexDataset, self).__init__()
        self.ans2label, self.label2ans, self.num_ans_candidates = ans2label, label2ans, len(ans2label)
        self.dictionary, self.img2idx = dictionary, img2idx

        # Load HDF5 Image Features
        self.dim = 2048
        self.features = None

        # Create the Dataset Entries by Limiting the Indices
        self.entries, self.indices = load_index_dataset(
            self.img2idx, ans2label, vqa2_q=vqa2_q, split=split, mode=mode, indices=indices
        )

        self.tokenize()
        self.tensorize()

    def tokenize(self, max_length=14):
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
        features = self.features[entry["image"]]
        question = entry["q_token"]
        target = entry["answer"]
        idx = entry["idx"]

        return features, question, target, idx

    def __len__(self):
        return len(self.entries)
