"""
answers.py

Core script for pre-processing GQA Answer Data --> computes a dictionary/inverse mapping between answers and indices.
"""
import json
import os
import pickle


def gqa_create_answers(gqa_q="data/GQA-Questions", cache="data/GQA-Cache"):
    """Create mapping from answers to labels"""

    # Create File Paths and Load from Disk (if cached)
    dfile = os.path.join(cache, "answers.pkl")
    if os.path.exists(dfile):
        with open(dfile, "rb") as f:
            ans2label, label2ans = pickle.load(f)

        return ans2label, label2ans

    ans2label, label2ans = {}, []
    questions = ["train_balanced_questions.json", "val_balanced_questions.json", "testdev_balanced_questions.json"]

    # Iterate through Answer in Question Files and update Mapping
    print("\t[*] Creating Answer Labels from GQA Question/Answers...")
    for qfile in questions:
        qpath = os.path.join(gqa_q, qfile)
        with open(qpath, "r") as f:
            examples = json.load(f)

        for ex_key in examples:
            ex = examples[ex_key]
            if not ex["answer"].lower() in ans2label:
                ans2label[ex["answer"].lower()] = len(ans2label)
                label2ans.append(ex["answer"])

    # Dump Dictionaries to File
    with open(dfile, "wb") as f:
        pickle.dump((ans2label, label2ans), f)

    return ans2label, label2ans
