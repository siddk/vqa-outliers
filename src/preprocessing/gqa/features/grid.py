"""
grid.py

Reads in a pickle file mapping _all_ GQA Image IDs to Feature IDX in an HDF5 File.

Hierarchy of HDF5 file:
    { 'spatial': num_images x feature_length x patch_sz x patch_sz
      'pool': num_images x feature_length }
"""
import json
import os
import pickle

PATCH_SIZE = 7
FEATURE_LENGTH = 2048


def gqa_create_grid_features(gqa_g="data/GQA-Spatials", gqa_q="data/GQA-Questions", cache="data/GQA-Cache"):
    """Load img2idx mappings from .pkl files"""
    print("\t[*] Loading Grid Img2Idx...")

    # Open Train Questions and TestDev Questions
    with open(os.path.join(gqa_q, "train_balanced_questions.json"), "r") as f:
        train_questions = json.load(f)

    with open(os.path.join(gqa_q, "testdev_balanced_questions.json"), "r") as f:
        testdev_questions = json.load(f)

    # Open and Load Pickle Files
    with open(os.path.join(gqa_g, "gqa101_img2idx.pkl"), "rb") as f:
        all_indices = pickle.load(f)

    # Create train_img2idx and testdev_img2idx
    trainval_img2idx, testdev_img2idx = {}, {}
    for trk in train_questions:
        assert train_questions[trk]["imageId"] in all_indices
        trainval_img2idx[train_questions[trk]["imageId"]] = all_indices[train_questions[trk]["imageId"]]

    for tdk in testdev_questions:
        assert testdev_questions[tdk]["imageId"] in all_indices
        testdev_img2idx[testdev_questions[tdk]["imageId"]] = all_indices[testdev_questions[tdk]["imageId"]]

    return trainval_img2idx, testdev_img2idx
