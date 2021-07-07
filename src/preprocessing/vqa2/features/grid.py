"""
grid.py

Reads in a pickle file mapping VQA-2 Image IDs to Feature IDX in an HDF5 File.

Hierarchy of HDF5 file:
    { 'spatial': num_images x feature_length x patch_sz x patch_sz
      'pool': num_images x feature_length }
"""
import os
import pickle

PATCH_SIZE = 7
FEATURE_LENGTH = 2048


def vqa2_create_grid_features(vqa2_g="data/VQA-Spatials"):
    """Load img2idx mappings from .pkl files"""
    print("\t[*] Loading Grid Img2Idx...")

    # Open and Load Pickle Files
    tid, vid = os.path.join(vqa2_g, "train101_img2idx.pkl"), os.path.join(vqa2_g, "val101_img2idx.pkl")
    with open(tid, "rb") as f:
        train_indices = pickle.load(f)

    with open(vid, "rb") as f:
        val_indices = pickle.load(f)

    return train_indices, val_indices
