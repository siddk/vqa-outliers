"""
object.py

Reads in a tsv file with pre-trained bottom up attention (object) features and writes them to an hdf5 file.
Additionally builds image ID --> Feature IDX Mapping.

Hierarchy of HDF5 file:
    { 'image_features': num_images x num_boxes x 2048 array of features
      'image_bb': num_images x num_boxes x 4 array of bounding boxes }

Reference: https://github.com/hengyuan-hu/bottom-up-attention-vqa/blob/master/tools/detection_features_converter.py
"""
import base64
import csv
import h5py
import numpy as np
import os
import pickle
import sys

# Set CSV Field Size Limit (Big TSV Files...)
csv.field_size_limit(sys.maxsize)

FIELDNAMES = [
    "img_id",
    "img_h",
    "img_w",
    "objects_id",
    "objects_conf",
    "attrs_id",
    "attrs_conf",
    "num_boxes",
    "boxes",
    "features",
]
NUM_FIXED_BOXES = 36
FEATURE_LENGTH = 2048


def vqa2_create_object_features(vqa2_f="data/VQA2-Features", cache="data/VQA2-Cache"):
    """Iterate through Object TSV and Build HDF5 Files with Bounding Box Features, Image ID --> IDX Mappings"""
    print("\t[*] Setting up HDF5 Files for Image/Object Features...")

    train_indices, val_indices = {}, {}
    tfile = os.path.join(cache, "train36.hdf5")
    vfile = os.path.join(cache, "val36.hdf5")

    tidxfile = os.path.join(cache, "train36_img2idx.pkl")
    vidxfile = os.path.join(cache, "val36_img2idx.pkl")

    # Shortcut --> Based on if Files Exist
    if os.path.exists(tfile) and os.path.exists(vfile) and os.path.exists(tidxfile) and os.path.exists(vidxfile):
        with open(tidxfile, "rb") as f:
            train_indices = pickle.load(f)

        with open(vidxfile, "rb") as f:
            val_indices = pickle.load(f)

        return train_indices, val_indices

    with h5py.File(tfile, "w") as h_train, h5py.File(vfile, "w") as h_val:
        # Get Number of Images in each Split
        with open(os.path.join(vqa2_f, "train2014_obj36.tsv"), "r") as f:
            ntrain = len(f.readlines())

        with open(os.path.join(vqa2_f, "val2014_obj36.tsv"), "r") as f:
            nval = len(f.readlines())

        # Setup HDF5 Files
        train_img_features = h_train.create_dataset("image_features", (ntrain, NUM_FIXED_BOXES, FEATURE_LENGTH), "f")
        train_img_bb = h_train.create_dataset("image_bb", (ntrain, NUM_FIXED_BOXES, 4), "f")
        train_spatial_features = h_train.create_dataset("spatial_features", (ntrain, NUM_FIXED_BOXES, 6), "f")

        val_img_features = h_val.create_dataset("image_features", (nval, NUM_FIXED_BOXES, FEATURE_LENGTH), "f")
        val_img_bb = h_val.create_dataset("image_bb", (nval, NUM_FIXED_BOXES, 4), "f")
        val_spatial_features = h_val.create_dataset("spatial_features", (nval, NUM_FIXED_BOXES, 6), "f")

        # Start Iterating through TSV
        print("\t[*] Reading Train TSV File and Populating HDF5 File...")
        train_counter, val_counter = 0, 0
        with open(os.path.join(vqa2_f, "train2014_obj36.tsv"), "r") as tsv:
            reader = csv.DictReader(tsv, delimiter="\t", fieldnames=FIELDNAMES)
            for item in reader:
                item["num_boxes"] = int(item["num_boxes"])
                image_id = item["img_id"]
                image_w = float(item["img_w"])
                image_h = float(item["img_h"])
                bb = np.frombuffer(base64.b64decode(item["boxes"]), dtype=np.float32).reshape((item["num_boxes"], -1))

                box_width = bb[:, 2] - bb[:, 0]
                box_height = bb[:, 3] - bb[:, 1]
                scaled_width = box_width / image_w
                scaled_height = box_height / image_h
                scaled_x = bb[:, 0] / image_w
                scaled_y = bb[:, 1] / image_h

                scaled_width = scaled_width[..., np.newaxis]
                scaled_height = scaled_height[..., np.newaxis]
                scaled_x = scaled_x[..., np.newaxis]
                scaled_y = scaled_y[..., np.newaxis]

                spatial_features = np.concatenate(
                    (scaled_x, scaled_y, scaled_x + scaled_width, scaled_y + scaled_height, scaled_width, scaled_height),
                    axis=1,
                )

                train_indices[image_id] = train_counter
                train_img_bb[train_counter, :, :] = bb
                train_img_features[train_counter, :, :] = np.frombuffer(
                    base64.b64decode(item["features"]), dtype=np.float32
                ).reshape((item["num_boxes"], -1))
                train_spatial_features[train_counter, :, :] = spatial_features
                train_counter += 1

        print("\t[*] Reading Val TSV File and Populating HDF5 File...")
        with open(os.path.join(vqa2_f, "val2014_obj36.tsv"), "r") as tsv:
            reader = csv.DictReader(tsv, delimiter="\t", fieldnames=FIELDNAMES)
            for item in reader:
                item["num_boxes"] = int(item["num_boxes"])
                image_id = item["img_id"]
                image_w = float(item["img_w"])
                image_h = float(item["img_h"])
                bb = np.frombuffer(base64.b64decode(item["boxes"]), dtype=np.float32).reshape((item["num_boxes"], -1))

                box_width = bb[:, 2] - bb[:, 0]
                box_height = bb[:, 3] - bb[:, 1]
                scaled_width = box_width / image_w
                scaled_height = box_height / image_h
                scaled_x = bb[:, 0] / image_w
                scaled_y = bb[:, 1] / image_h

                scaled_width = scaled_width[..., np.newaxis]
                scaled_height = scaled_height[..., np.newaxis]
                scaled_x = scaled_x[..., np.newaxis]
                scaled_y = scaled_y[..., np.newaxis]

                spatial_features = np.concatenate(
                    (scaled_x, scaled_y, scaled_x + scaled_width, scaled_y + scaled_height, scaled_width, scaled_height),
                    axis=1,
                )

                val_indices[image_id] = val_counter
                val_img_bb[val_counter, :, :] = bb
                val_img_features[val_counter, :, :] = np.frombuffer(
                    base64.b64decode(item["features"]), dtype=np.float32
                ).reshape((item["num_boxes"], -1))
                val_spatial_features[val_counter, :, :] = spatial_features
                val_counter += 1

    # Dump Train and Validation Indices to File
    with open(tidxfile, "wb") as f:
        pickle.dump(train_indices, f)

    with open(vidxfile, "wb") as f:
        pickle.dump(val_indices, f)

    return train_indices, val_indices
