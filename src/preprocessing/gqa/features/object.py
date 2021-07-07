"""
object.py

Reads in a tsv file with pre-trained bottom up attention features (object features) and writes them to an hdf5 file.
Additionally builds image ID --> Feature IDX Mapping.

Hierarchy of HDF5 file:
    { 'image_features': num_images x num_boxes x 2048 array of features
      'image_bb': num_images x num_boxes x 4 array of bounding boxes }

Reference: https://github.com/hengyuan-hu/bottom-up-attention-vqa/blob/master/tools/detection_features_converter.py
Reference: https://github.com/airsplay/lxmert/blob/master/data/vg_gqa_imgfeat/extract_gqa_image.py
"""
import base64
import csv
import os
import pickle
import sys

import h5py
import numpy as np

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


def gqa_create_object_features(gqa_f="data/GQA-Features", cache="data/GQA-Cache"):
    """Iterate through Object TSV and Build HDF5 Files with Bounding Box Features, Image ID --> IDX Mappings"""
    print("\t[*] Setting up HDF5 Files for Image/Object Features...")

    trainval_indices, testdev_indices = {}, {}
    v_file = os.path.join(cache, "trainval36.hdf5")
    d_file = os.path.join(cache, "testdev36.hdf5")

    v_idxfile = os.path.join(cache, "trainval36_img2idx.pkl")
    d_idxfile = os.path.join(cache, "testdev36_img2idx.pkl")

    # Shortcut --> Based on if Files Exist
    if os.path.exists(v_file) and os.path.exists(d_file) and os.path.exists(v_idxfile) and os.path.exists(d_idxfile):
        with open(v_idxfile, "rb") as f:
            trainval_indices = pickle.load(f)

        with open(d_idxfile, "rb") as f:
            testdev_indices = pickle.load(f)

        return trainval_indices, testdev_indices

    with h5py.File(v_file, "w") as h_trainval, h5py.File(d_file, "w") as h_testdev:
        # Get Number of Images in each Split
        with open(os.path.join(gqa_f, "vg_gqa_obj36.tsv"), "r") as f:
            ntrainval = len(f.readlines())

        with open(os.path.join(gqa_f, "gqa_testdev_obj36.tsv"), "r") as f:
            ntestdev = len(f.readlines())

        # Setup HDF5 Files
        trainval_img_features = h_trainval.create_dataset(
            "image_features", (ntrainval, NUM_FIXED_BOXES, FEATURE_LENGTH), "f"
        )
        trainval_img_bb = h_trainval.create_dataset("image_bb", (ntrainval, NUM_FIXED_BOXES, 4), "f")
        trainval_spatial_features = h_trainval.create_dataset("spatial_features", (ntrainval, NUM_FIXED_BOXES, 6), "f")

        testdev_img_features = h_testdev.create_dataset(
            "image_features", (ntestdev, NUM_FIXED_BOXES, FEATURE_LENGTH), "f"
        )
        testdev_img_bb = h_testdev.create_dataset("image_bb", (ntestdev, NUM_FIXED_BOXES, 4), "f")
        testdev_spatial_features = h_testdev.create_dataset("spatial_features", (ntestdev, NUM_FIXED_BOXES, 6), "f")

        # Start Iterating through TSV
        print("\t[*] Reading Train-Val TSV File and Populating HDF5 File...")
        trainval_counter, testdev_counter = 0, 0
        with open(os.path.join(gqa_f, "vg_gqa_obj36.tsv"), "r") as tsv:
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

                trainval_indices[image_id] = trainval_counter
                trainval_img_bb[trainval_counter, :, :] = bb
                trainval_img_features[trainval_counter, :, :] = np.frombuffer(
                    base64.b64decode(item["features"]), dtype=np.float32
                ).reshape((item["num_boxes"], -1))
                trainval_spatial_features[trainval_counter, :, :] = spatial_features
                trainval_counter += 1

        print("\t[*] Reading Test-Dev TSV File and Populating HDF5 File...")
        with open(os.path.join(gqa_f, "gqa_testdev_obj36.tsv"), "r") as tsv:
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

                testdev_indices[image_id] = testdev_counter
                testdev_img_bb[testdev_counter, :, :] = bb
                testdev_img_features[testdev_counter, :, :] = np.frombuffer(
                    base64.b64decode(item["features"]), dtype=np.float32
                ).reshape((item["num_boxes"], -1))
                testdev_spatial_features[testdev_counter, :, :] = spatial_features
                testdev_counter += 1

    # Dump TrainVal and TestDev Indices to File
    with open(v_idxfile, "wb") as f:
        pickle.dump(trainval_indices, f)

    with open(d_idxfile, "wb") as f:
        pickle.dump(testdev_indices, f)

    return trainval_indices, testdev_indices
