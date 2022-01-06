"""
extract.py

Extracts ResNet-101 Spatial Features from the given Dataset, and stores them in the appropriate directory.
"""
from PIL import Image
from tap import Tap
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from tqdm import tqdm

import h5py
import os
import pickle
import torch
import torch.nn as nn

FEATURE_LENGTH = 2048
GRID = 7


class ArgumentParser(Tap):
    # fmt: off
    dataset: str = "vqa2"                   # Dataset for Extracting Image Features in < vqa2 | gqa >
    images: str = "data/VQA-Images"         # Path to Raw Images in < data/VQA-Images | data/GQA-Images >
    spatial: str = "data/VQA-Spatials"      # Path for Output HDF5 in < data/VQA-Spatials | data/GQA-Spatials >
    # fmt: on

class OOMDataset(Dataset):
    def __init__(self, path, transform):
        self.path, self.transform = path, transform
        self.images = [x for x in os.listdir(os.path.join(path)) if ".jpg" in x]

    def __getitem__(self, index):
        i_path = os.path.join(self.path, self.images[index])
        with open(i_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")

        return i_path, self.transform(img)

    def __len__(self):
        return len(self.images)


def extract():
    args = ArgumentParser().parse_args()

    # Book-Keeping
    if not os.path.exists(args.spatial):
        os.makedirs(args.spatial)

    # CUDA Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Normalizing/Cropping Transform used for all ResNet Models
    #   Ref: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Load Thyself a ResNet-101
    resnet101 = models.resnet101(pretrained=True)

    # Create Spatial and Fully Connected Forks
    modules = list(resnet101.children())
    spatial, pooled = nn.Sequential(*modules[:-2]), nn.Sequential(*modules[:-1])
    for p in resnet101.parameters():
        p.requires_grad = False

    # Device Management
    spatial.to(device)
    pooled.to(device)

    # Dataset-Specific Loading
    if args.dataset == "vqa2":
        # Create Torchvision Datasets
        print("[*] Loading and Processing Train and Validation Data...")
        image_dataset = datasets.ImageFolder(args.images, transform)

        # Get List of Images
        images, labels = zip(*image_dataset.imgs)
        images = [os.path.basename(x) for x in images]
        mapping = image_dataset.classes

        # Create an HDF5 File for Training and Validation
        tfile, vfile = os.path.join(args.spatial, "train101.hdf5"), os.path.join(args.spatial, "val101.hdf5")
        tid, vid = os.path.join(args.spatial, "train101_img2idx.pkl"), os.path.join(args.spatial, "val101_img2idx.pkl")
        train_idx, val_idx = {}, {}

        with h5py.File(tfile, "w") as h_train, h5py.File(vfile, "w") as h_val:
            # Get Number of Images in Each Split
            n_train, n_val = len(labels) - sum(labels), sum(labels)

            # Setup HDF5 Files
            train_spatial = h_train.create_dataset("spatial", (n_train, FEATURE_LENGTH, GRID, GRID))
            train_pool = h_train.create_dataset("pool", (n_train, FEATURE_LENGTH))

            val_spatial = h_val.create_dataset("spatial", (n_train, FEATURE_LENGTH, GRID, GRID))
            val_pool = h_val.create_dataset("pool", (n_train, FEATURE_LENGTH))

            train_counter, val_counter = 0, 0
            for i in tqdm(range(len(image_dataset))):
                # Retrieve Image, Label
                img, label = image_dataset[i]

                # Forward Pass through ResNet
                img = img.unsqueeze(0).to(device)
                s, p = spatial(img).squeeze().cpu().numpy(), pooled(img).squeeze().cpu().numpy()

                if mapping[label] == "train2014":
                    train_idx[images[i]] = train_counter
                    train_spatial[train_counter] = s
                    train_pool[train_counter] = p
                    train_counter += 1

                elif mapping[label] == "val2014":
                    val_idx[images[i]] = val_counter
                    val_spatial[val_counter] = s
                    val_pool[val_counter] = p
                    val_counter += 1

        # Dump Train and Validation Indices to File
        with open(tid, "wb") as f:
            pickle.dump(train_idx, f)

        with open(vid, "wb") as f:
            pickle.dump(val_idx, f)

    elif args.dataset == "gqa":
        # Create Torchvision Datasets
        print("[*] Loading and Processing ALL Images in Dataset...")
        image_dataset = OOMDataset(args.images, transform)

        # Create one Grande-Sized HDF5 File for Training and Validation
        gfile = os.path.join(args.spatial, "gqa101.hdf5")
        gid = os.path.join(args.spatial, "gqa101_img2idx.pkl")
        img2idx = {}

        with h5py.File(gfile, "w") as h_gqa:
            # Get Total Number of Images in GQA
            n = len(image_dataset)

            # Setup HDF5 File
            gqa_spatial = h_gqa.create_dataset("spatial", (n, FEATURE_LENGTH, GRID, GRID))
            gqa_pool = h_gqa.create_dataset("pool", (n, FEATURE_LENGTH))

            counter = 0
            for i in tqdm(range(len(image_dataset))):
                # Retrieve Image, Label
                id_path, img = image_dataset[i]
                img_id = os.path.basename(id_path).split(".")[0]

                # Forward Pass through ResNet
                img = img.unsqueeze(0).to(device)
                s, p = spatial(img).squeeze().cpu().numpy(), pooled(img).squeeze().cpu().numpy()
                img2idx[img_id] = counter
                gqa_spatial[counter] = s
                gqa_pool[counter] = p
                counter += 1

        # Dump Indices to File
        with open(gid, "wb") as f:
            pickle.dump(img2idx, f)


if __name__ == "__main__":
    extract()
