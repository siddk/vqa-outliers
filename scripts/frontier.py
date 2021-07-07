"""
frontier.py

Using the dataset maps created during standard cartography analysis (`cartograph.py`) assemble a ranked list of
examples by *correctness*. Use these examples to perform outlier adjusted training (start with 50% as Burn-In).
"""
from tqdm import tqdm

import json
import numpy as np
import os
import pickle
import random
from tap import Tap


class ArgumentParser(Tap):
    # fmt: off
    model: str = "butd"         # Model to use to Generate "Outlier" Frontier
    # fmt: on


def frontier():
    """Create Sorted List of Indices to Establish 'Outlier-Frontier' for Outlier Analysis Jobs"""
    print("[*] Establishing Frontier...")
    args = ArgumentParser().parse_args()
    model = args.model

    # Set Random Seed
    random.seed(21)

    # Open Base Map File
    mfile = [x for x in os.listdir("checkpoints/map/metrics") if "vqa2-all-%s-map" % model in x]
    assert len(mfile) == 1, "Why do we have more than one valid map?"
    with open(os.path.join("checkpoints/map/metrics", mfile[0]), "r") as f:
        cartography = json.load(f)["cartography"]

    # Compute Necessary Charting Information
    index = sorted(map(int, list(cartography[0].keys())))
    correctness = np.zeros(len(index))

    # Compute Mean Correctness so we can rank!
    print("\n[*] Computing Correctness...")
    for i, entry_id in tqdm(enumerate(index), total=len(index)):
        for ep in cartography:
            correct_conf, max_conf = ep[str(entry_id)]
            correctness[i] += 1 if correct_conf == max_conf else 0

    # Normalize by Epochs
    correctness /= len(cartography)

    # Sort List by Confidence
    idx_correctness = [(i, v) for i, v in enumerate(correctness)]
    idx_correctness = sorted(idx_correctness, key=lambda x: x[1], reverse=True)

    # Keep an even 400K of them!
    drop_indices = set(random.sample(range(len(idx_correctness)), len(idx_correctness) - 400000))
    idx_correctness = [idx_correctness[i] for i in range(len(idx_correctness)) if i not in drop_indices]

    # Establish Splits of 50, 60, 75, and 90%
    total_len = len(idx_correctness)
    assert total_len == 400000, "Oops?"
    buckets = {
        "f50": [index[x[0]] for x in idx_correctness[: int(total_len * 0.5)]],
        "f60": [index[x[0]] for x in idx_correctness[: int(total_len * 0.6)]],
        "f75": [index[x[0]] for x in idx_correctness[: int(total_len * 0.75)]],
        "f90": [index[x[0]] for x in idx_correctness[: int(total_len * 0.9)]],
    }

    # Sanity Check
    for k in ["f50", "f60", "f75", "f90"]:
        print("Bucket Key %s: %d Examples" % (k, len(buckets[k])))

    # Create Frontier File!
    ffile = "vqa2-all-%s-frontier.pkl" % model
    with open(os.path.join("data/Maps/", ffile), "wb") as f:
        pickle.dump(buckets, f)


if __name__ == "__main__":
    frontier()
