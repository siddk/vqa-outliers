"""
ptolemy.py

Generate & Plot Dataset Maps from the output of `cartograph.py` -- additionally, after active learning has been run,
visualize acquisitions relative to maps (plots of Easy, Medium, Hard, and Impossible examples).
"""
from math import floor, log10
from tap import Tap
from tqdm import tqdm
from typing import List

import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pickle

# Define Average Confidence Thresholds for Bucketing
BUCKETS = {"easy": 0.75, "medium": 0.50, "hard": 0.25, "impossible": 0.0}
DECISION_THRESHOLD = 0.5

# Matplotlib Colors + Fonts
matplotlib.rcParams["font.sans-serif"] = "Raleway"
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.size"] = 18
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["xtick.bottom"] = True
matplotlib.rcParams["ytick.left"] = True
matplotlib.rcParams["legend.loc"] = "lower right"

# Colors
cmap = sns.color_palette("coolwarm_r", as_cmap=True)

# Readable Names
datasets = {"vqa2-sports": "VQA-Sports", "vqa2-food": "VQA-Food", "vqa2-all": "VQA-2", "gqa-all": "GQA"}

models = {
    "glreg": "LogReg (ResNet-101)",
    "olreg": "LogReg (Faster R-CNN)",
    "cnn": "LSTM-CNN",
    "butd": "BUTD",
    "lxmert": "LXMERT",
}

strat2string = {
    "baseline": "Random Baseline",
    "least-conf": "Least-Confidence",
    "entropy": "Entropy",
    "mc-entropy": "MC-Dropout Entropy",
    "mc-bald": "BALD",
    "coreset-fused": "Core-Set (Fused)",
    "coreset-language": "Core-Set (Language)",
    "coreset-vision": "Core-Set (Vision)",
}

# Dataset Statistics -- Specify Different Datasets and Active Learning Splits...
n_examples = {
    "vqa2": {
        # Total of 443,757 (400K) Training Examples --> 214,354 Validation Examples
        "all": {
            # Burn-In + Iterations
            "p05": [20000, 60000, 100000, 140000, 180000, 220000, 260000, 300000, 340000, 380000, 400000],
            "p10": [40000, 80000, 120000, 160000, 200000, 240000, 280000, 320000, 360000, 400000],
            "p25": [100000, 140000, 180000, 220000, 260000, 300000, 340000, 380000, 400000],
            "p50": [200000, 240000, 280000, 320000, 360000, 400000],
        },
        # Total of 5411 (5000) Training Examples --> 2481 Validation Examples
        "sports": {
            # Burn-In + Iterations
            "p05": [250, 750, 1250, 1750, 2250, 2750, 3250, 3750, 4250, 4750, 5000],
            "p10": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000],
            "p25": [1250, 1750, 2250, 2750, 3250, 3750, 4250, 4750, 5000],
            "p50": [2500, 3000, 3500, 4000, 4500, 5000],
        },
        # Total of 4082 Train Examples --> 2049 Validation Examples
        "food": {
            # Burn-In + Iterations
            "p05": [200, 600, 1000, 1400, 1800, 2200, 2600, 3000, 3400, 3800, 4000],
            "p10": [400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000],
            "p25": [1000, 1400, 1800, 2200, 2600, 3000, 3400, 3800, 4000],
            "p50": [2000, 2400, 2800, 3200, 3600, 4000],
        },
    },
    "gqa": {
        # Total of 943,000 (900K) Questions --> 1944 TestDev Questions
        "all": {
            # Burn-In + Iterations
            "p05": [45000, 135000, 225000, 315000, 405000, 495000, 585000, 675000, 765000, 855000, 900000],
            "p10": [90000, 180000, 270000, 360000, 450000, 540000, 630000, 720000, 810000, 900000],
            "p25": [225000, 315000, 405000, 495000, 585000, 675000, 765000, 855000, 900000],
            "p50": [450000, 540000, 630000, 720000, 810000, 900000],
        }
    },
}

max_lengths = {"vqa2": {"food": 4000, "sports": 5000, "moderated": 250000, "all": 400000}, "gqa": {"all": 900000}}


class ArgumentParser(Tap):
    # fmt: off
    # Checkpoint Parameters (Map)
    maps: str = 'checkpoints/map/metrics'               # Path to Dataset Cartography (Mapping) Outputs (Metrics Files)
    active: str = 'checkpoints/active/active-indices'   # Path to Active Indices
    out: str = 'data/Maps'                              # Local Path (on Cluster --> write to data/Maps!)

    # Mode
    mode: str = "map"                                   # Mode to run with: "map" (Map only) or "acquisitions"

    # Dataset and Split Parameters
    dataset: str = 'vqa2'                               # Core VQA Dataset to run with :: < vqa2 | gqa >
    split: str = 'all'                                  # Split for Map Generation :: < sports | food | all >

    # Model Parameters
    model: str = 'butd'                                 # Generate Map with respect to this Model

    # Burn to Plot
    burn: str = 'p10'

    # Strategies to Plot
    strategies: List[str] = [
        "baseline",
        "least-conf",
        "entropy",
        "mc-entropy",
        "mc-bald",
        "coreset-fused",
        "coreset-language",
        "coreset-vision",
    ]

    # Random Seed
    seed: int = 7                                       # Seed for Map Generation (Should always be 7)

    # fmt: on


def mapmake(cart_conf, dataset, split, model, out="data/Maps"):
    """Core Dataset Cartography Logic -- take in logged confidences/variances during training, build Map!"""

    # Create Output Directory if it doesn't exist
    if not os.path.exists(out):
        os.makedirs(out)

    # Save ourselves some time and load from file if it exists
    cfile = os.path.join(out, "%s-%s-%s-map.pkl" % (dataset, split, model))

    # Short-Circuit and Return Early if Possible
    if os.path.exists(cfile):
        with open(cfile, "rb") as f:
            confidence, correctness, variability, buckets = pickle.load(f)

        return confidence, correctness, variability, buckets

    # Compute the necessary charting information
    index = sorted(map(int, list(cart_conf[0].keys())))
    confidence, correctness, variability = np.zeros(len(index)), np.zeros(len(index)), np.zeros(len(index))

    # Compute Mean Confidence, Correctness
    print("\n[*] Computing Confidence and Correctness...")
    for i, entry_id in tqdm(enumerate(index), total=len(index)):
        for ep in cart_conf:
            correct_conf, max_conf = ep[str(entry_id)]
            confidence[i] += correct_conf
            correctness[i] += 1 if correct_conf == max_conf else 0

    # Normalize by Epochs
    confidence /= len(cart_conf)
    correctness /= len(cart_conf)

    # Compute Variability
    print("\n[*] Computing Variability...")
    for i, entry_id in tqdm(enumerate(index), total=len(index)):
        for ep in cart_conf:
            variability[i] += (ep[str(entry_id)][0] - confidence[i]) ** 2

    # Normalize by Epochs then Sqrt
    variability /= len(cart_conf)
    variability = np.sqrt(variability)

    # Create Buckets
    buckets = {bucket: [] for bucket in BUCKETS}
    print("\n[*] Bucketing...")
    for idx, c in tqdm(enumerate(correctness), total=len(correctness)):
        for b in ["easy", "medium", "hard", "impossible"]:
            if c >= BUCKETS[b]:
                buckets[b].append(index[idx])
                break

    # Pickle
    with open(cfile, "wb") as f:
        pickle.dump((confidence, correctness, variability, buckets), f)

    return confidence, correctness, variability, buckets


def human_format(num):
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format("{:f}".format(num).rstrip("0").rstrip("."), ["", "K", "M", "B", "T"][magnitude])


def chart():
    # Parse Args
    print("[*] Starting Expedition...")
    args = ArgumentParser().parse_args()
    print('\t> "Now, we enter uncharted territory..." (Anonymous)')

    # Mode Check
    assert args.mode in ["map", "acquisitions"], "%s is not a valid mode!" % args.mode

    # Open Up Map File and Generate Dataset Map via Dataset Cartography (Swayamdipta et. al. 2020)
    print("\n[*] Opening Cartography File...")
    mfile = [x for x in os.listdir(args.maps) if "%s-%s-%s-map" % (args.dataset, args.split, args.model) in x]
    assert len(mfile) == 1, "Why do we have more than one valid map?"
    with open(os.path.join(args.maps, mfile[0]), "r") as f:
        data = json.load(f)
    confidence, correctness, variability, buckets = mapmake(
        data["cartography"], args.dataset, args.split, args.model, out=args.out
    )

    # Start Plotting
    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 8))
    p = ax1.scatter(variability, confidence, c=correctness, cmap=cmap, alpha=0.77)

    # Labeling
    ax1.set_xlabel("Variability", fontname="Raleway", fontsize=21)
    ax1.set_ylabel("Confidence", fontname="Raleway", fontsize=21)

    # Axis Handling
    ax1.set_xlim([0, 0.5])
    ax1.set_ylim([0, 1])

    # Styling
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.xaxis.set_ticks_position("bottom")
    ax1.xaxis.set_ticks(np.arange(0, 0.51, 0.05))
    ax1.yaxis.set_ticks_position("left")
    ax1.yaxis.set_ticks(np.arange(0, 1.01, 0.1))

    # Colorbar
    cbar = fig.colorbar(p, ax=ax1, fraction=0.043, pad=0.00, shrink=0.9, aspect=50, orientation="vertical")
    cbar.solids.set(alpha=1)
    cbar.ax.set_title("Correctness", fontsize=21, pad=7)

    # Book-Keeping
    path = os.path.join("visualizations", "atlas", "%s-%s" % (args.dataset, args.split))
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path, "%s-%s-%s.pdf" % (args.model, args.dataset, args.split)), bbox_inches="tight")

    # Only Visualize Acquisitions in Given Mode
    if args.mode in ["acquisitions"]:
        for strat in args.strategies:
            plt.figure(figsize=(8, 6))

            # No Monte-Carlo for Logistic Regression
            if args.model in ["glreg", "olreg"] and "mc-" in strat:
                continue

            # Retrieve Active Indices in Order
            afile = [
                x
                for x in os.listdir(args.active)
                if args.model in x
                and args.dataset in x
                and args.split in x
                and ("x%d" % args.seed) in x
                and strat in x
                and ("nex-%d" % max_lengths[args.dataset][args.split]) in x
                and args.burn in x
            ]

            # Error-Handling
            if len(afile) == 0:
                continue
            else:
                afile = afile[0]

            # Grab Indices
            with open(os.path.join(args.active, afile, "active-indices.json"), "r") as f:
                indices = json.load(f)

            # Make Set of Buckets
            checkable_bucket = {b: set(buckets[b]) for b in buckets}

            # Plot Moving Time-Series of Various Difficulties
            easy, medium = [0 for _ in range(len(indices))], [0 for _ in range(len(indices))]
            hard, impossible = [0 for _ in range(len(indices))], [0 for _ in range(len(indices))]
            for i, nex in enumerate(n_examples[args.dataset][args.split][args.burn]):
                # Skip Iteration 1
                if i == 0:
                    continue

                # Only Visualize Newly Chosen Examples
                prev_active_indices = indices[str(i - 1)]
                curr_active_indices = indices[str(i)]
                active_indices = set(curr_active_indices) - set(prev_active_indices)

                # Active Colors
                for aidx in list(active_indices):
                    for b in ["easy", "medium", "hard", "impossible"]:
                        if aidx in checkable_bucket[b]:
                            if b == "easy":
                                easy[i] += 1
                            elif b == "medium":
                                medium[i] += 1
                            elif b == "hard":
                                hard[i] += 1
                            elif b == "impossible":
                                impossible[i] += 1

            # Book-Keeping
            path = os.path.join("visualizations", "atlas", "%s-%s" % (args.dataset, args.split), "acquisitions")
            if not os.path.exists(path):
                os.makedirs(path)

            # Create X-Axis
            x_axis = n_examples[args.dataset][args.split][args.burn][:-1]
            easy, medium, hard, impossible = (
                np.array(easy[1:]),
                np.array(medium[1:]),
                np.array(hard[1:]),
                np.array(impossible[1:]),
            )

            # Bar Graph (Easy on the Bottom)
            plt.bar(list(range(len(x_axis))), easy, color=cmap(1.0), label="Easy [p > 0.75]", alpha=0.7)
            plt.bar(list(range(len(x_axis))), medium, bottom=easy, color=cmap(0.66), label="Medium [p > 0.5]", alpha=0.7)
            plt.bar(
                list(range(len(x_axis))),
                hard,
                bottom=easy + medium,
                color=cmap(0.33),
                label="Hard [p > 0.25]",
                alpha=0.7,
            )
            plt.bar(
                list(range(len(x_axis))),
                impossible,
                bottom=easy + medium + hard,
                color=cmap(0.0),
                label="Impossible [p > 0.0]",
                alpha=0.7,
            )

            def format_func(value, tick_number=None):
                num_thousands = 0 if abs(value) < 1000 else floor(log10(abs(value)) / 3)
                value = round(value / 1000 ** num_thousands, 2)
                return f"{value:g}" + " KMGTPEZY"[num_thousands]

            plt.title("%s" % (strat2string[strat]), fontsize=25)
            plt.xlabel("Number of Training Examples", fontsize=25)
            plt.ylabel("Acquisitions by Difficulty", fontsize=25)

            # Get Axis & Format Y
            ax = plt.axes()
            ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
            plt.xticks(list(range(len(x_axis))), list(map(human_format, x_axis)), fontsize=21, rotation=45)

            # Show Legend & Save
            if strat == "baseline":
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles[::-1], labels[::-1], framealpha=0.95)
            plt.savefig(
                os.path.join(path, "%s-%s-%s-acquired.pdf" % (args.model, strat, args.burn)), bbox_inches="tight"
            )
            plt.clf()


if __name__ == "__main__":
    chart()
