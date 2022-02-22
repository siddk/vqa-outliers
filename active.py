"""
active.py

Run active-learning experiments across various VQA datasets -- packages code for running random baselines (random
subsets of VQA data), as well as standard active learning techniques (least confidence, entropy-based uncertainty
sampling, monte-carlo methods, and coresets).

Additionally saves model checkpoints and logs training statistics.
"""
from argparse import Namespace
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from tap import Tap
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.logging import MetricLogger
from src.models import BUTD, GridLogisticRegression, ObjectLogisticRegression, LSTMCNN, LXMERT
from src.preprocessing.gqa import (
    gqa_create_dictionary_glove,
    gqa_create_answers,
    gqa_create_object_features,
    gqa_create_grid_features,
    GQAObjectDataset,
    GQAGridDataset,
    GQAObjectIndexDataset,
    GQAGridIndexDataset,
)
from src.preprocessing.vqa2 import (
    vqa2_create_dictionary_glove,
    vqa2_create_answers,
    vqa2_create_grid_features,
    vqa2_create_object_features,
    VQAGridDataset,
    VQAObjectDataset,
    VQAGridIndexDataset,
    VQAObjectIndexDataset,
)

import json
import numpy as np
import os
import random
import pickle
import pytorch_lightning as pl
import time
import torch


# Specify Different Datasets and Active Learning Splits...
N_EXAMPLES = {
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
    # Outliers --> Start with only Easy Examples, then selectively add "outliers" (hard examples from Dataset Maps)
    "vqa2-frontier": {
        # Stage 0 (First 50% of Dataset by Confidence) --> 200K Examples Total
        "f50": {"p10": [20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000]},
        # Stage 1 (First 60% of Dataset by Confidence) --> 240K Examples Total
        "f60": {"p10": [24000, 48000, 72000, 96000, 120000, 144000, 168000, 192000, 216000, 240000]},
        # Stage 2 (First 75% of Dataset by Confidence) --> 300K Examples Total
        "f75": {"p10": [30000, 60000, 90000, 120000, 150000, 180000, 210000, 240000, 270000, 300000]},
        # Stage 3 (First 90% of Dataset by Confidence) --> 360K Examples Total
        "f90": {"p10": [36000, 72000, 108000, 144000, 180000, 216000, 252000, 288000, 324000, 360000]},
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

CATEGORY2IDX = {"easy": 0, "medium": 1, "hard": 2, "impossible": 3}


class ArgumentParser(Tap):
    # fmt: off

    # Data and Checkpoint Parameters
    data_dir: str = "data/"                         # Path to downloaded data
    save_dir: str = "checkpoints/active"            # Path to checkpoints, serialized statistics, and WandB artifacts

    # GQA Specific Parameters
    gqa_questions: str = 'data/GQA-Questions'       # Path to GQA Questions
    gqa_features: str = 'data/GQA-Features'         # Path to GQA Features
    gqa_grid: str = 'data/GQA-Spatials'             # Path to GQA Spatial/Grid Features
    gqa_cache: str = "data/GQA-Cache"               # Path to GQA Cache Directory for storing serialized data

    # VQA-2 Specific Parameters
    vqa2_questions: str = 'data/VQA2-Questions'     # Path to VQA-2 Questions
    vqa2_features: str = 'data/VQA2-Features'       # Path to VQA-2 Features
    vqa2_grid: str = 'data/VQA2-Spatials'           # Path to VQA-2 Spatial/Grid Features
    vqa2_cache: str = 'data/VQA2-Cache'             # Path to VQA-2 Cache Directory for storing serialized data

    # GloVe Vectors
    glove: str = 'data/GloVe/glove.6B.300d.txt'     # Path to GloVe Embeddings File (300-dim)

    # LXMERT (HF Transformers Cache)
    lxmert_cache: str = 'data/LXMERT'               # Path to LXMERT Checkpoint & Cache Directory

    # Map Directory
    maps: str = 'data/Maps'                         # Path to Pre-Computed Dataset Maps (for running Oracles)

    # Run/WandB Parameters
    sync: bool = False                              # Whether or not to store run details on WandB
    run_name: str = None                            # Informative Run-ID for WandB

    # GPUs
    gpus: int = 0                                   # Number of GPUs to run with

    # Modes
    dataset: str = 'vqa2'                           # Dataset to run model with -- < vqa2 | gqa | vqa2-frontier >
    split: str = 'all'                              # Dataset Split to Run with in (consult dict above for options)
    mode: str = 'butd'                              # Mode to run - < glreg | olreg | cnn | butd | obj-film | lxmert >
    burn: str = 'p10'                               # Burn-in Examples (+ Possible Acquisition Batch Schedule)
    strategy: str = 'baseline'                      # Capability Selection Mode in -
                                                    #   < baseline
                                                    #     least-conf | entropy                  (Uncertainty Sampling)
                                                    #     mc-entropy | mc-bald                  (Deep Active Learning)
                                                    #     coreset-< fused | language | vision > (Diversity-Based)

    # MC-Dropout Parameters
    k_dropout: int = 10                             # Number of MC Dropout Runs per Example

    # CoreSets Parameters
    pca_components: int = 32                        # Down-Sample to this dimension (via PCA) prior to running CoreSets
    amortized_iterations: int = 20                  # Number of times to recompute cluster distances in an iteration

    # Model Parameters
    emb_dim: int = 300                              # Word Embedding Dimension --> Should Match GloVe (300)
    emb_dropout: float = 0.0                        # Dropout to Apply to Word Embeddings

    rnn: str = 'GRU'                                # RNN Type for Question Encoder --> one of < 'GRU' | 'LSTM' >
    rnn_layers: int = 1                             # Number of RNN Stacked Layers (for Question Encoder)
    bidirectional: bool = False                     # Whether or not RNN is Bidirectional
    q_dropout: float = 0.0                          # RNN Dropout (for Question Encoder)

    fusion: str = 'product'                         # Fusion Mechanism for Attention in < 'product' | 'concat' >
    attention_dropout: float = 0.2                  # Dropout for Attention Operation (fusing Image + Question)

    answer_dropout: float = 0.5                     # Dropout to Apply to Answer Classifier

    hidden: int = 1024                              # Dimensionality of Hidden Layer (Question Encoder & Object Encoder)

    weight_norm: bool = True                        # Boolean whether or not to use Weight Normalization
    weight_decay: float = 0.0                       # L2 Weight Decay Penalty for regularization

    # Training Parameters
    bsz: int = 512                                  # Batch Size --> the Bigger the Better
    epochs: int = 15                                # Number of Training Epochs

    opt: str = 'adamax'                             # Optimizer for Performing Gradient Updates
    gradient_clip: float = 0.25                     # Value for Gradient Clipping

    # Random Seed
    seed: int = 7                                   # Random Seed (for Reproducibility)

    # fmt: on


def active():
    # Parse Arguments --> Convert from Namespace --> Dict --> Namespace because of weird WandB Bug
    print("[*] Starting up...")
    args = Namespace(**ArgumentParser().parse_args().as_dict())
    print("\t> \"To go wrong in one's own way is better than to go right in someone else's\" (Dostoyevsky)")

    # Set Randomness
    print("\n[*] Setting Random Seed to %d!" % args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Tokenizers Parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Dataset-Specific Pre-Processing
    if args.dataset == "gqa":
        # Preprocess Question Data --> Return Dictionary and GloVe-initialized Embeddings
        print("\n[*] Pre-processing GQA Questions...")
        dictionary, emb = gqa_create_dictionary_glove(gqa_q=args.gqa_questions, glove=args.glove, cache=args.gqa_cache)

        # Preprocess Answer Data
        print("\n[*] Pre-processing GQA Answers...")
        ans2label, label2ans = gqa_create_answers(gqa_q=args.gqa_questions, cache=args.gqa_cache)

        # Create Image Features
        if args.mode in ["glreg", "cnn"]:
            print("\n[*] Pre-processing GQA Object Image Features")
            trainval_img2idx, testdev_img2idx = gqa_create_grid_features(gqa_g=args.gqa_grid, cache=args.gqa_cache)

            print("\n[*] Initializing Full (Pool) Training Dataset...")
            pool_dataset = GQAGridDataset(
                dictionary,
                ans2label,
                label2ans,
                trainval_img2idx,
                gqa_q=args.gqa_questions,
                gqa_g=args.gqa_grid,
                split=args.split,
                mode="train",
            )

            # Compute Set of "available indices"
            print("\n[*] Computing Set of Available Indices in Dataset...")
            available_indices = pool_dataset.indices
            idx2available = {x: i for i, x in enumerate(available_indices)}

            # Create Validation Dataset
            print("\n[*] Initializing Validation Dataset...")
            val_dataset = GQAGridDataset(
                dictionary,
                ans2label,
                label2ans,
                testdev_img2idx,
                gqa_q=args.gqa_questions,
                gqa_g=args.gqa_grid,
                split=args.split,
                mode="testdev",
            )

            # Build Initial Train Dataset
            print(
                "\n[*] Iteration 0 [%d Examples] :: Active Learning Dataset..."
                % N_EXAMPLES[args.dataset][args.split][args.burn][0]
            )
            active_indices = list(
                np.random.choice(
                    available_indices, size=N_EXAMPLES[args.dataset][args.split][args.burn][0], replace=False
                )
            )
            active_dataset = GQAGridIndexDataset(
                dictionary,
                ans2label,
                label2ans,
                trainval_img2idx,
                indices=list(active_indices),
                gqa_q=args.gqa_questions,
                gqa_g=args.gqa_grid,
                split=args.split,
                mode="train",
            )

            # Cheat Memory (Not really cuz it's just a file handle) --> Use Features from Pool Dataset
            active_dataset.features = pool_dataset.features

        elif args.mode in ["olreg", "butd", "lxmert"]:
            print("\n[*] Pre-processing GQA Object Image Features...")
            trainval_img2idx, testdev_img2idx = gqa_create_object_features(gqa_f=args.gqa_features, cache=args.gqa_cache)

            print("\n[*] Initializing Full (Pool) Training Dataset...")
            pool_dataset = GQAObjectDataset(
                dictionary,
                ans2label,
                label2ans,
                trainval_img2idx,
                gqa_q=args.gqa_questions,
                cache=args.gqa_cache,
                split=args.split,
                mode="train",
                lxmert=args.mode == "lxmert",
                lxmert_cache=args.lxmert_cache,
            )

            # Compute Set of "available indices"
            print("\n[*] Computing Set of Available Indices in Dataset...")
            available_indices = pool_dataset.indices
            idx2available = {x: i for i, x in enumerate(available_indices)}

            # Create Validation Dataset
            print("\n[*] Initializing Validation Dataset...")
            val_dataset = GQAObjectDataset(
                dictionary,
                ans2label,
                label2ans,
                testdev_img2idx,
                gqa_q=args.gqa_questions,
                cache=args.gqa_cache,
                split=args.split,
                mode="testdev",
                lxmert=args.mode == "lxmert",
                lxmert_cache=args.lxmert_cache,
            )

            # Build Initial Train Dataset
            print(
                "\n[*] Iteration 0 [%d Examples] :: Active Learning Dataset..."
                % N_EXAMPLES[args.dataset][args.split][args.burn][0]
            )
            active_indices = list(
                np.random.choice(
                    available_indices, size=N_EXAMPLES[args.dataset][args.split][args.burn][0], replace=False
                )
            )
            active_dataset = GQAObjectIndexDataset(
                dictionary,
                ans2label,
                label2ans,
                trainval_img2idx,
                indices=list(active_indices),
                gqa_q=args.gqa_questions,
                cache=args.gqa_cache,
                split=args.split,
                mode="train",
                lxmert=args.mode == "lxmert",
                lxmert_cache=args.lxmert_cache,
            )

            # Cheat Memory (Not Really cuz it's just a file handle) --> Use Features/Spatials from Pool Dataset
            active_dataset.features = pool_dataset.features
            active_dataset.spatials = pool_dataset.spatials

        else:
            raise NotImplementedError("No preprocessing pipeline for Mode '%s'" % args.mode)

    elif args.dataset in ["vqa2", "vqa2-frontier"]:
        # Preprocess Question Data --> Return Dictionary and GloVe-initialized Embeddings
        print("\n[*] Pre-processing VQA-2 Questions...")
        dictionary, emb = vqa2_create_dictionary_glove(
            vqa2_q=args.vqa2_questions, glove=args.glove, cache=args.vqa2_cache
        )

        # Preprocess Answer Data
        print("\n[*] Pre-processing VQA-2 Answers...")
        ans2label, label2ans = vqa2_create_answers(split=args.split, vqa2_q=args.vqa2_questions, cache=args.vqa2_cache)

        # If 'frontier' ("outliers") split, get valid indices!
        frontier_indices = None
        if args.dataset == "vqa2-frontier":
            # Fetch Map
            m_file = os.path.join(args.maps, "vqa2-all-%s-frontier.pkl" % args.mode)
            assert os.path.exists(m_file), "Map File %s Does Not Exist!" % m_file
            with open(m_file, "rb") as f:
                buckets = pickle.load(f)

            # Create Valid Indices (just 'easy' and 'medium')
            frontier_indices = buckets[args.split]

        # Create Image Features
        if args.mode in ["glreg", "cnn"]:
            print("\n[*] Pre-processing VQA-2 Grid Image Features...")
            train_img2idx, val_img2idx = vqa2_create_grid_features(vqa2_g=args.vqa2_grid)

            print("\n[*] Initializing Full Training Dataset...")
            pool_dataset = VQAGridDataset(
                dictionary,
                ans2label,
                label2ans,
                train_img2idx,
                frontier_indices,
                vqa2_q=args.vqa2_questions,
                vqa2_g=args.vqa2_grid,
                split=args.split,
                mode="train",
                mtype=args.mode,
            )

            # Compute Set of "available indices"
            print("\n[*] Computing Set of Available Indices in Dataset...")
            available_indices = pool_dataset.indices
            idx2available = {x: i for i, x in enumerate(available_indices)}

            # Create Validation Dataset
            print("\n[*] Initializing Validation Dataset...")
            val_dataset = VQAGridDataset(
                dictionary,
                ans2label,
                label2ans,
                val_img2idx,
                vqa2_q=args.vqa2_questions,
                vqa2_g=args.vqa2_grid,
                split=args.split,
                mode="val",
                mtype=args.mode,
            )

            # Build Initial Train Dataset
            print(
                "\n[*] Iteration 0 [%d Examples] :: Active Learning Dataset..."
                % N_EXAMPLES[args.dataset][args.split][args.burn][0]
            )
            active_indices = list(
                np.random.choice(
                    available_indices, size=N_EXAMPLES[args.dataset][args.split][args.burn][0], replace=False
                )
            )
            active_dataset = VQAGridIndexDataset(
                dictionary,
                ans2label,
                label2ans,
                train_img2idx,
                indices=list(active_indices),
                vqa2_q=args.vqa2_questions,
                vqa2_g=args.vqa2_grid,
                split=args.split,
                mode="train",
                mtype=args.mode,
            )

            # Cheat Memory --> Use Features/Spatials from Pool Dataset :)
            active_dataset.features = pool_dataset.features

        elif args.mode in ["olreg", "butd", "lxmert"]:
            print("\n[*] Pre-processing VQA-2 Object Image Features...")
            train_img2idx, val_img2idx = vqa2_create_object_features(vqa2_f=args.vqa2_features, cache=args.vqa2_cache)

            print("\n[*] Initializing Full Training Dataset...")
            pool_dataset = VQAObjectDataset(
                dictionary,
                ans2label,
                label2ans,
                train_img2idx,
                frontier_indices,
                vqa2_q=args.vqa2_questions,
                cache=args.vqa2_cache,
                split=args.split,
                mode="train",
                lxmert=args.mode == "lxmert",
                lxmert_cache=args.lxmert_cache,
            )

            # Compute Set of "available indices"
            print("\n[*] Computing Set of Available Indices in Dataset...")
            available_indices = pool_dataset.indices
            idx2available = {x: i for i, x in enumerate(available_indices)}

            # Create Validation Dataset
            print("\n[*] Initializing Validation Dataset...")
            val_dataset = VQAObjectDataset(
                dictionary,
                ans2label,
                label2ans,
                val_img2idx,
                vqa2_q=args.vqa2_questions,
                cache=args.vqa2_cache,
                split=args.split,
                mode="val",
                lxmert=args.mode == "lxmert",
                lxmert_cache=args.lxmert_cache,
            )

            # Build Initial Train Dataset
            print(
                "\n[*] Iteration 0 [%d Examples] :: Active Learning Dataset..."
                % N_EXAMPLES[args.dataset][args.split][args.burn][0]
            )
            active_indices = list(
                np.random.choice(
                    available_indices, size=N_EXAMPLES[args.dataset][args.split][args.burn][0], replace=False
                )
            )

            active_dataset = VQAObjectIndexDataset(
                dictionary,
                ans2label,
                label2ans,
                train_img2idx,
                indices=list(active_indices),
                vqa2_q=args.vqa2_questions,
                cache=args.vqa2_cache,
                split=args.split,
                mode="train",
                lxmert=args.mode == "lxmert",
                lxmert_cache=args.lxmert_cache,
            )

            # Cheat Memory --> Use Features/Spatials from Pool Dataset :)
            active_dataset.features = pool_dataset.features
            active_dataset.spatials = pool_dataset.spatials

        else:
            raise NotImplementedError("No preprocessing pipeline for Mode '%s'" % args.mode)

    # Iterate through N_EXAMPLES
    ALL_TRAIN_INDICES = {}
    for EX_IDX, EX in enumerate(N_EXAMPLES[args.dataset][args.split][args.burn]):
        # Setup Run Name
        print(
            "[*] Starting Train Job w/ %d Examples in Mode Indexed %s for split '%s'!" % (EX, args.dataset, args.split)
        )
        if args.run_name is None:
            run_name = (
                "%s-%s-%s-%s-nex-%d-%s-x%d"
                % (args.dataset, args.split, args.mode, args.strategy, EX, args.burn, args.seed)
                + "+"
                + datetime.now().strftime("%m-%d-[%H:%M]")
            )
        else:
            run_name = args.run_name + "-%d" % EX + "+" + datetime.now().strftime("%m-%d-[%H:%M]")

        # Set Randomness
        print("[*] Setting Random Seed to %d!" % args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # Setup Logger
        mt_logger = MetricLogger(name=run_name, save_dir=args.save_dir)

        # Update Args w/ N-Examples
        args.nexamples = EX

        # Set Batch Size for LXMERT
        args.bsz = 128 if args.mode == "lxmert" else args.bsz

        # Save Active Indices over time (for mapping)
        ALL_TRAIN_INDICES[EX_IDX] = list(map(int, active_indices))
        indices_path = os.path.join(args.save_dir, "active-indices", run_name)
        if not os.path.exists(indices_path):
            os.makedirs(indices_path)

        with open(os.path.join(indices_path, "active-indices.json"), "w") as f:
            json.dump(ALL_TRAIN_INDICES, f)

        # Create Model
        if args.mode == "glreg":
            nn = GridLogisticRegression(args, active_dataset, val_dataset, ans2label, label2ans)
            nn.w_emb.load_embeddings(emb)

        elif args.mode == "olreg":
            nn = ObjectLogisticRegression(args, active_dataset, val_dataset, ans2label, label2ans)
            nn.w_emb.load_embeddings(emb)

        elif args.mode == "cnn":
            # Update Corresponding Arguments
            args.rnn, args.rnn_layers = "LSTM", 2
            nn = LSTMCNN(args, active_dataset, val_dataset, ans2label, label2ans)
            nn.w_emb.load_embeddings(emb)

        elif args.mode == "butd":
            nn = BUTD(args, active_dataset, val_dataset, ans2label, label2ans)
            nn.w_emb.load_embeddings(emb)

        elif args.mode == "lxmert":
            print("[*] Spinning up LXMERT!")
            nn = LXMERT(args, active_dataset, val_dataset, ans2label, label2ans)

            # Set Epochs = 5 (https://github.com/airsplay/lxmert)
            if args.split == "all":
                args.epochs = 5
            else:
                args.epochs = 10

        else:
            raise NotImplementedError("Model %s not yet implemented -- try < butd >" % args.mode)

        # Create DataLoaders
        active_dataloader = DataLoader(active_dataset, batch_size=args.bsz, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=args.bsz, num_workers=4)

        # Create Trainer
        print("\n[*] Training...\n")
        nn.train()
        mt_logger.log_hyperparams(args)
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(args.save_dir, "runs", run_name),
            filename= args.mode + "-{epoch:02d}-{val_loss:.3f}-{val_acc:.3f}",
            monitor="val_acc",
            mode="max",
            save_top_k=1,
        )

        trainer = pl.Trainer(
            default_root_dir=args.save_dir,
            max_epochs=args.epochs,
            gradient_clip_val=args.gradient_clip,
            gpus=args.gpus,
            benchmark=True,
            logger=False if not args.sync else mt_logger,
            callbacks=[checkpoint_callback],
        )

        # Fit
        trainer.fit(nn, active_dataloader, val_dataloader)

        # New Dataset Curation
        nn.eval()

        # Helper Variables for Coresets
        if EX_IDX < len(N_EXAMPLES[args.dataset][args.split][args.burn]) - 1:
            # Compute how many examples to retrieve this iteration!
            dset, seen = args.dataset, set(active_indices)
            before, after = (
                N_EXAMPLES[dset][args.split][args.burn][EX_IDX],
                N_EXAMPLES[dset][args.split][args.burn][EX_IDX + 1],
            )

            if args.strategy == "baseline":
                print("\n[*] Evaluating Random Baseline...")

                # Pick another random subset --> first normalize probabilities (zero-out seen examples)
                probabilities = np.ones(len(available_indices))
                for i, idx in enumerate(available_indices):
                    if idx in seen:
                        probabilities[i] = 0
                probabilities /= np.sum(probabilities)

                # Sample Add Indices
                add_indices = np.random.choice(available_indices, size=(after - before), p=probabilities, replace=False)

                # Make sure there is no overlap!
                assert set(active_indices).isdisjoint(add_indices)
                active_indices += list(add_indices)
                assert len(active_indices) == after, "Something went horribly wrong with the random baseline..."

            elif args.strategy == "least-conf":
                print("\n[*] Evaluating Uncertainty Sampling :: Least Confidence...")

                # Evaluate
                dl, max_probs = DataLoader(pool_dataset, batch_size=args.bsz, shuffle=False, num_workers=4), []
                nn.cuda()
                print("\n[*] Evaluating Greedy Least-Conf on Entire Pool Set...")
                for i, batch in tqdm(enumerate(dl)):
                    batch = [x.cuda() for x in batch]
                    r_probs = nn.active_step(batch, i, mode="least-conf")
                    max_probs.extend(r_probs)

                # Create Probability Gap + Pick ARGMAX Values (Greedily)
                prob_gap = 1 - np.array(max_probs)
                for i, idx in enumerate(available_indices):
                    if idx in seen:
                        prob_gap[i] = 0
                    else:
                        # Guard against Max-Conf = 1.0...
                        prob_gap[i] += 1e-9
                prob_gap /= np.sum(prob_gap)

                # Get Greedy Add Indices
                k = after - before
                add_indices = [available_indices[x] for x in np.argpartition(prob_gap, -k)[-k:]]

                # Make sure there is no overlap!
                assert set(active_indices).isdisjoint(add_indices)
                active_indices += list(add_indices)
                assert len(active_indices) == after, "Something went horribly wrong with least confidence..."

            elif args.strategy == "entropy":
                print("\n[*] Evaluating Uncertainty-Sampling :: Entropy...")

                # Evaluate
                dl, entropies = DataLoader(pool_dataset, batch_size=args.bsz, shuffle=False, num_workers=4), []
                nn.cuda()
                print("\n[*] Evaluating Entropy on Entire Pool Set...")
                for i, batch in tqdm(enumerate(dl)):
                    batch = [x.cuda() for x in batch]
                    r_ents = nn.active_step(batch, i, mode="entropy")
                    entropies.extend(r_ents)

                # Create Normalized Probabilities
                entropies = np.array(entropies)
                for i, idx in enumerate(available_indices):
                    if idx in seen:
                        entropies[i] = 0
                entropies /= np.sum(entropies)

                # Get Greedy Add Indices
                k = after - before
                add_indices = [available_indices[x] for x in np.argpartition(entropies, -k)[-k:]]

                # Make sure there is no overlap!
                assert set(active_indices).isdisjoint(add_indices)
                active_indices += list(add_indices)
                assert len(active_indices) == after, "Something went horribly wrong with entropy..."

            elif args.strategy == "mc-entropy":
                print("\n[*] Evaluating Monte-Carlo Dropout :: Entropy-Based Acquisition...")

                # Run Monte-Carlo Dropout w/ Entropy --> Set back in Train Mode!
                nn.train()

                # Evaluate
                dl, entropies = DataLoader(pool_dataset, batch_size=args.bsz, shuffle=False, num_workers=4), []
                nn.cuda()
                for i, batch in tqdm(enumerate(dl)):
                    batch = [x.cuda() for x in batch]
                    ents = nn.mc_step(batch, i, k=args.k_dropout, mode="entropy")
                    entropies.extend(ents)

                # Create Normalized Probabilities
                entropies = np.array(entropies)
                for i, idx in enumerate(available_indices):
                    if idx in seen:
                        entropies[i] = 0
                    else:
                        # Guard against MC-Entropy = 0.0...
                        entropies[i] += 1e-9
                entropies /= np.sum(entropies)

                # Get Greedy Add Indices
                k = after - before
                add_indices = [available_indices[x] for x in np.argpartition(entropies, -k)[-k:]]

                # Make sure there is no overlap!
                assert set(active_indices).isdisjoint(add_indices)
                active_indices += list(add_indices)
                assert len(active_indices) == after, "Something went horribly wrong with mc-dropout entropy..."

            elif args.strategy == "mc-bald":
                print("\n[*] Evaluating Monte-Carlo Dropout :: BALD-Based Acquisition...")

                # Run Monte-Carlo Dropout w/ BALD --> Set back in Train Mode!
                nn.train()

                # Evaluate
                dl, infos = DataLoader(pool_dataset, batch_size=args.bsz, shuffle=False, num_workers=4), []
                nn.cuda()
                for i, batch in tqdm(enumerate(dl)):
                    batch = [x.cuda() for x in batch]
                    info = nn.mc_step(batch, i, k=args.k_dropout, mode="bald")
                    infos.extend(info)

                # BALD not guaranteed to be positive... set "seen" to be large negative value (don't normalize!)
                infos = np.array(infos)
                for i, idx in enumerate(available_indices):
                    if idx in seen:
                        infos[i] = -999999

                # Get Greedy Add Indices
                k = after - before
                add_indices = [available_indices[x] for x in np.argpartition(infos, -k)[-k:]]

                # Make sure there is no overlap!
                assert set(active_indices).isdisjoint(add_indices)
                active_indices += list(add_indices)
                assert len(active_indices) == after, "Something went horribly wrong with mc-dropout bald..."

            elif "coreset" in args.strategy:
                representation = args.strategy.split("-")[1]
                print("\n[*] Evaluating Coresets w/ Representation %s..." % representation.capitalize())

                # Evaluate and Extract all Features
                dl, features = DataLoader(pool_dataset, batch_size=args.bsz, shuffle=False, num_workers=4), []
                nn.cuda()
                for i, batch in tqdm(enumerate(dl)):
                    batch = [x.cuda() for x in batch]
                    feats = nn.extract(batch, i, mode=representation)
                    features.append(feats)

                # Create Features
                start_time = time.time()
                features, n_choose, min_distances = np.concatenate(features, axis=0), after - before, None

                # Swap on Split --> if big split ("all"), run 'Approximate Coresets'
                if args.split in ["all", "f50", "f60", "f75", "f90"]:
                    print("\t[*] Computing Approximate Coresets on Split %s..." % args.split.capitalize())

                    # Initialize PCA w/ appropriate parameters
                    print("\t[*] Downsampling with PCA...")
                    pca_time = time.time()
                    pca = PCA(n_components=args.pca_components)

                    # Down-sample Features w/ PCA
                    features = pca.fit_transform(features)
                    print("\t[*] PCA took %.3f seconds..." % (time.time() - pca_time))

                # Otherwise it's a small enough dataset and we can run 'Exact Coresets'
                else:
                    print("\t[*] Computing Exact Coresets on Split %s..." % args.split.capitalize())

                # Ref :: https://github.com/google/active-learning/blob/master/sampling_methods/kcenter_greedy.py
                def update_distances(cluster_centers, min_distances, only_new=True):
                    if only_new:
                        cluster_centers = [d for d in cluster_centers if available_indices[d] not in active_indices]

                    # Get Cluster Center Features and compute Pairwise Distances between All and Cluster Centers
                    cluster_feats = features[cluster_centers]

                    # Update Min_Distances for All Examples given new Cluster Centers
                    if min_distances is None:
                        # Compute "effective" features
                        cluster_index_set = set(cluster_centers)
                        effective_features = [x for x in range(len(features)) if x not in cluster_index_set]

                        def batch(iterable, n=2048):
                            l = len(iterable)
                            for ndx in range(0, l, n):
                                yield ndx, min(ndx + n, l)

                        # Iterate through Generator and Build Up Minimum Distances
                        min_distances = np.zeros((len(features), 1))
                        for start, end in tqdm(batch(effective_features), total=len(effective_features) // 2048):
                            d = pairwise_distances(
                                features[effective_features[start:end]], cluster_feats, metric="sqeuclidean"
                            )
                            min_distances[effective_features[start:end], 0] = d.min(axis=1)

                    else:
                        dist = pairwise_distances(features, cluster_feats, metric="sqeuclidean")
                        min_distances = np.minimum(min_distances, dist).min(axis=1).reshape(-1, 1)

                    return min_distances

                # Initial Pass (Compute all Distances)
                print("\t[*] Initial Full Pairwise Distance Pass")
                initial_pass = time.time()
                seen_indices = [idx2available[x] for x in active_indices]
                min_distances = update_distances(seen_indices, min_distances, only_new=False)

                # Enforce Minimums are 0!
                for i in seen_indices:
                    min_distances[i] = 0.0

                print("\t[*] Initial Pairwise Distance Pass took %.3f seconds..." % (time.time() - initial_pass))

                # Initialize Add-Indices and Start Greedily Adding
                add_indices, leftover_set = [], set()
                print("\t[*] Choosing Examples and Recomputing Distances")

                # Swap on Split --> if 'all', run 'Approximate Coresets'
                if args.split in ["all", "f50", "f60", "f75", "f90"]:
                    # Implement Amortized Coresets --> Split into <amortized_iterations> updates
                    n_choose_per_ai = n_choose // args.amortized_iterations
                    for ai in range(args.amortized_iterations):
                        print("\t\t[*] Amortized Iteration - %d" % ai)
                        ai_time = time.time()

                        # Chosen Indices
                        ai_inds = []

                        # Boundary Handling
                        nc = (
                            n_choose_per_ai
                            if ai < (args.amortized_iterations - 1)
                            else n_choose_per_ai + (n_choose % args.amortized_iterations)
                        )

                        # Choose!
                        for _ in tqdm(list(range(nc))):
                            ind = np.argmax(min_distances)
                            if available_indices[ind] in active_indices or available_indices[ind] in add_indices:
                                # For whatever reason, distance computation from clusters is returning 0...
                                print("\t\t[*] Duplicated features/representations --> Picking Randomly")
                                if len(leftover_set) == 0:
                                    leftover_set = set(available_indices) - set(active_indices).union(set(add_indices))

                                # Pop from Leftover Set
                                add_indices.append(leftover_set.pop())

                            else:
                                assert (
                                    available_indices[ind] not in active_indices
                                    and available_indices[ind] not in add_indices
                                ), "I should never be getting here!"

                                # Update Distances w/ New Cluster Center
                                add_indices.append(available_indices[ind])
                                ai_inds.append(ind)

                                # Enforce Minimums are 0!
                                min_distances[ind] = 0.0

                        # Recompute Distances if necessary
                        if len(ai_inds) > 0:
                            print("\t\t[*] Recomputing Diversity/Inner-Batch Coreset Pairwise Distances...")
                            min_distances = update_distances(ai_inds, min_distances, only_new=True)

                        print(
                            "\t\t[*] Amortized Iteration %d / %d Time: %.3f..."
                            % (ai, args.amortized_iterations, time.time() - ai_time)
                        )

                # Otherwise --> small enough dataset, run 'Exact Coresets'
                else:
                    for _ in tqdm(list(range(n_choose))):
                        ind = np.argmax(min_distances)
                        if available_indices[ind] in active_indices or available_indices[ind] in add_indices:
                            # For whatever reason, distance computation from clusters is returning 0...
                            print("\t[*] Duplicated features/representations --> Picking Randomly")
                            if len(leftover_set) == 0:
                                leftover_set = set(available_indices) - set(active_indices).union(set(add_indices))

                            # Pop from Leftover Set
                            add_indices.append(leftover_set.pop())

                        else:
                            assert (
                                available_indices[ind] not in active_indices
                                and available_indices[ind] not in add_indices
                            ), "I should never be getting here!"

                            # Update Distances w/ New Cluster Center
                            min_distances = update_distances([ind], min_distances, only_new=True)
                            add_indices.append(available_indices[ind])

                            # Enforce Minimums are 0!
                            min_distances[ind] = 0.0

                # Make sure there is no overlap!
                assert set(active_indices).isdisjoint(add_indices)
                active_indices += list(add_indices)
                assert len(active_indices) == after, "Something went horribly wrong with coresets..."

                end_time = time.time()
                print("[*] Takes %.3f seconds to do Coresets Iteration!" % (end_time - start_time))

            # Create New Active Dataset
            print("\n[*] Creating New Training Dataset via Active Learning...")
            if args.dataset == "gqa":
                if args.mode in ["glreg", "cnn"]:
                    active_dataset = GQAGridIndexDataset(
                        dictionary,
                        ans2label,
                        label2ans,
                        trainval_img2idx,
                        indices=list(active_indices),
                        gqa_q=args.gqa_questions,
                        gqa_g=args.gqa_grid,
                        split=args.split,
                        mode="train",
                    )

                    # Cheat Memory (Not Really cuz it's just a file handle) --> Use Features from Pool Dataset
                    active_dataset.features = pool_dataset.features

                elif args.mode in ["olreg", "butd", "lxmert"]:
                    active_dataset = GQAObjectIndexDataset(
                        dictionary,
                        ans2label,
                        label2ans,
                        trainval_img2idx,
                        indices=list(active_indices),
                        gqa_q=args.gqa_questions,
                        cache=args.gqa_cache,
                        split=args.split,
                        mode="train",
                        lxmert=args.mode == "lxmert",
                        lxmert_cache=args.lxmert_cache,
                    )

                    # Cheat Memory (Not Really cuz it's just a file handle) --> Use Features/Spatials from Pool Dataset
                    active_dataset.features = pool_dataset.features
                    active_dataset.spatials = pool_dataset.spatials

            elif args.dataset in ["vqa2", "vqa2-frontier"]:
                if args.mode in ["glreg", "cnn"]:
                    active_dataset = VQAGridIndexDataset(
                        dictionary,
                        ans2label,
                        label2ans,
                        train_img2idx,
                        indices=list(active_indices),
                        vqa2_q=args.vqa2_questions,
                        vqa2_g=args.vqa2_grid,
                        split=args.split,
                        mode="train",
                        mtype=args.mode,
                    )

                    # Cheat Memory --> Use Features from Pool Dataset :)
                    active_dataset.features = pool_dataset.features

                elif args.mode in ["olreg", "butd", "lxmert"]:
                    active_dataset = VQAObjectIndexDataset(
                        dictionary,
                        ans2label,
                        label2ans,
                        train_img2idx,
                        indices=list(active_indices),
                        vqa2_q=args.vqa2_questions,
                        cache=args.vqa2_cache,
                        split=args.split,
                        mode="train",
                        lxmert=args.mode == "lxmert",
                        lxmert_cache=args.lxmert_cache,
                    )

                    # Cheat Memory --> Use Features/Spatials from Pool Dataset :)
                    active_dataset.features = pool_dataset.features
                    active_dataset.spatials = pool_dataset.spatials

            else:
                raise NotImplementedError("Dataset '%s' not supported!" % args.dataset)


if __name__ == "__main__":
    active()
