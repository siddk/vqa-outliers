"""
cartograph.py

Run Dataset Cartography (https://arxiv.org/abs/2009.10795) on various splits of the VQA dataset conditioned on a model,
tracking statistics per-example over training to create Dataset Maps.

Additionally saves model checkpoints and logs training statistics.
"""
import os
import random
from argparse import Namespace
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from tap import Tap
from torch.utils.data import DataLoader

from src.logging import MetricLogger
from src.models import BUTD, LSTMCNN, LXMERT, GridLogisticRegression, ObjectLogisticRegression
from src.preprocessing.gqa import (
    GQAGridDataset,
    GQAObjectDataset,
    gqa_create_answers,
    gqa_create_dictionary_glove,
    gqa_create_grid_features,
    gqa_create_object_features,
)
from src.preprocessing.vqa2 import (
    VQAObjectDataset,
    VQAGridDataset,
    vqa2_create_answers,
    vqa2_create_object_features,
    vqa2_create_dictionary_glove,
    vqa2_create_grid_features,
)


class ArgumentParser(Tap):
    # fmt: off

    # Data and Checkpoint Parameters
    data_dir: str = "data/"                         # Path to downloaded data
    save_dir: str = "checkpoints/map"               # Path to checkpoints, serialized statistics, and artifacts

    # GQA Specific Parameters
    gqa_questions: str = "data/GQA-Questions"       # Path to GQA Questions
    gqa_features: str = "data/GQA-Features"         # Path to GQA Features
    gqa_grid: str = "data/GQA-Spatials"             # Path to GQA Spatial/Grid Features
    gqa_cache: str = "data/GQA-Cache"               # Path to GQA Cache Directory for storing serialized data

    # VQA-2 Specific Parameters
    vqa2_questions: str = "data/VQA2-Questions"     # Path to VQA-2 Questions
    vqa2_features: str = "data/VQA2-Features"       # Path to VQA-2 Features
    vqa2_grid: str = "data/VQA2-Spatials"           # Path to VQA-2 Spatial/Grid Features
    vqa2_cache: str = "data/VQA2-Cache"             # Path to VQA-2 Cache Directory for storing serialized data

    # GloVe Vectors
    glove: str = "data/GloVe/glove.6B.300d.txt"     # Path to GloVe Embeddings File (300-dim)

    # LXMERT (HF Transformers Cache)
    lxmert_cache: str = "data/LXMERT"               # Path to LXMERT Checkpoint & Cache Directory

    # Run/WandB Parameters
    sync: bool = False                              # Whether or not to store logs & artifacts
    run_name: str = None                            # Informative Run-ID for saving logs & artifacts

    # GPUs
    gpus: int = 0                                   # Number of GPUs to run with

    # Modes
    dataset: str = "vqa2"                           # Dataset to run model with -- < gqa | vqa2 >
    split: str = "all"                              # Dataset Split to Run with in (consult dict above for options)
    mode: str = "butd"                              # Mode to run - < glreg | olreg | cnn | butd | lxmert >

    # MC-Dropout Parameters
    k_dropout: int = 10                             # Number of MC Dropout Trials per Example

    # Model Parameters
    emb_dim: int = 300                              # Word Embedding Dimension --> Should Match GloVe (300)
    emb_dropout: float = 0.0                        # Dropout to Apply to Word Embeddings

    rnn: str = "GRU"                                # RNN Type for Question Encoder --> one of < 'GRU' | 'LSTM' >
    rnn_layers: int = 1                             # Number of RNN Stacked Layers (for Question Encoder)
    bidirectional: bool = False                     # Whether or not RNN is Bidirectional
    q_dropout: float = 0.0                          # RNN Dropout (for Question Encoder)

    fusion: str = "product"                         # Fusion for Attention --> one of < 'product' | 'concat' >
    attention_dropout: float = 0.2                  # Dropout for Attention Operation (fusing Image + Question)
    answer_dropout: float = 0.5                     # Dropout to Apply to Answer Classifier
    hidden: int = 1024                              # Dimensionality of Hidden Layer (Question & Object Encoder)
    weight_norm: bool = True                        # Boolean whether or not to use Weight Normalization
    weight_decay: float = 0.0                       # L2 Weight Decay Penalty for regularization

    # Training Parameters
    bsz: int = 512                                  # Batch Size --> the Bigger the Better
    epochs: int = 15                                # Number of Training Epochs

    opt: str = "adamax"                             # Optimizer for Performing Gradient Updates
    gradient_clip: float = 0.25                     # Value for Gradient Clipping

    # Random Seed
    seed: int = 7                                   # Random Seed (for Reproducibility)

    # fmt: on


def cartograph():
    # Parse Arguments --> Convert from Namespace --> Dict --> Namespace because of weird WandB Bug
    print("[*] Starting up...")
    args = Namespace(**ArgumentParser().parse_args().as_dict())
    print('\t> "Here be Dragons" (Anonymous)')

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
        print("\n[*] Pre-processing GQA Questions")
        dictionary, emb = gqa_create_dictionary_glove(gqa_q=args.gqa_questions, glove=args.glove, cache=args.gqa_cache)

        # Preprocess Answer Data
        ans2label, label2ans = gqa_create_answers(gqa_q=args.gqa_questions, cache=args.gqa_cache)

        # Create Image Features
        if args.mode in ["glreg", "cnn"]:
            print("\n[*] Pre-processing GQA Grid Image Features...")
            trainval_img2idx, testdev_img2idx = gqa_create_grid_features(
                gqa_g=args.gqa_grid, gqa_q=args.gqa_questions, cache=args.gqa_cache
            )

            # Create Train Dataset
            print("\n[*] Initializing Full Training Dataset...")
            train_dataset = GQAGridDataset(
                dictionary,
                ans2label,
                label2ans,
                trainval_img2idx,
                gqa_q=args.gqa_questions,
                gqa_g=args.gqa_grid,
                split=args.split,
                mode="train",
            )

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

        elif args.mode in ["olreg", "butd", "lxmert"]:
            print("\n[*] Pre-processing GQA Object Image Features")
            trainval_img2idx, testdev_img2idx = gqa_create_object_features(gqa_f=args.gqa_features, cache=args.gqa_cache)

            # Create Train Dataset
            print("\n[*] Initializing Full GQA Training Dataset...")
            train_dataset = GQAObjectDataset(
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

        else:
            raise NotImplementedError("No preprocessing pipeline for Mode '%s'" % args.mode)

    elif args.dataset == "vqa2":
        # Preprocess Question Data --> Return Dictionary and GloVe-initialized Embeddings
        print("\n[*] Pre-processing VQA-2 Questions...")
        dictionary, emb = vqa2_create_dictionary_glove(
            vqa2_q=args.vqa2_questions, glove=args.glove, cache=args.vqa2_cache
        )

        # Preprocess Answer Data
        print("\n[*] Pre-processing VQA-2 Answers...")
        ans2label, label2ans = vqa2_create_answers(split=args.split, vqa2_q=args.vqa2_questions, cache=args.vqa2_cache)

        # Create Image Features
        if args.mode in ["glreg", "cnn"]:
            print("\n[*] Pre-processing VQA-2 Grid Image Features...")
            train_img2idx, val_img2idx = vqa2_create_grid_features(vqa2_g=args.vqa2_grid)

            # Create Train Dataset
            print("\n[*] Initializing Full Training Dataset...")
            train_dataset = VQAGridDataset(
                dictionary,
                ans2label,
                label2ans,
                train_img2idx,
                vqa2_q=args.vqa2_questions,
                vqa2_g=args.vqa2_grid,
                split=args.split,
                mode="train",
                mtype=args.mode,
            )

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

        elif args.mode in ["olreg", "butd", "lxmert"]:
            print("\n[*] Pre-processing VQA-2 Object Image Features...")
            train_img2idx, val_img2idx = vqa2_create_object_features(vqa2_f=args.vqa2_features, cache=args.vqa2_cache)

            # Create Train Dataset
            print("\n[*] Initializing Full VQA-2 Training Dataset...")
            train_dataset = VQAObjectDataset(
                dictionary,
                ans2label,
                label2ans,
                train_img2idx,
                vqa2_q=args.vqa2_questions,
                cache=args.vqa2_cache,
                split=args.split,
                mode="train",
                lxmert=args.mode == "lxmert",
                lxmert_cache=args.lxmert_cache,
            )

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

        else:
            raise NotImplementedError("No preprocessing pipeline for Mode '%s'" % args.mode)

    # Setup Run Name
    print("[*] Starting Full Mapping Job for %s Dataset '%s'!" % (args.dataset.upper(), args.split))
    if args.run_name is None:
        run_name = (
            "%s-%s-%s-map" % (args.dataset, args.split, args.mode) + "+" + datetime.now().strftime("%m-%d-[%H:%M]")
        )
    else:
        run_name = args.run_name + "+" + datetime.now().strftime("%m-%d-[%H:%M]")

    # Create Model
    if args.mode == "glreg":
        nn = GridLogisticRegression(args, train_dataset, val_dataset, ans2label, label2ans, chart=True, chart_val=True)
        nn.w_emb.load_embeddings(emb)

    elif args.mode == "olreg":
        nn = ObjectLogisticRegression(args, train_dataset, val_dataset, ans2label, label2ans, chart=True, chart_val=True)
        nn.w_emb.load_embeddings(emb)

    elif args.mode == "cnn":
        # Update Corresponding Arguments
        args.rnn, args.rnn_layers = "LSTM", 2
        nn = LSTMCNN(
            args, train_dataset, val_dataset, ans2label, label2ans, chart=True, chart_val=True, k_dropout=args.k_dropout
        )
        nn.w_emb.load_embeddings(emb)

    elif args.mode == "butd":
        nn = BUTD(
            args, train_dataset, val_dataset, ans2label, label2ans, chart=True, chart_val=True, k_dropout=args.k_dropout
        )
        nn.w_emb.load_embeddings(emb)

    elif args.mode == "lxmert":
        nn = LXMERT(
            args, train_dataset, val_dataset, ans2label, label2ans, chart=True, chart_val=False, k_dropout=args.k_dropout
        )

        # Set Epochs = 5 for VQA/GQA All, otherwise 10 for Smaller Datasets (See: https://github.com/airsplay/lxmert)
        if args.split == "all":
            args.epochs = 5
        else:
            args.epochs = 10

    else:
        raise NotImplementedError("Model %s not yet implemented -- try < butd >" % args.mode)

    # Create Trainer
    print("\n[*] Training...\n")

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.bsz, num_workers=4)

    # Setup Logger
    mt_logger = MetricLogger(name=run_name, save_dir=args.save_dir)
    mt_logger.log_hyperparams(args)

    # Setup Checkpoints
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
    trainer.fit(nn, train_dataloader, val_dataloader)


if __name__ == "__main__":
    cartograph()
