"""
lxmert.py

Implementation of LXMERT for VQA, pretrained on non-VQA data (checkpoint courtesy of Hao Tan @ UNC)!

Reference: Based mostly of off HuggingFace & Hao Tan's Original LXMERT Implementation:
    - https://huggingface.co/transformers/model_doc/lxmert.html
    - https://github.com/airsplay/lxmert
"""
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from scipy.stats import entropy
from transformers import AdamW, LxmertForQuestionAnswering


class LXMERT(pl.LightningModule):
    def __init__(
        self,
        hparams,
        train_dataset,
        val_dataset,
        ans2label=None,
        label2ans=None,
        chart=False,
        chart_val=False,
        k_dropout=10,
    ):
        super(LXMERT, self).__init__()

        # Retrieve Config & LXMERT-Cache Path
        self.lxmert_config, self.lxmert_cache = train_dataset.lxmert_config, hparams.lxmert_cache

        # Save Hyper-Parameters and Dataset
        self.save_hyperparameters(hparams)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        self.ans2label, self.label2ans = ans2label, label2ans
        self.chart, self.chart_val, self.k_dropout = chart, chart_val, k_dropout

        # Set Default Batch Size to 128 and Max Grad Norm to 0.5 (from LXMERT repo)
        self.hparams.bsz = 128
        self.hparams.gradient_clip = 0.5

        # Build Model
        self.build_model()

    def build_model(self):
        # Create Model from Pretrained (HF API)
        self.lxrt = LxmertForQuestionAnswering.from_pretrained(
            None, config=self.lxmert_config, state_dict=torch.load(os.path.join(self.lxmert_cache, "Epoch19_LXRT.pth"))
        )

    def forward(self, input_ids, attn, obj_features, obj_boxes, token_types):
        output = self.lxrt(
            input_ids=input_ids.squeeze(),
            attention_mask=attn.squeeze(),
            token_type_ids=token_types.squeeze(),
            visual_feats=obj_features,
            visual_pos=obj_boxes,
        )

        return output.question_answering_score

    def configure_optimizers(self):
        # From https://github.com/airsplay/lxmert --> Different Learning Rates for VQA/GQA
        lr = 5e-5 if self.hparams.dataset != "gqa" else 1e-5
        optimizer = AdamW(self.parameters(), lr=lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        input_ids, attn, obj_features, obj_boxes, token_types, answer, idxs = train_batch

        # Run Forward Pass
        logits = self.forward(input_ids, attn, obj_features, obj_boxes, token_types)

        # Dataset Cartography
        if self.chart or self.chart_val:
            # Compute Probabilities
            probabilities = torch.softmax(logits, dim=1)
            hot_answers = torch.nn.functional.one_hot(answer, num_classes=len(self.ans2label))
            class_probabilities = torch.sum(probabilities * hot_answers, dim=1)
            max_probabilities, _ = torch.max(probabilities, dim=1)

            # Create Dictionary Mapping idxs --> class_probabilities, max_probabilities (if equal, then correct!)
            bdict_conf = {}
            lidxs, lcp = idxs.cpu().numpy().tolist(), class_probabilities.detach().cpu().numpy().tolist()
            lmp = max_probabilities.detach().cpu().numpy().tolist()
            for i in range(len(lidxs)):
                bdict_conf[lidxs[i]] = (lcp[i], lmp[i])

        # Compute Loss (Cross-Entropy)
        loss = nn.functional.cross_entropy(logits, answer)

        # Compute Answer Accuracy
        accuracy = torch.mean((logits.argmax(dim=1) == answer).float())

        # Set up Data to be Logged
        log = {"train_loss": loss, "train_acc": accuracy}
        if self.chart or self.chart_val:
            return {
                "loss": loss,
                "train_loss": loss,
                "train_acc": accuracy,
                "cartography": bdict_conf,
                "progress_bar": log,
                "log": log,
            }
        else:
            return {"loss": loss, "train_loss": loss, "train_acc": accuracy, "progress_bar": log, "log": log}

    def training_epoch_end(self, outputs):
        # Outputs --> List of Individual Step Outputs
        avg_loss = torch.stack([x["train_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["train_acc"] for x in outputs]).mean()

        if self.chart or self.chart_val:
            cartography_dict = {}
            for x in outputs:
                cartography_dict.update(x["cartography"])

        pbar = {"train_epoch_loss": avg_loss, "train_epoch_acc": avg_acc}
        if self.chart or self.chart_val:
            log = {"train_epoch_loss": avg_loss, "train_epoch_acc": avg_acc, "cartography": cartography_dict}
        else:
            log = dict(pbar)

        for k, v in log.items():
            self.log(k, v)

    def validation_step(self, val_batch, batch_idx):
        input_ids, attn, obj_features, obj_boxes, token_types, answer, idxs = val_batch

        # Run Forward Pass
        logits = self.forward(input_ids, attn, obj_features, obj_boxes, token_types)
        if self.chart_val:
            # Compute Probabilities
            probabilities = torch.softmax(logits, dim=1)
            hot_answers = torch.nn.functional.one_hot(answer, num_classes=len(self.ans2label))
            class_probabilities = torch.sum(probabilities * hot_answers, dim=1)
            max_probabilities, _ = torch.max(probabilities, dim=1)

            # Create Dictionary Mapping idxs --> class_probabilities
            bdict_conf = {}
            lidxs, lcp = idxs.cpu().numpy().tolist(), class_probabilities.detach().cpu().numpy().tolist()
            lmp = max_probabilities.cpu().numpy().tolist()
            for i in range(len(lidxs)):
                bdict_conf[lidxs[i]] = (lcp[i], lmp[i])

            # MC-Dropout Probabilities!
            self.train()
            probabilities = []
            for _ in range(self.k_dropout):
                lg = self.forward(input_ids, attn, obj_features, obj_boxes, token_types)
                probability = torch.softmax(lg, dim=1).detach().cpu().numpy()
                probabilities.append(probability)

            # Compute Average Probability across "Ensemble"
            probabilities = np.mean(probabilities, axis=0)
            hot_answers = hot_answers.cpu().numpy()
            class_probabilities = np.sum(probabilities * hot_answers, axis=1)
            max_probabilities = np.max(probabilities, axis=1)

            # Create Dictionary Mapping idxs --> class_probabilities
            mcdict_conf = {}
            mcidxs, mcp = idxs.cpu().numpy().tolist(), class_probabilities.tolist()
            mcmp = max_probabilities.tolist()
            for i in range(len(mcidxs)):
                mcdict_conf[mcidxs[i]] = (mcp[i], mcmp[i])
            self.eval()

        # Compute Loss (Cross-Entropy)
        loss = nn.functional.cross_entropy(logits, answer)

        # Compute Answer Accuracy
        accuracy = torch.mean((logits.argmax(dim=1) == answer).float())

        if self.chart_val:
            return {
                "val_loss": loss,
                "val_acc": accuracy,
                "val_cartography": bdict_conf,
                "val_mc_cartography": mcdict_conf,
            }
        else:
            return {"val_loss": loss, "val_acc": accuracy}

    def validation_epoch_end(self, outputs):
        # Outputs --> List of Individual Step Outputs
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()

        if self.chart_val:
            cartography_dict, mc_cartography_dict = {}, {}
            for x in outputs:
                cartography_dict.update(x["val_cartography"])
                mc_cartography_dict.update(x["val_mc_cartography"])

        pbar = {"val_loss": avg_loss, "val_acc": avg_acc}
        if self.chart_val:
            log = {
                "val_loss": avg_loss,
                "val_acc": avg_acc,
                "val_cartography": cartography_dict,
                "val_mc_cartography": mc_cartography_dict,
            }
        else:
            log = dict(pbar)

        return {"progress_bar": pbar, "log": log}

    def active_step(self, active_batch, batch_idx, mode="least-conf"):
        # Run Uncertainty Sampling with the given acquisition strategy
        input_ids, attn, obj_features, obj_boxes, token_types, _, _ = active_batch

        # Run Forward Pass
        with torch.no_grad():
            logits = self.forward(input_ids, attn, obj_features, obj_boxes, token_types)

            if mode in ["least-conf"]:
                probabilities, _ = torch.max(torch.softmax(logits, dim=1), dim=1)
                probabilities = probabilities.detach().cpu().numpy()
                return list(probabilities)

            elif mode in ["entropy"]:
                probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
                entropies = entropy(probabilities, axis=1)
                return list(entropies)

    def mc_step(self, active_batch, batch_idx, k=10, mode="entropy"):
        # Run Monte-Carlo Dropout w/ the given acquisition strategy
        input_ids, attn, obj_features, obj_boxes, token_types, _, _ = active_batch

        with torch.no_grad():
            if mode == "entropy":
                # Accumulate Output Probabilities
                probabilities = []
                for _ in range(k):
                    logits = self.forward(input_ids, attn, obj_features, obj_boxes, token_types)
                    probability = torch.softmax(logits, dim=1).detach().cpu().numpy()
                    probabilities.append(probability)

                # Compute Entropy of Average...
                entropies = entropy(np.mean(probabilities, axis=0), axis=1)
                return list(entropies)

            elif mode == "bald":
                # Accumulate Output Probabilities (Term 1) and per-run entropies (Term 2)
                probabilities, disagreement = [], []
                for _ in range(k):
                    logits = self.forward(input_ids, attn, obj_features, obj_boxes, token_types)
                    probability = torch.softmax(logits, dim=1).detach().cpu().numpy()

                    probabilities.append(probability)
                    disagreement.append(entropy(probability, axis=1))

                # Compute Entropy of Average
                entropies = entropy(np.mean(probabilities, axis=0), axis=1)
                disagreements = np.mean(disagreement, axis=0)
                return list(entropies - disagreements)

    def extract(self, extract_batch, batch_idx, mode="fused"):
        raise NotImplementedError("Under no circumstances are you going to run Coresets for LXMERT again!")
