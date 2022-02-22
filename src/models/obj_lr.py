"""
obj_lr.py

Implementation of BottomUp (Object) Feature-Based Logistic Regression for VQA.
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
from scipy.stats import entropy


class WordEmbedding(nn.Module):
    def __init__(self, ntoken, dim, dropout=0.0):
        """Initialize an Embedding Matrix with the appropriate dimensions --> Defines padding as last token in dict"""
        super(WordEmbedding, self).__init__()
        self.ntoken, self.dim = ntoken, dim
        self.emb = nn.Embedding(ntoken + 1, dim, padding_idx=ntoken)

        # Freeze Embeddings for Logistic Regression
        for param in self.parameters():
            param.requires_grad = False

    def load_embeddings(self, weights):
        """Set Embedding Weights from Numpy Array"""
        assert weights.shape == (self.ntoken, self.dim)
        self.emb.weight.data[: self.ntoken] = torch.from_numpy(weights)

    def forward(self, x):
        # x : [bsz, seq_len] --> [bsz, seq_len, emb_dim]
        return self.emb(x)


class ObjectLogisticRegression(pl.LightningModule):
    def __init__(
        self, hparams, train_dataset, val_dataset, ans2label=None, label2ans=None, chart=False, chart_val=False
    ):
        super(ObjectLogisticRegression, self).__init__()

        # Save Hyper-Parameters and Dataset
        self.save_hyperparameters(hparams)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        self.ans2label, self.label2ans = ans2label, label2ans
        self.chart, self.chart_val = chart, chart_val

        # Build Model
        self.build_model()

    def build_model(self):
        # Build Word Embeddings (for Questions)
        self.w_emb = WordEmbedding(ntoken=self.train_dataset.dictionary.ntoken, dim=self.hparams.emb_dim)
        self.pad_token = self.w_emb.ntoken

        # Linear Layer --> Woo Logistic Regression!
        self.linear = nn.Linear(self.train_dataset.v_dim + 6 + self.hparams.emb_dim, len(self.ans2label))

    def forward(self, image_features, spatial_features, question_features):
        # image_features: [bsz, K, image_dim]
        # question_features: [bsz, seq_len]

        # Embed and Encode Question --> [bsz, q_hidden]
        w_emb = self.w_emb(question_features)
        embs = w_emb.mean(dim=1)

        # Create new Image Features --> KEY POINT: Concatenate Spatial Features!
        image_features = torch.cat([image_features, spatial_features], dim=2)

        # Compute Mean-Pooled Image Features
        img_feats = image_features.mean(dim=1)

        # Joint Features
        joint = torch.cat([embs, img_feats], dim=1)

        # Return Logits
        return self.linear(joint)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def training_step(self, train_batch, batch_idx):
        img, spatials, question, answer, idxs = train_batch

        # Run Forward Pass
        logits = self.forward(img, spatials, question)

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

        if self.chart:
            cartography_dict = {}
            for x in outputs:
                cartography_dict.update(x["cartography"])

        pbar = {"train_epoch_loss": avg_loss, "train_epoch_acc": avg_acc}
        if self.chart:
            log = {"train_epoch_loss": avg_loss, "train_epoch_acc": avg_acc, "cartography": cartography_dict}
        else:
            log = dict(pbar)

        for k, v in log.items():
            self.log(k, v)

    def validation_step(self, val_batch, batch_idx):
        img, spatials, question, answer, idxs = val_batch

        # Run Forward Pass
        logits = self.forward(img, spatials, question)

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

        # Compute Loss (Cross-Entropy)
        loss = nn.functional.cross_entropy(logits, answer)

        # Compute Answer Accuracy
        accuracy = torch.mean((logits.argmax(dim=1) == answer).float())

        if self.chart_val:
            return {"val_loss": loss, "val_acc": accuracy, "val_cartography": bdict_conf}
        else:
            return {"val_loss": loss, "val_acc": accuracy}

    def validation_epoch_end(self, outputs):
        # Outputs --> List of Individual Step Outputs
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()

        if self.chart_val:
            cartography_dict = {}
            for x in outputs:
                cartography_dict.update(x["val_cartography"])

        pbar = {"val_loss": avg_loss, "val_acc": avg_acc}
        if self.chart_val:
            log = {"val_loss": avg_loss, "val_acc": avg_acc, "val_cartography": cartography_dict}
        else:
            log = dict(pbar)

        return {"progress_bar": pbar, "log": log}

    def active_step(self, active_batch, batch_idx, mode="least-conf"):
        # Max-Prob/Least Conf
        img, spatials, question, _, _ = active_batch

        # Run Forward Pass
        logits = self.forward(img, spatials, question)

        if mode in ["max-prob", "least-conf"]:
            probabilities, _ = torch.max(torch.softmax(logits, dim=1), dim=1)

            probabilities = probabilities.detach().cpu().numpy()
            return list(probabilities)

        elif mode in ["entropy"]:
            probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
            entropies = entropy(probabilities, axis=1)
            return list(entropies)

    def extract(self, extract_batch, batch_idx, mode="fused"):
        # Extract a specific (multi/single)-modal representation that the model builds for Coresets.
        img, spatials, question, _, _ = extract_batch

        if mode == "language":
            # Get Encoding pre-projecting
            w_emb = self.w_emb(question)
            enc = w_emb.mean(dim=1)

        elif mode == "vision":
            # Get Image Features concatenated with Spatial Features
            enc = torch.cat([img, spatials], dim=2).mean(dim=1)

        elif mode == "fused":
            # Embed and Encode Question --> [bsz, q_hidden]
            w_emb = self.w_emb(question)
            embs = w_emb.mean(dim=1)

            # Create new Image Features --> KEY POINT: Concatenate Spatial Features!
            image_features = torch.cat([img, spatials], dim=2)

            # Compute Mean-Pooled Image Features
            img_feats = image_features.mean(dim=1)

            # Joint Features
            enc = torch.cat([embs, img_feats], dim=1)

        else:
            raise AssertionError("Mode %s not defined!" % mode)

        # Detach and Turn to Numpy
        enc = enc.detach().cpu().numpy()
        return enc
