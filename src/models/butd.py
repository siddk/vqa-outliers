"""
butd.py

Implementation of the Bottom-Up Top-Down Attention Model, as applied to VQA.

Reference: https://github.com/hengyuan-hu/bottom-up-attention-vqa/blob/master/base_model.py
"""
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from scipy.stats import entropy
from torch.nn.utils.weight_norm import weight_norm


class MLP(nn.Module):
    def __init__(self, dims, use_weight_norm=True):
        """Simple utility class defining a fully connected network (multi-layer perceptron)"""
        super(MLP, self).__init__()

        layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            if use_weight_norm:
                layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            else:
                layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # [bsz, *, dims[0]] --> [bsz, *, dims[-1]]
        return self.mlp(x)


class WordEmbedding(nn.Module):
    def __init__(self, ntoken, dim, dropout=0.0):
        """Initialize an Embedding Matrix with the appropriate dimensions --> Defines padding as last token in dict"""
        super(WordEmbedding, self).__init__()
        self.ntoken, self.dim = ntoken, dim

        self.emb = nn.Embedding(ntoken + 1, dim, padding_idx=ntoken)
        self.dropout = nn.Dropout(dropout)

    def load_embeddings(self, weights):
        """Set Embedding Weights from Numpy Array"""
        assert weights.shape == (self.ntoken, self.dim)
        self.emb.weight.data[: self.ntoken] = torch.from_numpy(weights)

    def forward(self, x):
        # x : [bsz, seq_len] --> [bsz, seq_len, emb_dim]
        return self.dropout(self.emb(x))


class QuestionEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, nlayers=1, bidirectional=False, dropout=0.0, rnn="GRU"):
        """Initialize the RNN Question Encoder with the appropriate configuration"""
        super(QuestionEncoder, self).__init__()
        self.in_dim, self.hidden, self.nlayers, self.bidirectional = in_dim, hidden_dim, nlayers, bidirectional
        self.rnn_type, self.rnn_cls = rnn, nn.GRU if rnn == "GRU" else nn.LSTM

        # Initialize RNN
        self.rnn = self.rnn_cls(
            self.in_dim, self.hidden, self.nlayers, bidirectional=self.bidirectional, dropout=dropout, batch_first=True
        )

    def forward(self, x):
        # x: [bsz, seq_len, emb_dim] --> ([bsz, seq_len, ndirections * hidden], [bsz, nlayers * ndirections, hidden])
        output, hidden = self.rnn(x)  # Note that Hidden Defaults to 0

        # If not Bidirectional --> Just Return last Output State
        if not self.bidirectional:
            # [bsz, hidden]
            return output[:, -1]

        # Otherwise, concat forward state for last element and backward state for first element
        else:
            # [bsz, 2 * hidden]
            f, b = output[:, -1, : self.hidden], output[:, 0, self.hidden :]
            return torch.cat([f, b], dim=1)


class Attention(nn.Module):
    def __init__(self, image_dim, question_dim, hidden, dropout=0.2, use_weight_norm=True):
        """Initialize the Attention Mechanism with the appropriate fusion operation"""
        super(Attention, self).__init__()

        # Attention w/ Product Fusion
        self.image_proj = MLP([image_dim, hidden], use_weight_norm=use_weight_norm)
        self.question_proj = MLP([question_dim, hidden], use_weight_norm=use_weight_norm)
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(hidden, 1), dim=None) if use_weight_norm else nn.Linear(hidden, 1)

    def forward(self, image_features, question_emb):
        # image_features: [bsz, k, image_dim = 2048]
        # question_emb: [bsz, question_dim]

        # Project both image and question embedding to hidden and repeat question_emb
        num_objs = image_features.size(1)
        image_proj = self.image_proj(image_features)
        question_proj = self.question_proj(question_emb).unsqueeze(1).repeat(1, num_objs, 1)

        # Fuse w/ Product
        image_question = image_proj * question_proj

        # Dropout Joint Representation
        joint_representation = self.dropout(image_question)

        # Compute Logits --> Softmax
        logits = self.linear(joint_representation)
        return nn.functional.softmax(logits, dim=1)


class BUTD(pl.LightningModule):
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
        super(BUTD, self).__init__()

        # Save Hyper-Parameters and Dataset
        self.save_hyperparameters(hparams)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        self.ans2label, self.label2ans = ans2label, label2ans
        self.chart, self.chart_val, self.k_dropout = chart, chart_val, k_dropout

        # Build Model
        self.build_model()

    def build_model(self):
        # Build Word Embeddings (for Questions)
        self.w_emb = WordEmbedding(
            ntoken=self.train_dataset.dictionary.ntoken, dim=self.hparams.emb_dim, dropout=self.hparams.emb_dropout
        )

        # Build Question Encoder
        self.q_enc = QuestionEncoder(
            in_dim=self.hparams.emb_dim,
            hidden_dim=self.hparams.hidden,
            nlayers=self.hparams.rnn_layers,
            bidirectional=self.hparams.bidirectional,
            dropout=self.hparams.q_dropout,
            rnn=self.hparams.rnn,
        )

        # Build Attention Mechanism
        self.att = Attention(
            image_dim=self.train_dataset.v_dim + 6,
            question_dim=self.q_enc.hidden,
            hidden=self.hparams.hidden,
            dropout=self.hparams.attention_dropout,
            use_weight_norm=self.hparams.weight_norm,
        )

        # Build Projection Networks
        self.q_project = MLP([self.q_enc.hidden, self.hparams.hidden], use_weight_norm=self.hparams.weight_norm)
        self.img_project = MLP(
            [self.train_dataset.v_dim + 6, self.hparams.hidden], use_weight_norm=self.hparams.weight_norm
        )

        # Build Answer Classifier
        self.ans_classifier = nn.Sequential(
            *[
                weight_norm(nn.Linear(self.hparams.hidden, 2 * self.hparams.hidden), dim=None)
                if self.hparams.weight_norm
                else nn.Linear(self.hparams.hidden, 2 * self.hparams.hidden),
                nn.ReLU(),
                nn.Dropout(self.hparams.answer_dropout),
                weight_norm(nn.Linear(2 * self.hparams.hidden, len(self.ans2label)), dim=None)
                if self.hparams.weight_norm
                else nn.Linear(2 * self.hparams.hidden, len(self.ans2label)),
            ]
        )

    def forward(self, image_features, spatial_features, question_features, attention=False):
        # image_features: [bsz, K, image_dim]
        # question_features: [bsz, seq_len]

        # Embed and Encode Question --> [bsz, q_hidden]
        w_emb = self.w_emb(question_features)
        q_enc = self.q_enc(w_emb)

        # Create new Image Features --> KEY POINT: Concatenate Spatial Features!
        image_features = torch.cat([image_features, spatial_features], dim=2)

        # Attend over Image Features and Create Image Encoding --> [bsz, img_hidden]
        att = self.att(image_features, q_enc)
        img_enc = (image_features * att).sum(dim=1)

        # Project Image and Question Features --> [bsz, hidden]
        q_repr = self.q_project(q_enc)
        img_repr = self.img_project(img_enc)

        # Merge
        joint_repr = q_repr * img_repr

        # Compute and Return Logits
        if not attention:
            return self.ans_classifier(joint_repr)
        else:
            return self.ans_classifier(joint_repr), att

    def configure_optimizers(self):
        if self.hparams.opt == "adamax":
            return torch.optim.Adamax(self.parameters(), weight_decay=self.hparams.weight_decay)
        else:
            raise NotImplementedError("Only Adamax Optimizer is Supported!")

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

        if self.chart or self.chart_val:
            cartography_dict = {}
            for x in outputs:
                cartography_dict.update(x["cartography"])

        pbar = {"train_epoch_loss": avg_loss, "train_epoch_acc": avg_acc}
        if self.chart or self.chart_val:
            log = {"train_epoch_loss": avg_loss, "train_epoch_acc": avg_acc, "cartography": cartography_dict}
        else:
            log = dict(pbar)

        return {"progress_bar": pbar, "log": log}

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

            # MC-Dropout Probabilities!
            self.train()
            probabilities = []
            for _ in range(self.k_dropout):
                lg = self.forward(img, spatials, question)
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
        img, spatials, question, _, _ = active_batch

        # Run Forward Pass
        logits = self.forward(img, spatials, question)

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
        img, spatials, question, _, _ = active_batch

        if mode == "entropy":
            # Accumulate Output Probabilities
            probabilities = []
            for _ in range(k):
                logits = self.forward(img, spatials, question)
                probability = torch.softmax(logits, dim=1).detach().cpu().numpy()
                probabilities.append(probability)

            # Compute Entropy of Average...
            entropies = entropy(np.mean(probabilities, axis=0), axis=1)
            return list(entropies)

        elif mode == "bald":
            # Accumulate Output Probabilities (Term 1) and per-run entropies (Term 2)
            probabilities, disagreement = [], []
            for _ in range(k):
                logits = self.forward(img, spatials, question)
                probability = torch.softmax(logits, dim=1).detach().cpu().numpy()

                probabilities.append(probability)
                disagreement.append(entropy(probability, axis=1))

            # Compute Entropy of Average
            entropies = entropy(np.mean(probabilities, axis=0), axis=1)
            disagreements = np.mean(disagreement, axis=0)
            return list(entropies - disagreements)

    def extract(self, extract_batch, batch_idx, mode="fused"):
        # Extract a specific (multi/single)-modal representation that the model builds for Coresets.
        img, spatials, question, _, _ = extract_batch

        if mode == "language":
            # Get Encoding pre-projection
            w_emb = self.w_emb(question)
            enc = self.q_enc(w_emb)

        elif mode == "vision":
            # Get Image Features concatenated with Spatial Features
            enc = torch.cat([img, spatials], dim=2).mean(dim=1)

        elif mode == "fused":
            # Embed and Encode Question --> [bsz, q_hidden]
            w_emb = self.w_emb(question)
            q_enc = self.q_enc(w_emb)

            # Create new Image Features --> KEY POINT: Concatenate Spatial Features!
            image_features = torch.cat([img, spatials], dim=2)

            # Attend over Image Features and Create Image Encoding --> [bsz, img_hidden]
            att = self.att(image_features, q_enc)
            img_enc = (image_features * att).sum(dim=1)

            # Project Image and Question Features --> [bsz, hidden]
            q_repr = self.q_project(q_enc)
            img_repr = self.img_project(img_enc)

            # Merge
            enc = q_repr * img_repr

        else:
            raise AssertionError("Mode %s not defined!" % mode)

        # Detach and Turn to Numpy
        enc = enc.detach().cpu().numpy()
        return enc
