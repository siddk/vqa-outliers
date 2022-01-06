"""
lstm_cnn.py

Implementation of a basic LSTM + CNN (Deeper LSTM Q + Img Norm as described in https://arxiv.org/abs/1505.00468).

Ref: https://arxiv.org/abs/1505.00468
"""
import numpy as np
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
        _, hidden = self.rnn(x)  # Note that Hidden Defaults to 0

        # If not Bidirectional --> Just Return last Output State
        if self.bidirectional:
            raise NotImplementedError("Bidirectional Not Implemented for LSTM-CNN!")

        # Otherwise, concat forward state for last element and backward state for first element
        else:
            if self.rnn_type == "LSTM":
                # Transpose Hidden States
                h, c = hidden
                h = h.transpose(1, 0).reshape(-1, self.nlayers * self.hidden)

                return h

            else:
                h = hidden
                h = h.transpose(1, 0).reshape(-1, self.nlayers * self.hidden)
                return h


class LSTMCNN(pl.LightningModule):
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
        super(LSTMCNN, self).__init__()

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
            hidden_dim=self.hparams.hidden // 2,
            nlayers=self.hparams.rnn_layers,
            bidirectional=self.hparams.bidirectional,
            dropout=self.hparams.q_dropout,
            rnn=self.hparams.rnn,
        )

        # Build Projection Networks
        if self.hparams.rnn == "LSTM":
            self.q_project = nn.Sequential(
                *[nn.Linear((self.hparams.hidden // 2) * 1 * self.hparams.rnn_layers, self.hparams.hidden), nn.Tanh()]
            )
        elif self.hparams.rnn == "GRU":
            self.q_project = nn.Sequential(
                *[nn.Linear((self.hparams.hidden // 2) * 1 * self.hparams.rnn_layers, self.hparams.hidden), nn.Tanh()]
            )

        self.img_project = nn.Sequential(*[nn.Linear(self.train_dataset.dim, self.hparams.hidden), nn.Tanh()])

        # Build Answer Classifier
        self.ans_classifier = nn.Sequential(
            *[
                nn.Linear(self.hparams.hidden, self.hparams.hidden),
                nn.Tanh(),
                nn.Dropout(p=self.hparams.answer_dropout),
                nn.Linear(self.hparams.hidden, len(self.ans2label)),
            ]
        )

    def forward(self, image_features, question_features):
        # image_features: [bsz, image_dim]
        # question_features: [bsz, seq_len]

        # Embed and Encode Question --> [bsz, q_hidden]
        w_emb = self.w_emb(question_features)
        q_enc = self.q_enc(w_emb)

        # Project Image and Question
        q_repr = self.q_project(q_enc)
        img_repr = self.img_project(nn.functional.normalize(image_features))

        # Merge
        joint_repr = q_repr * img_repr

        # Return Logits
        return self.ans_classifier(joint_repr)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)

    def training_step(self, train_batch, batch_idx):
        img, question, answer, idxs = train_batch

        # Run Forward Pass
        logits = self.forward(img, question)

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
        img, question, answer, idxs = val_batch

        # Run Forward Pass
        logits = self.forward(img, question)
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
                lg = self.forward(img, question)
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
        img, question, _, _ = active_batch

        # Run Forward Pass
        logits = self.forward(img, question)

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
        img, question, _, _ = active_batch

        if mode == "entropy":
            # Accumulate Output Probabilities
            probabilities = []
            for _ in range(k):
                logits = self.forward(img, question)
                probability = torch.softmax(logits, dim=1).detach().cpu().numpy()
                probabilities.append(probability)

            # Compute Entropy of Average...
            entropies = entropy(np.mean(probabilities, axis=0), axis=1)
            return list(entropies)

        elif mode == "bald":
            # Accumulate Output Probabilities (Term 1) and per-run entropies (Term 2)
            probabilities, disagreement = [], []
            for _ in range(k):
                logits = self.forward(img, question)
                probability = torch.softmax(logits, dim=1).detach().cpu().numpy()

                probabilities.append(probability)
                disagreement.append(entropy(probability, axis=1))

            # Compute Entropy of Average
            entropies = entropy(np.mean(probabilities, axis=0), axis=1)
            disagreements = np.mean(disagreement, axis=0)
            return list(entropies - disagreements)

    def extract(self, extract_batch, batch_idx, mode="fused"):
        # Extract a specific (multi/single)-modal representation that the model builds for Coresets.
        img, question, _, _ = extract_batch

        if mode == "language":
            # Embed and Encode Question --> [bsz, q_hidden]
            w_emb = self.w_emb(question)
            enc = self.q_enc(w_emb)

        elif mode == "vision":
            # Get Image Features
            enc = nn.functional.normalize(img)

        elif mode == "fused":
            # Embed and Encode Question --> [bsz, q_hidden]
            w_emb = self.w_emb(question)
            q_enc = self.q_enc(w_emb)

            # Project Image and Question
            q_repr = self.q_project(q_enc)
            img_repr = self.img_project(nn.functional.normalize(img))

            # Merge
            enc = q_repr * img_repr

        else:
            raise AssertionError("Mode %s not defined!" % mode)

        # Detach and Turn to Numpy
        enc = enc.detach().cpu().numpy()
        return enc
