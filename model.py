import torch
import torch.nn as nn


class FocalLoss(nn.Module):

    def __init__(self, alpha=1, gamma=2):

        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

        self.ce = nn.CrossEntropyLoss(
            reduction="none"
        )

    def forward(self, logits, targets):

        ce_loss = self.ce(logits, targets)

        pt = torch.exp(-ce_loss)

        loss = (
            self.alpha
            * (1 - pt) ** self.gamma
            * ce_loss
        )

        return loss.mean()


class LogLSTM(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden_dim=256,
        dropout=0.3
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim
        )

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            hidden_dim * 2,
            2
        )

    def forward(self, x):

        x = self.embedding(x)

        out, _ = self.lstm(x)

        out = torch.max(
            out,
            dim=1
        ).values

        out = self.dropout(out)

        return self.fc(out)