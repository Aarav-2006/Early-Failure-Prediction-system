from __future__ import annotations

import json
import os
import pandas as pd
import numpy as np
import torch

from torch.utils.data import (
    DataLoader,
    TensorDataset
)

from sklearn.model_selection import (
    train_test_split
)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

from dataset_loader import load_data
from log_parser import parse_dataset

from sequence_builder import (
    build_prediction_sequences
)

from model import (
    LogLSTM,
    FocalLoss
)

from utils import (
    best_f1_threshold
)


WINDOW_SIZE = 10
PREDICTION_HORIZON = 5


def main():

    if (
        os.path.exists("X_seq_improved.npy")
        and
        os.path.exists("y_seq_improved.npy")
    ):

        print("Loading cached sequences...")

        X = np.load("X_seq_improved.npy")
        y = np.load("y_seq_improved.npy")

    else:

        print("Generating sequences...")

        train_df, val_df, test_df = load_data()

        train_df = parse_dataset(train_df)
        val_df = parse_dataset(val_df)
        test_df = parse_dataset(test_df)

        combined_df = pd.concat([
            train_df,
            val_df,
            test_df
        ])

        X, y = build_prediction_sequences(
            combined_df,
            window_size=WINDOW_SIZE,
            prediction_horizon=PREDICTION_HORIZON
        )

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.2,
        stratify=y_temp,
        random_state=42
    )

    X_train_t = torch.tensor(X_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.long)

    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=64,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=64
    )

    vocab_size = int(X.max()) + 1

    model = LogLSTM(vocab_size)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)

    criterion = FocalLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=5e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    best_f1 = 0.0
    best_threshold = 0.5

    patience = 10
    counter = 0

    epochs = 50

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        model.eval()

        all_probs = []
        y_true_list = []

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                X_batch = X_batch.to(device)

                outputs = model(X_batch)

                probs = torch.softmax(
                    outputs,
                    dim=1
                )[:, 1].cpu().numpy()

                all_probs.extend(probs.tolist())

                y_true_list.extend(
                    y_batch.numpy().tolist()
                )

        thr, epoch_f1 = best_f1_threshold(
            np.array(y_true_list),
            np.array(all_probs)
        )

        scheduler.step(epoch_f1)

        print(
            f"Epoch {epoch+1} | "
            f"Loss: {total_loss:.2f} | "
            f"Val F1: {epoch_f1:.4f}"
        )

        if epoch_f1 > best_f1:

            best_f1 = epoch_f1
            best_threshold = thr

            counter = 0

            torch.save(
                model.state_dict(),
                "best_model.pt"
            )

            with open(
                "best_threshold.json",
                "w"
            ) as f:

                json.dump(
                    {
                        "threshold": best_threshold,
                        "val_f1": best_f1
                    },
                    f,
                    indent=2
                )

        else:

            counter += 1

            if counter >= patience:

                print("Early stopping.")

                break

    preds = (
        np.array(all_probs)
        >= best_threshold
    ).astype(int)

    precision = precision_score(
        y_true_list,
        preds
    )

    recall = recall_score(
        y_true_list,
        preds
    )

    f1 = f1_score(
        y_true_list,
        preds
    )

    print("\nFINAL TRAINING RESULTS")
    print("=" * 50)

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print(
        f"\nBest Threshold: "
        f"{best_threshold:.4f}"
    )


if __name__ == "__main__":
    main()