import json
import numpy as np
import torch

from torch.utils.data import (
    DataLoader,
    TensorDataset
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from model import LogLSTM


WINDOW_SIZE = 10
PREDICTION_HORIZON = 5


def main():

    X = np.load("X_seq_improved.npy")
    y = np.load("y_seq_improved.npy")

    split_idx = int(len(X) * 0.8)

    X_test = X[split_idx:]
    y_test = y[split_idx:]

    X_test_t = torch.tensor(
        X_test,
        dtype=torch.long
    )

    y_test_t = torch.tensor(
        y_test,
        dtype=torch.long
    )

    test_loader = DataLoader(
        TensorDataset(
            X_test_t,
            y_test_t
        ),
        batch_size=64
    )

    vocab_size = int(X.max()) + 1

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = LogLSTM(vocab_size)

    model.load_state_dict(
        torch.load(
            "best_model.pt",
            map_location=device
        )
    )

    model = model.to(device)

    with open(
        "best_threshold.json",
        "r"
    ) as f:

        threshold_data = json.load(f)

    best_threshold = threshold_data[
        "threshold"
    ]

    model.eval()

    y_true = []
    y_pred = []

    print("\nSEQUENCE FORECASTS")
    print("=" * 60)

    with torch.no_grad():

        for idx, (X_batch, y_batch) in enumerate(test_loader):

            X_batch = X_batch.to(device)

            outputs = model(X_batch)

            probs = torch.softmax(
                outputs,
                dim=1
            )[:, 1].cpu().numpy()

            preds = (
                probs >= best_threshold
            ).astype(int)

            for i in range(len(preds)):

                sequence = X_batch[i].cpu().numpy()

                probability = probs[i] * 100

                prediction = preds[i]

                print(
                    f"\nSequence {idx*64+i+1}"
                )

                print(
                    f"Input Events: "
                    f"{sequence.tolist()}"
                )

                if prediction == 1:

                    print(
                        f"{probability:.2f}% chance "
                        f"of system failure likely "
                        f"in next "
                        f"{PREDICTION_HORIZON} events"
                    )

                else:

                    print(
                        f"{100-probability:.2f}% chance "
                        f"of no system failure "
                        f"in next "
                        f"{PREDICTION_HORIZON} events"
                    )

                y_pred.append(prediction)

                y_true.append(
                    y_batch[i].item()
                )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred
    )

    recall = recall_score(
        y_true,
        y_pred
    )

    f1 = f1_score(
        y_true,
        y_pred
    )

    print("\nFINAL TEST RESULTS")
    print("=" * 60)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")


if __name__ == "__main__":
    main()