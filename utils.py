import numpy as np

from sklearn.metrics import f1_score


def best_f1_threshold(y_true, probs):

    best_f1 = 0.0
    best_t = 0.5

    for t in np.linspace(0.01, 0.99, 500):

        preds = (
            probs >= t
        ).astype(int)

        f1 = f1_score(
            y_true,
            preds,
            zero_division=0
        )

        if f1 > best_f1:

            best_f1 = f1
            best_t = float(t)

    return best_t, best_f1