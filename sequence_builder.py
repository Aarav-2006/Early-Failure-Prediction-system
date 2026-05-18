import numpy as np

from tensorflow.keras.preprocessing.sequence import (
    pad_sequences
)


ERROR_EVENTS = {7, 8, 9}


def contains_failure(sequence):

    return int(
        any(event in ERROR_EVENTS for event in sequence)
    )


def build_prediction_sequences(
    df,
    window_size=10,
    prediction_horizon=5
):

    X = []
    y = []

    all_sequences = df["event_sequence"].tolist()

    for sequence in all_sequences:

        if len(sequence) < (
            window_size + prediction_horizon
        ):
            continue

        for i in range(
            len(sequence)
            - window_size
            - prediction_horizon
        ):

            input_window = sequence[
                i : i + window_size
            ]

            future_window = sequence[
                i + window_size :
                i + window_size + prediction_horizon
            ]

            label = contains_failure(
                future_window
            )

            X.append(input_window)

            y.append(label)

    X = pad_sequences(
        X,
        maxlen=window_size,
        padding="post"
    )

    y = np.array(y)

    np.save("X_seq_improved.npy", X)
    np.save("y_seq_improved.npy", y)

    return X, y