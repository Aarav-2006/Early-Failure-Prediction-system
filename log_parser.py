from collections import Counter


ERROR_EVENTS = {7, 8, 9}


def parse_event_sequence(event_encoded_str):

    segments = event_encoded_str.split("<|sep|>")

    sequence = []

    for segment in segments:

        segment = segment.strip()

        if not segment:
            continue

        try:

            event_type = int(segment.split()[0])

            sequence.append(event_type)

        except (ValueError, IndexError):

            continue

    return sequence


def parse_event_counts(sequence, n_event_types=10):

    counts = Counter(sequence)

    return {
        i: counts.get(i, 0)
        for i in range(n_event_types)
    }


def parse_dataset(df):

    df = df.copy()

    df["event_sequence"] = df["event_encoded"].apply(
        parse_event_sequence
    )

    df["seq_len"] = df["event_sequence"].apply(len)

    df["error_count"] = df["event_sequence"].apply(
        lambda seq: sum(
            1 for e in seq if e in ERROR_EVENTS
        )
    )

    df["has_errors"] = (
        df["error_count"] > 0
    )

    for i in range(10):

        df[f"count_e{i}"] = df["event_sequence"].apply(
            lambda seq, i=i: seq.count(i)
        )

    return df