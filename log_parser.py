from datasets import load_dataset
import pandas as pd
import numpy as np
from collections import Counter

 
def load_data():

    print("Loading dataset from Hugging Face...")
    print("(This may take a moment on first run — it will cache locally after)\n")
 
    ds = load_dataset("honicky/hdfs-logs-encoded-blocks")
 
    train_df = ds["train"].to_pandas()
    val_df   = ds["validation"].to_pandas()
    test_df  = ds["test"].to_pandas()
 
    print("✓ Dataset loaded successfully!")
    print(f"  Train size      : {len(train_df):,} blocks")
    print(f"  Validation size : {len(val_df):,} blocks")
    print(f"  Test size       : {len(test_df):,} blocks")
    print(f"  Total           : {len(train_df) + len(val_df) + len(test_df):,} blocks")
 
    return train_df, val_df, test_df
 
 
def explore_data(df, split_name="train"):
    """
    Print a summary of the dataset: columns, dtypes, class balance,
    and a sample row so you understand the structure.
    """
    print(f"\n{'='*55}")
    print(f"  Dataset Overview — {split_name.upper()} split")
    print(f"{'='*55}")
 
    # Columns and types
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
 
    # Class distribution
    print(f"\nLabel distribution:")
    counts = df["label"].value_counts()
    ratios = df["label"].value_counts(normalize=True)
    for label in counts.index:
        print(f"  {label:8s} : {counts[label]:>6,}  ({ratios[label]:.2%})")
 
    # Sample row
    print(f"\nSample row (block_id): {df['block_id'].iloc[0]}")
    print(f"Sample label         : {df['label'].iloc[0]}")
    print(f"\nSample event_encoded (truncated):")
    print(f"  {df['event_encoded'].iloc[0][:120]}...")
    print(f"\nSample tokenized_block (first 10 tokens):")
    print(f"  {df['tokenized_block'].iloc[0][:10]}")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# PART 2: LOG PARSER
# ══════════════════════════════════════════════════════════════════════════════
 
# Event type reference — what each integer means in the HDFS logs
EVENT_TYPE_MAP = {
    0: "PacketResponder start",      # Normal data transfer start
    1: "Received block",             # Block successfully received
    2: "Transmitted block",          # Block successfully sent
    3: "Replicating block",          # Block replication initiated
    4: "Served block",               # Block served to client
    5: "PacketResponder termination",# Normal termination
    6: "Receiving block",            # Incoming block transfer
    7: "Exception in pipeline",      # ⚠ Error — pipeline write failed
    8: "Exception from target",      # ⚠ Error — connection to target failed
    9: "Unexpected error",           # ⚠ Error — unexpected exception
}
 
ERROR_EVENTS = {7, 8, 9}   # these are the anomaly-indicating events
 
 
def parse_event_sequence(event_encoded_str):
    """
    Parse one block's event_encoded string into a list of event type integers.
 
    The format is:
        "<|sep|>EVENT_TYPE rest_of_log<|sep|>EVENT_TYPE rest_of_log..."
 
    Example input:
        "<|sep|>0 /10.1.1.1:50010<|sep|>6 <|sep|>7 10.1.1.1 /10.1.1.2<|sep|>..."
 
    Example output:
        [0, 6, 7, 1, 5, 4]
 
    Parameters:
        event_encoded_str (str): raw value from the event_encoded column
 
    Returns:
        list[int]: ordered list of event type integers for this block
    """
    segments = event_encoded_str.split("<|sep|>")
    event_sequence = []
 
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        try:
            # The event type is always the first token (integer) in the segment
            event_type = int(segment.split()[0])
            event_sequence.append(event_type)
        except (ValueError, IndexError):
            # Skip malformed segments
            continue
 
    return event_sequence
 
 
def parse_event_counts(event_encoded_str, n_event_types=10):
    """
    Parse one block and return a count of each event type.
 
    Example output:
        {0: 3, 1: 1, 2: 3, 3: 3, 4: 3, 5: 3, 6: 1, 7: 2, 8: 1, 9: 0}
 
    Parameters:
        event_encoded_str (str): raw value from the event_encoded column
        n_event_types (int)    : total number of event types (0–9)
 
    Returns:
        dict[int, int]: count of each event type
    """
    sequence = parse_event_sequence(event_encoded_str)
    counts = Counter(sequence)
    # Ensure all event types 0–9 are present even if count is 0
    return {i: counts.get(i, 0) for i in range(n_event_types)}
 
 
def parse_block_summary(event_encoded_str):
    """
    Return a full human-readable summary of a block's log sequence.
    Useful for understanding what a block looks like before/during an anomaly.
 
    Parameters:
        event_encoded_str (str): raw value from the event_encoded column
 
    Returns:
        dict with keys: sequence, counts, seq_len, error_count, has_errors
    """
    sequence = parse_event_sequence(event_encoded_str)
    counts   = parse_event_counts(event_encoded_str)
    error_count = sum(counts[e] for e in ERROR_EVENTS)
 
    return {
        "sequence"    : sequence,
        "seq_len"     : len(sequence),
        "counts"      : counts,
        "error_count" : error_count,
        "has_errors"  : error_count > 0,
        "event_names" : [EVENT_TYPE_MAP.get(e, f"Unknown({e})") for e in sequence],
    }
 
 
def parse_dataset(df):
    """
    Apply the parser to an entire DataFrame.
    Adds columns: event_sequence, seq_len, error_count, has_errors,
    and one count column per event type (count_e0 ... count_e9).
 
    Parameters:
        df (pd.DataFrame): train/val/test DataFrame
 
    Returns:
        pd.DataFrame: original df with new parsed columns added
    """
    print(f"Parsing {len(df):,} blocks...", end=" ")
 
    df = df.copy()
 
    # Parse sequence and summary for each block
    summaries = df["event_encoded"].apply(parse_block_summary)
 
    df["event_sequence"] = summaries.apply(lambda s: s["sequence"])
    df["seq_len"]        = summaries.apply(lambda s: s["seq_len"])
    df["error_count"]    = summaries.apply(lambda s: s["error_count"])
    df["has_errors"]     = summaries.apply(lambda s: s["has_errors"])
 
    # Add individual event type count columns (useful for Random Forest)
    for i in range(10):
        df[f"count_e{i}"] = summaries.apply(lambda s, i=i: s["counts"][i])
 
    # Binary label column (1 = Anomaly, 0 = Normal)
    df["label_int"] = (df["label"] == "Anomaly").astype(int)
 
    print("✓ Done!")
    return df
 
 
def show_parsed_examples(df, n=2):
    """
    Print parsed examples — one Normal and one Anomaly block — 
    so you can visually verify the parser is working correctly.
    """
    for label in ["Normal", "Anomaly"]:
        sample = df[df["label"] == label].iloc[0]
        summary = parse_block_summary(sample["event_encoded"])
 
        print(f"\n{'─'*50}")
        print(f"Block ID : {sample['block_id']}")
        print(f"Label    : {sample['label']}")
        print(f"Seq len  : {summary['seq_len']}")
        print(f"Sequence : {summary['sequence']}")
        print(f"Names    : {summary['event_names']}")
        print(f"Counts   : {summary['counts']}")
        print(f"Errors   : {summary['error_count']} "
              f"(event types 7, 8, 9)")
 
 
if __name__ == "__main__":
 
    # 1. Load data
    train_df, val_df, test_df = load_data()
 
    # 2. Explore the raw data
    explore_data(train_df, split_name="train")
 
    # 3. Parse all splits
    print("\nParsing datasets...")
    train_df = parse_dataset(train_df)
    val_df   = parse_dataset(val_df)
    test_df  = parse_dataset(test_df)
 
    # 4. Show parsed examples (Normal vs Anomaly)
    print("\nParsed examples:")
    show_parsed_examples(train_df)

    print(f"\n{'='*55}")
    print("  Post-Parsing Stats (train split)")
    print(f"{'='*55}")
    print(f"Avg sequence length  : {train_df['seq_len'].mean():.1f}")
    print(f"Max sequence length  : {train_df['seq_len'].max()}")
    print(f"Min sequence length  : {train_df['seq_len'].min()}")
    print(f"\nBlocks with errors   : {train_df['has_errors'].sum():,} "
          f"({train_df['has_errors'].mean():.2%})")
    print(f"Actual anomaly count : {train_df['label_int'].sum():,} "
          f"({train_df['label_int'].mean():.2%})")
 
    print(f"\nAvg error events in Normal blocks  : "
          f"{train_df[train_df['label']=='Normal']['error_count'].mean():.3f}")
    print(f"Avg error events in Anomaly blocks : "
          f"{train_df[train_df['label']=='Anomaly']['error_count'].mean():.3f}")
 
    print(f"\nNew columns added by parser:")
    new_cols = ["event_sequence", "seq_len", "error_count", "has_errors",
                "label_int"] + [f"count_e{i}" for i in range(10)]
    print(train_df[new_cols].head(3).to_string())
 
    print("\n✓ Data loading and parsing complete!")
    print("  train_df, val_df, test_df are ready for the next step (feature engineering + model training).")