from datasets import load_dataset


def load_data():

    print("Loading dataset...")

    ds = load_dataset("honicky/hdfs-logs-encoded-blocks")

    train_df = ds["train"].to_pandas()
    val_df = ds["validation"].to_pandas()
    test_df = ds["test"].to_pandas()

    print("Dataset loaded successfully.")

    return train_df, val_df, test_df