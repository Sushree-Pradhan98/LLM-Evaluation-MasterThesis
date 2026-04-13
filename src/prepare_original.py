import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ===============================
# PATH SETUP
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(BASE_DIR, "data", "combined.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "Result", "LLM_Output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Emotion label mapping for readable output
LABEL_NAMES = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}

SAMPLE_SIZE = 1000
RANDOM_STATE = 42


def load_full_dataset(path):
    """Load the full combined emotion dataset."""
    df = pd.read_csv(path)
    df = df[["text", "label"]].dropna().reset_index(drop=True)
    return df


def stratified_train_test_split(df):
    """
    Split dataset into train (~80%) and test (~20%) with stratification.

    Returns:
        train_df, test_df
    """
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, test_df


def stratified_sample(test_df, n=SAMPLE_SIZE):
    """
    Take a proportional stratified sample of n rows from the test set.

    Each label gets floor(n * label_proportion) samples. Remaining slots
    are distributed one-by-one to the largest-remainder labels so the
    total is exactly n.

    Returns:
        DataFrame with exactly n rows, proportional emotion distribution.
    """
    label_counts = test_df["label"].value_counts()
    total = len(test_df)

    # Calculate proportional counts per label
    exact_proportions = {
        label: n * (count / total) for label, count in label_counts.items()
    }
    floor_counts = {label: int(np.floor(v)) for label, v in exact_proportions.items()}
    remainders = {
        label: exact_proportions[label] - floor_counts[label]
        for label in exact_proportions
    }

    # Distribute leftover slots to labels with largest remainders
    leftover = n - sum(floor_counts.values())
    sorted_by_remainder = sorted(remainders, key=remainders.get, reverse=True)
    for i in range(leftover):
        floor_counts[sorted_by_remainder[i]] += 1

    # Sample from each label group
    sampled_parts = []
    for label, count in floor_counts.items():
        group = test_df[test_df["label"] == label]
        sampled = group.sample(n=count, random_state=RANDOM_STATE)
        sampled_parts.append(sampled)

    result = pd.concat(sampled_parts, ignore_index=True)

    # Shuffle the final result
    result = result.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return result


def main():
    """Main entry point."""
    # ---- Step 1: Load full dataset ----
    print("Loading full dataset...")
    df = load_full_dataset(INPUT_FILE)
    print(f"  Full dataset size: {len(df)}")
    print(f"  Label distribution:\n{df['label'].value_counts().sort_index()}\n")

    # ---- Step 2: Stratified train/test split ----
    print("Performing stratified train/test split (80/20)...")
    train_df, test_df = stratified_train_test_split(df)
    print(f"  Train size: {len(train_df)}")
    print(f"  Test size:  {len(test_df)}\n")

    # ---- Step 3: Sample 1000 from test set ----
    print(f"Sampling {SAMPLE_SIZE} rows from test set (stratified)...")
    original_df = stratified_sample(test_df, n=SAMPLE_SIZE)
    print(f"  Sampled size: {len(original_df)}")

    print("\n  Label distribution in original.csv:")
    dist = original_df["label"].value_counts().sort_index()
    for label, count in dist.items():
        name = LABEL_NAMES.get(label, f"label_{label}")
        pct = count / len(original_df) * 100
        print(f"    {label} ({name}): {count}  ({pct:.1f}%)")

    # ---- Step 4: Save outputs ----
    original_path = os.path.join(OUTPUT_DIR, "original.csv")
    original_df.to_csv(original_path, index=False)
    print(f"\n✅ Saved original.csv ({len(original_df)} rows): {original_path}")

    train_path = os.path.join(OUTPUT_DIR, "train_split.csv")
    train_df.to_csv(train_path, index=False)
    print(f"✅ Saved train_split.csv ({len(train_df)} rows): {train_path}")


if __name__ == "__main__":
    main()