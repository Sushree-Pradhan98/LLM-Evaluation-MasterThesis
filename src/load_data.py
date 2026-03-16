import pandas as pd


def load_dataset(path):
    """
    Load emotion dataset from CSV file
    """

    df = pd.read_csv(path)

    # Ensure correct column names
    df = df[['text', 'label']]

    return df