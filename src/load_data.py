import pandas as pd


def load_dataset(path):
    """
    Load dataset from CSV file
    """

    df = pd.read_csv(path)

    # Ensure required columns exist
    df = df[['text', 'label']]

    return df