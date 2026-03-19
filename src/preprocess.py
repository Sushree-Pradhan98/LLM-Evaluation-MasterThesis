import re
import os
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

PREPROCESS_DIR = r"C:\Users\dell\PycharmProjects\LLM-Evaluation-MasterThesis\data\Preprocessed"
os.makedirs(PREPROCESS_DIR, exist_ok=True)


def clean_text(text):
    # Handle missing values safely
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    words = text.split()

    # Remove stopwords
    words = [w for w in words if w not in stop_words]

    return " ".join(words)


def preprocess_dataframe(df, dataset_name="dataset"):

    print(f"\nPreprocessing {dataset_name} dataset...")

    # 🔥 IMPORTANT: Ensure correct column name
    if "text" not in df.columns:
        raise ValueError("Column 'text' not found in dataset")

    # Apply cleaning
    df["clean_text"] = df["text"].apply(clean_text)

    # Remove empty rows (VERY IMPORTANT for models)
    df = df[df["clean_text"].str.strip() != ""]

    # Save preprocessed file
    output_file = os.path.join(
        PREPROCESS_DIR,
        f"{dataset_name}_preprocessed.csv"
    )

    df.to_csv(output_file, index=False)

    print("Saved preprocessed file:", output_file)
    print("Columns after preprocessing:", df.columns)

    return df