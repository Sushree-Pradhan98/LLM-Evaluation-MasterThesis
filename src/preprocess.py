import re
import os
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

PREPROCESS_DIR = r"C:\Users\dell\PycharmProjects\LLM-Evaluation-MasterThesis\data\Preprocessed"

os.makedirs(PREPROCESS_DIR, exist_ok=True)


def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^\w\s]", "", text)

    words = text.split()

    words = [w for w in words if w not in stop_words]

    cleaned_text = " ".join(words)

    return cleaned_text


def preprocess_dataframe(df, dataset_name="dataset"):

    print(f"\nPreprocessing {dataset_name} dataset...")

    df["clean_text"] = df["text"].apply(clean_text)

    output_file = os.path.join(
        PREPROCESS_DIR,
        f"{dataset_name}_preprocessed.csv"
    )

    df.to_csv(output_file, index=False)

    print("Saved preprocessed file:", output_file)

    return df