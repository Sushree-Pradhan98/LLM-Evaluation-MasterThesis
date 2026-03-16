import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# Directory to store extracted features
FEATURE_DIR = r"C:\Users\dell\PycharmProjects\LLM-Evaluation-MasterThesis\Result\Features"

# Create folder if it doesn't exist
os.makedirs(FEATURE_DIR, exist_ok=True)


def create_tfidf_features(train_texts, eval_texts, dataset_name="dataset"):

    print("\nStarting TF-IDF feature extraction...")

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True
    )

    # Fit on training data
    X_train = vectorizer.fit_transform(train_texts)

    # Transform evaluation data
    X_eval = vectorizer.transform(eval_texts)

    print("Feature matrix shape (train):", X_train.shape)
    print("Feature matrix shape (eval):", X_eval.shape)

    # Convert sparse matrices to dataframe
    train_df = pd.DataFrame(X_train.toarray())
    eval_df = pd.DataFrame(X_eval.toarray())

    # Define file paths
    train_file = os.path.join(
        FEATURE_DIR,
        f"{dataset_name}_train_features.csv"
    )

    eval_file = os.path.join(
        FEATURE_DIR,
        f"{dataset_name}_eval_features.csv"
    )

    vocab_file = os.path.join(
        FEATURE_DIR,
        f"{dataset_name}_feature_names.csv"
    )

    # Save feature matrices
    print("\nSaving feature matrices...")

    train_df.to_csv(train_file, index=False)
    eval_df.to_csv(eval_file, index=False)

    print("Saved train features to:", train_file)
    print("Saved eval features to:", eval_file)

    # Save vocabulary
    feature_names = vectorizer.get_feature_names_out()

    vocab_df = pd.DataFrame(feature_names, columns=["feature"])

    vocab_df.to_csv(vocab_file, index=False)

    print("Saved vocabulary to:", vocab_file)

    return X_train, X_eval, vectorizer