import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


FEATURE_DIR = r"C:\Users\dell\PycharmProjects\LLM-Evaluation-MasterThesis\Result\Features"
os.makedirs(FEATURE_DIR, exist_ok=True)


def extract_tfidf_and_save(texts):

    print("\nExtracting TF-IDF features...")

    vectorizer = TfidfVectorizer(
        max_features=25000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        sublinear_tf=True,
        norm='l2'
    )

    X = vectorizer.fit_transform(texts)

    # Save features
    feature_path = os.path.join(FEATURE_DIR, "tfidf_features.pkl")
    joblib.dump(X, feature_path)

    # Save vectorizer
    vectorizer_path = os.path.join(FEATURE_DIR, "tfidf_vectorizer.pkl")
    joblib.dump(vectorizer, vectorizer_path)

    print("Saved TF-IDF features:", feature_path)

    return X