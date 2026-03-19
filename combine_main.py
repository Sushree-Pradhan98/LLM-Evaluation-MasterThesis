import os
import warnings

import joblib

# ===============================
# SETUP
# ===============================
RESULT_DIR = r"C:\Users\dell\PycharmProjects\LLM-Evaluation-MasterThesis\Result\Score"
os.makedirs(RESULT_DIR, exist_ok=True)

# Disable warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# ===============================
# IMPORTS
# ===============================
import numpy as np
import pandas as pd

from src.load_data import load_dataset
from src.preprocess import preprocess_dataframe
from src.features import extract_tfidf_and_save
from src.deep_features import extract_sbert_and_save
from src.cross_validation import run_kfold
from src.models import get_models
from sklearn.naive_bayes import GaussianNB


# ===============================
# UTILITY FUNCTION
# ===============================
def summarize_results(results):
    summary = {}
    for model, scores in results.items():
        summary[model] = float(np.mean(scores))
    return summary


# ===============================
# STEP 1: LOAD DATA
# ===============================
combined = load_dataset("data/combined.csv")
print("Combined dataset size:", len(combined))


# ===============================
# STEP 2: PREPROCESS
# ===============================
combined = preprocess_dataframe(combined, "combined")

print("Columns after preprocessing:", combined.columns)

if "clean_text" not in combined.columns:
    raise ValueError("❌ clean_text column missing after preprocessing!")

# Reduce dataset for faster testing
#combined = combined.sample(n=5000, random_state=42)

texts = combined["clean_text"].tolist()
labels = combined["label"].tolist()


# ===============================
# STEP 3: TF-IDF FEATURES
# ===============================
X_tfidf = extract_tfidf_and_save(texts)


# ===============================
# STEP 4: SBERT FEATURES
# ===============================
sbert_models = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "paraphrase-MiniLM-L6-v2",
    "distiluse-base-multilingual-cased-v2"
]

sbert_features = {}

for model_name in sbert_models:
    sbert_features[model_name] = extract_sbert_and_save(texts, model_name)

print("\n✅ Feature extraction completed!")


# ===============================
# STEP 5: PREPARE MODELS
# ===============================
y = np.array(labels)
models = get_models()

# SBERT-compatible models (NO MultinomialNB)
models_sbert = {
    "LinearSVC": models["LinearSVC"],
    "RandomForest": models["RandomForest"],
    "LogisticRegression": models["LogisticRegression"],
    "MLP": models["MLP"],
    "GaussianNB": GaussianNB()
}


# ===============================
# STEP 6: TF-IDF K-FOLD
# ===============================
print("\nRunning K-Fold on TF-IDF features...")
tfidf_results = run_kfold(models, X_tfidf, y)

joblib.dump(tfidf_results, os.path.join(RESULT_DIR, "tfidf_fold_results.pkl"))
# ===============================
# STEP 7: SBERT K-FOLD
# ===============================
all_results = {}

for model_name, features in sbert_features.items():

    print(f"\nRunning K-Fold for {model_name}...")

    X = np.array(features)

    results = run_kfold(models_sbert, X, y)

    all_results[model_name] = results
file_name = f"{model_name}_fold_results.pkl"
joblib.dump(results, os.path.join(RESULT_DIR, file_name))

# ===============================
# STEP 8: PRINT RESULTS
# ===============================
print("\n===== TF-IDF RESULTS =====")
tfidf_summary = summarize_results(tfidf_results)
print(tfidf_summary)

for model_name, results in all_results.items():
    print(f"\n===== {model_name} RESULTS =====")
    print(summarize_results(results))


# ===============================
# STEP 9: SAVE ALL RESULTS (ONE FILE)
# ===============================
final_rows = []

# TF-IDF
for model, score in tfidf_summary.items():
    final_rows.append({
        "Feature": "TF-IDF",
        "Model": model,
        "Score": score
    })

# SBERT
for feature_name, results in all_results.items():

    summary = summarize_results(results)

    for model, score in summary.items():
        final_rows.append({
            "Feature": feature_name,
            "Model": model,
            "Score": score
        })

# Save
final_df = pd.DataFrame(final_rows)

final_path = os.path.join(RESULT_DIR, "final_results.csv")
final_df.to_csv(final_path, index=False)

print("\n✅ All results saved to:", final_path)


# ===============================
# STEP 10: BEST MODEL
# ===============================
best = final_df.sort_values(by="Score", ascending=False).iloc[0]

print("\n🔥 BEST CONFIGURATION:")
print(best)
# ===============================
# STEP 11: TRAIN FINAL MODEL
# ===============================
from sklearn.svm import LinearSVC
import joblib

print("\n🚀 Training FINAL model on FULL dataset...")

# Reload FULL dataset (no sampling!)
combined_full = load_dataset("data/combined.csv")
combined_full = preprocess_dataframe(combined_full, "combined_full")

texts_full = combined_full["clean_text"].tolist()
labels_full = combined_full["label"].tolist()

# TF-IDF on FULL data
X_full = extract_tfidf_and_save(texts_full)

y_full = np.array(labels_full)

# Train model
final_model = LinearSVC()
final_model.fit(X_full, y_full)

print("✅ Final model trained!")


# ===============================
# STEP 12: SAVE MODEL
# ===============================
MODEL_DIR = r"C:\Users\dell\PycharmProjects\LLM-Evaluation-MasterThesis\Result\Model"
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, "final_model.pkl")
vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

# Save model
joblib.dump(final_model, model_path)

# Save vectorizer (IMPORTANT)
joblib.dump(
    joblib.load("Result/Features/tfidf_vectorizer.pkl"),
    vectorizer_path
)

print("\n✅ Final model saved to:", model_path)
print("✅ Vectorizer saved to:", vectorizer_path)