import os
import pandas as pd
import joblib

# ===============================
# PATH SETUP
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

INPUT_DIR = os.path.join(BASE_DIR, "Result", "LLM_Output")
OUTPUT_DIR = os.path.join(BASE_DIR, "Result", "LLM_Output", "predictions")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# LOAD MODEL + VECTORIZER
# ===============================
model = joblib.load(os.path.join(BASE_DIR, "Result", "Model", "final_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "Result", "Model", "tfidf_vectorizer.pkl"))

print("✅ Model and vectorizer loaded")


# ===============================
# CLASSIFY ALL FILES
# ===============================
files = os.listdir(INPUT_DIR)

for file in files:

    if not file.endswith(".csv"):
        continue

    print(f"\nProcessing: {file}")

    file_path = os.path.join(INPUT_DIR, file)
    df = pd.read_csv(file_path)

    # Convert text → features
    X = vectorizer.transform(df["text"])

    # Predict
    preds = model.predict(X)

    # Save predictions
    df["predicted_label"] = preds

    save_path = os.path.join(OUTPUT_DIR, f"pred_{file}")
    df.to_csv(save_path, index=False)

    print("Saved:", save_path)