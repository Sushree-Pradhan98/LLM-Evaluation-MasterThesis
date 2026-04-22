import os
import pandas as pd
import joblib

# ===============================
# PATH SETUP
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "Result", "LLM_Output")
PRED_DIR = os.path.join(DATA_DIR, "predictions")

MODEL_PATH = os.path.join(BASE_DIR, "Result", "Model", "final_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "Result", "Model", "tfidf_vectorizer.pkl")

os.makedirs(PRED_DIR, exist_ok=True)

# ===============================
# LOAD MODEL
# ===============================
print("🔄 Loading model and vectorizer...")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

print("✅ Model loaded!\n")

# ===============================
# GET FILES
# ===============================
files = [
    f for f in os.listdir(DATA_DIR)
    if f.endswith(".csv") and "train" not in f
]

print(f"📂 Found {len(files)} files\n")

# ===============================
# LOOP
# ===============================
for file in files:

    print(f"➡️ Processing: {file}")

    path = os.path.join(DATA_DIR, file)
    df = pd.read_csv(path, sep=",", quotechar='"', engine="python")
    print("Columns detected:", df.columns.tolist())

    # Normalize column names (VERY IMPORTANT)
    df.columns = df.columns.str.strip().str.lower()

    print("Columns:", list(df.columns))

    # ===============================
    # SELECT TEXT COLUMN
    # ===============================
    if file == "original.csv":
        if "text" in df.columns:
            texts = df["text"].astype(str)
        else:
            print("❌ original.csv missing 'text'")
            continue

    elif "generated_text" in df.columns:
        texts = df["generated_text"].astype(str)

    elif "original_text" in df.columns:
        # fallback (just in case)
        texts = df["original_text"].astype(str)

    else:
        print("⚠️ No usable text column, skipping")
        continue

    # ===============================
    # PREDICT
    # ===============================
    X = vectorizer.transform(texts)
    preds = model.predict(X)

    df["prediction"] = preds

    # ===============================
    # SAVE
    # ===============================
    save_path = os.path.join(PRED_DIR, f"pred_{file}")
    df.to_csv(save_path, index=False)

    print(f"✅ Saved: {save_path}\n")

print("🎉 DONE: All predictions generated!")