import os
import pandas as pd

# ===============================
# PATH
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_DIR = os.path.join(BASE_DIR, "Result", "LLM_Output", "predictions")

# ===============================
# LOAD ORIGINAL
# ===============================
orig_path = os.path.join(PRED_DIR, "pred_original.csv")

orig = pd.read_csv(orig_path)

print("Original columns:", orig.columns.tolist())

# Normalize column names
orig.columns = orig.columns.str.strip().str.lower()

# Ensure correct column exists
if "prediction" not in orig.columns:
    raise ValueError("❌ 'prediction' column NOT found in pred_original.csv")

y_orig = orig["prediction"]

results = []

# ===============================
# LOOP FILES
# ===============================
for file in os.listdir(PRED_DIR):

    if file == "pred_original.csv":
        continue

    path = os.path.join(PRED_DIR, file)

    df = pd.read_csv(path)

    print(f"\nProcessing {file}")
    print("Columns:", df.columns.tolist())

    df.columns = df.columns.str.strip().str.lower()

    if "prediction" not in df.columns:
        print("⚠️ Skipping (no prediction column)")
        continue

    y_new = df["prediction"]

    # Compare
    changes = (y_orig != y_new).sum()
    total = len(y_orig)

    change_percent = (changes / total) * 100

    results.append({
        "File": file,
        "Changed_Count": changes,
        "Change_%": round(change_percent, 2)
    })

# ===============================
# SAVE
# ===============================
result_df = pd.DataFrame(results)

output_path = os.path.join(PRED_DIR, "change_summary.csv")
result_df.to_csv(output_path, index=False)

print("\n📊 FINAL RESULTS:")
print(result_df)

print("\n✅ Saved:", output_path)