import os
import pandas as pd

# ===============================
# PATH SETUP
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRED_DIR = os.path.join(BASE_DIR, "Result", "LLM_Output", "predictions")
OUTPUT_DIR = os.path.join(BASE_DIR, "Result", "LLM_Output", "transition_matrices")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# LABEL MAPPING
# ===============================
LABEL_NAMES = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}

def format_label(x):
    return f"{x} ({LABEL_NAMES.get(x, 'unknown')})"

# ===============================
# LOAD ORIGINAL
# ===============================
orig_path = os.path.join(PRED_DIR, "pred_original.csv")
orig = pd.read_csv(orig_path)

# Normalize column names
orig.columns = orig.columns.str.strip().str.lower()

if "prediction" not in orig.columns:
    raise ValueError("❌ 'prediction' column missing in original file")

y_orig = orig["prediction"]

print("✅ Loaded original predictions")

# ===============================
# LOOP THROUGH FILES
# ===============================
for file in os.listdir(PRED_DIR):

    # Only process prediction files
    if not file.startswith("pred_") or file == "pred_original.csv":
        continue

    file_path = os.path.join(PRED_DIR, file)
    df = pd.read_csv(file_path)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    if "prediction" not in df.columns:
        print(f"⚠️ Skipping {file} (no prediction column)")
        continue

    y_new = df["prediction"]

    print(f"\n📊 Processing: {file}")

    # ===============================
    # CREATE TRANSITION MATRIX
    # ===============================
    transition = pd.crosstab(y_orig, y_new)

    # Sort rows and columns
    transition = transition.sort_index().sort_index(axis=1)

    # ===============================
    # ADD LABEL NAMES (NUMBER + TEXT)
    # ===============================
    transition.index = transition.index.map(format_label)
    transition.columns = transition.columns.map(format_label)

    # Axis labels
    transition.index.name = "Original Label"
    transition.columns.name = "Transformed Label"

    print(transition)

    # ===============================
    # SAVE MATRIX
    # ===============================
    save_name = file.replace("pred_", "transition_")
    save_path = os.path.join(OUTPUT_DIR, save_name)

    transition.to_csv(save_path)

    print(f"✅ Saved: {save_path}")

print("\n🎉 All transition matrices saved successfully!")