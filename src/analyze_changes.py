import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# ===============================
# PATH SETUP
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PRED_DIR = os.path.join(BASE_DIR, "Result", "LLM_Output", "predictions")

# ===============================
# LOAD ORIGINAL
# ===============================
orig = pd.read_csv(os.path.join(PRED_DIR, "pred_original.csv"))
y_true = orig["predicted_label"]

# ===============================
# RESULT STORAGE
# ===============================
all_results = []

# ===============================
# LOOP THROUGH FILES
# ===============================
for file in os.listdir(PRED_DIR):

    if file == "pred_original.csv":
        continue

    print(f"\n=== {file} ===")

    new = pd.read_csv(os.path.join(PRED_DIR, file))
    y_new = new["predicted_label"]

    # ===============================
    # CONFUSION MATRIX
    # ===============================
    cm = confusion_matrix(y_true, y_new)

    print("Change Matrix:\n", cm)

    # ===============================
    # CALCULATE CHANGE %
    # ===============================
    total = np.sum(cm)
    same = np.trace(cm)

    change_percent = ((total - same) / total) * 100

    print(f"Changed predictions: {change_percent:.2f}%")

    # ===============================
    # STORE RESULTS
    # ===============================
    all_results.append({
        "File": file,
        "Change_Percentage": change_percent
    })

# ===============================
# SAVE SUMMARY
# ===============================
df = pd.DataFrame(all_results)

output_path = os.path.join(PRED_DIR, "change_summary.csv")
df.to_csv(output_path, index=False)

print("\n✅ Summary saved to:", output_path)