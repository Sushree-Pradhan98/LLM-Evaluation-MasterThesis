import os
import pandas as pd

# ===============================
# PATH
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_DIR = os.path.join(BASE_DIR, "Result", "LLM_Output", "predictions")

# ===============================
# LABELS
# ===============================
SADNESS = 0
JOY = 1
LOVE = 2
ANGER = 3
FEAR = 4

# ===============================
# LOAD ORIGINAL
# ===============================
orig = pd.read_csv(os.path.join(PRED_DIR, "pred_original.csv"))
orig.columns = orig.columns.str.strip().str.lower()

y_orig = orig["prediction"]

results = []

# ===============================
# LOOP FILES
# ===============================
for file in os.listdir(PRED_DIR):

    if not file.startswith("pred_") or file == "pred_original.csv":
        continue

    df = pd.read_csv(os.path.join(PRED_DIR, file))
    df.columns = df.columns.str.strip().str.lower()

    if "prediction" not in df.columns:
        continue

    y_new = df["prediction"]

    # ----------------------------
    # TOTAL COUNTS
    # ----------------------------
    total_sad = (y_orig == SADNESS).sum()
    total_ang = (y_orig == ANGER).sum()
    total_fear = (y_orig == FEAR).sum()

    # ----------------------------
    # TRANSITIONS
    # ----------------------------
    sad_to_joy = ((y_orig == SADNESS) & (y_new == JOY)).sum()
    sad_to_love = ((y_orig == SADNESS) & (y_new == LOVE)).sum()

    ang_to_joy = ((y_orig == ANGER) & (y_new == JOY)).sum()
    ang_to_love = ((y_orig == ANGER) & (y_new == LOVE)).sum()

    fear_to_joy = ((y_orig == FEAR) & (y_new == JOY)).sum()
    fear_to_love = ((y_orig == FEAR) & (y_new == LOVE)).sum()

    # ----------------------------
    # PERCENTAGES
    # ----------------------------
    results.append({
        "File": file,

        "sad→joy_%": round((sad_to_joy / total_sad) * 100, 2) if total_sad else 0,
        "sad→love_%": round((sad_to_love / total_sad) * 100, 2) if total_sad else 0,

        "anger→joy_%": round((ang_to_joy / total_ang) * 100, 2) if total_ang else 0,
        "anger→love_%": round((ang_to_love / total_ang) * 100, 2) if total_ang else 0,

        "fear→joy_%": round((fear_to_joy / total_fear) * 100, 2) if total_fear else 0,
        "fear→love_%": round((fear_to_love / total_fear) * 100, 2) if total_fear else 0,
    })

# ===============================
# SAVE
# ===============================
result_df = pd.DataFrame(results)

output_path = os.path.join(PRED_DIR, "transition_analysis.csv")
result_df.to_csv(output_path, index=False)

print(result_df)
print("\nSaved:", output_path)