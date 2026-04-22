import os
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# PATH
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_DIR = os.path.join(BASE_DIR, "Result", "LLM_Output", "predictions")

INPUT_FILE = os.path.join(PRED_DIR, "change_summary.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "Result", "LLM_Output", "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(INPUT_FILE)

# Clean names for display
df["Label"] = df["File"].str.replace("pred_", "").str.replace(".csv", "")

# ===============================
# PLOT
# ===============================
plt.figure()

plt.bar(df["Label"], df["Change_%"])

plt.xticks(rotation=45)
plt.xlabel("LLM + Prompt")
plt.ylabel("Change Percentage (%)")
plt.title("Effect of LLM Transformations on Prediction Changes")

plt.tight_layout()

# ===============================
# SAVE
# ===============================
save_path = os.path.join(OUTPUT_DIR, "change_bar_chart.png")
plt.savefig(save_path)

print("✅ Saved:", save_path)
plt.show()