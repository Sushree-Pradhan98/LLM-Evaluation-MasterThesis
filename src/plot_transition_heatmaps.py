import os
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# PATH
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRANSITION_DIR = os.path.join(BASE_DIR, "Result", "LLM_Output", "transition_matrices")
OUTPUT_DIR = os.path.join(BASE_DIR, "Result", "LLM_Output", "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# GET ALL FILES AUTOMATICALLY
# ===============================
FILES = [f for f in os.listdir(TRANSITION_DIR) if f.endswith(".csv")]

print("📂 Found files:", FILES)

# ===============================
# LOOP THROUGH ALL FILES
# ===============================
for file in FILES:

    path = os.path.join(TRANSITION_DIR, file)

    df = pd.read_csv(path, index_col=0)

    plt.figure()

    plt.imshow(df.values)
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            plt.text(j, i, df.values[i, j], ha='center', va='center', fontsize=8)

    plt.xticks(range(len(df.columns)), df.columns, rotation=45)
    plt.yticks(range(len(df.index)), df.index)

    plt.xlabel("Transformed Label")
    plt.ylabel("Original Label")

    # Clean title
    title = file.replace("transition_", "").replace(".csv", "")
    plt.title(f"Heatmap: {title}")

    plt.colorbar()

    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, file.replace(".csv", ".png"))
    plt.savefig(save_path)

    print("✅ Saved:", save_path)

    plt.show()

print("\n🎉 All heatmaps generated!")