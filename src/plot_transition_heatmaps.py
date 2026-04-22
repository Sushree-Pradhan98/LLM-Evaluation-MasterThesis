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
# SELECT IMPORTANT FILES
# ===============================
# You can change these if needed
TARGET_FILES = [
    "transition_gemma_p1.csv",
    "transition_ollama_p3.csv",
    "transition_phi3_p3.csv"
]

# ===============================
# LOOP
# ===============================
for file in TARGET_FILES:

    path = os.path.join(TRANSITION_DIR, file)

    if not os.path.exists(path):
        print(f"⚠️ Missing: {file}")
        continue

    df = pd.read_csv(path, index_col=0)

    plt.figure()

    plt.imshow(df.values)

    plt.xticks(range(len(df.columns)), df.columns, rotation=45)
    plt.yticks(range(len(df.index)), df.index)

    plt.xlabel("Transformed Label")
    plt.ylabel("Original Label")
    plt.title(f"Transition Heatmap: {file.replace('.csv','')}")

    plt.colorbar()

    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, file.replace(".csv", ".png"))
    plt.savefig(save_path)

    print("✅ Saved:", save_path)

    plt.show()