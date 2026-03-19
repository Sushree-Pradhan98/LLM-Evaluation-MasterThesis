import os
import joblib
import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_rel, wilcoxon

# ===============================
# PATH SETUP
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "Result", "Score")

# ===============================
# LOAD DATA
# ===============================
tfidf_path = os.path.join(RESULT_DIR, "tfidf_fold_results.pkl")

# 👉 Use the SBERT file that exists in your folder
sbert_path = os.path.join(
    RESULT_DIR,
    "distiluse-base-multilingual-cased-v2_fold_results.pkl"
)

tfidf = joblib.load(tfidf_path)
sbert = joblib.load(sbert_path)

model_name = "LinearSVC"

tfidf_scores = np.array(tfidf[model_name])
sbert_scores = np.array(sbert[model_name])

print("\nTF-IDF Scores:", tfidf_scores)
print("SBERT Scores:", sbert_scores)


# ===============================
# STEP 1: SHAPIRO TEST
# ===============================
print("\n===== SHAPIRO-WILK TEST =====")

stat1, p1 = shapiro(tfidf_scores)
stat2, p2 = shapiro(sbert_scores)

print(f"TF-IDF p-value: {p1:.4f}")
print(f"SBERT p-value: {p2:.4f}")

if p1 > 0.05 and p2 > 0.05:
    normal = True
    normal_msg = "Normal distribution"
else:
    normal = False
    normal_msg = "Not normal distribution"


# ===============================
# STEP 2: STATISTICAL TEST
# ===============================
print("\n===== STATISTICAL TEST =====")

if normal:
    stat, p_value = ttest_rel(tfidf_scores, sbert_scores)
    test_name = "Paired T-test"
else:
    stat, p_value = wilcoxon(tfidf_scores, sbert_scores)
    test_name = "Wilcoxon Test"

print(f"{test_name} statistic: {stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    interpretation = "Significant difference"
else:
    interpretation = "No significant difference"


# ===============================
# STEP 3: SAVE RESULTS
# ===============================
results_data = [
    {
        "Test": "Shapiro (TF-IDF)",
        "Model_Comparison": "TF-IDF vs SBERT",
        "Statistic": stat1,
        "P_value": p1,
        "Interpretation": "Normal" if p1 > 0.05 else "Not Normal"
    },
    {
        "Test": "Shapiro (SBERT)",
        "Model_Comparison": "TF-IDF vs SBERT",
        "Statistic": stat2,
        "P_value": p2,
        "Interpretation": "Normal" if p2 > 0.05 else "Not Normal"
    },
    {
        "Test": test_name,
        "Model_Comparison": "TF-IDF vs SBERT",
        "Statistic": stat,
        "P_value": p_value,
        "Interpretation": interpretation
    }
]

df = pd.DataFrame(results_data)

output_path = os.path.join(RESULT_DIR, "statistical_results.csv")
df.to_csv(output_path, index=False)

print("\n✅ Statistical results saved to:", output_path)