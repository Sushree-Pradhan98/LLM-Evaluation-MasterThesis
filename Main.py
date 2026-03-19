# ==========================================
# Emotion Classification Pipeline
# Stage 1: Classifier Selection
# ==========================================

import os
import datetime
import matplotlib.pyplot as plt

from src.load_data import load_dataset
from src.preprocess import preprocess_dataframe
from src.features import create_tfidf_features
from src.models import get_models
from src.evaluate import evaluate_model


# ==========================================
# Result directories
# ==========================================

BASE_RESULT_DIR = r"C:\Users\dell\PycharmProjects\LLM-Evaluation-MasterThesis\Result"

PLOT_DIR = os.path.join(BASE_RESULT_DIR, "Plot")
SCORE_DIR = os.path.join(BASE_RESULT_DIR, "Score")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(SCORE_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# ==========================================
# 1 Load datasets
# ==========================================

print("\nLoading datasets...")

train = load_dataset("data/Raw/training.csv")
val = load_dataset("data/Raw/validation.csv")
test = load_dataset("data/Raw/test.csv")

print("Train size:", len(train))
print("Validation size:", len(val))
print("Test size:", len(test))


# ==========================================
# 2 Preprocess datasets
# ==========================================

train = preprocess_dataframe(train, "training")
val = preprocess_dataframe(val, "validation")
test = preprocess_dataframe(test, "test")


# ==========================================
# 3 Feature extraction for validation
# ==========================================

print("\nCreating TF-IDF features for validation experiment...")

X_train, X_val, vectorizer = create_tfidf_features(
    train["clean_text"],
    val["clean_text"],
    "validation"
)

y_train = train["label"]
y_val = val["label"]


# ==========================================
# 4 Load classifiers
# ==========================================

models = get_models()

results = {}

print("\nTraining and evaluating classifiers...\n")

for name, model in models.items():

    score = evaluate_model(
        model,
        X_train,
        X_val,
        y_train,
        y_val
    )

    results[name] = score

    print(name, "Balanced Accuracy:", round(score, 4))


# ==========================================
# 5 Print comparison
# ==========================================

print("\nValidation Results")
print("------------------")

for model, score in results.items():
    print(f"{model}: {score:.4f}")


# ==========================================
# 6 Select best classifier
# ==========================================

best_model_name = max(results, key=results.get)

print("\nBest Classifier:", best_model_name)
print("Validation Balanced Accuracy:", round(results[best_model_name], 4))


# ==========================================
# 7 Final evaluation on test set
# ==========================================

print("\nEvaluating best model on test set...")

best_model = models[best_model_name]

# create TF-IDF features for test
X_train_full, X_test, vectorizer = create_tfidf_features(
    train["clean_text"],
    test["clean_text"],
    "test"
)

y_train_full = train["label"]
y_test = test["label"]

test_score = evaluate_model(
    best_model,
    X_train_full,
    X_test,
    y_train_full,
    y_test
)

print("Test Balanced Accuracy:", round(test_score, 4))


# ==========================================
# 8 Save score results
# ==========================================

score_file = os.path.join(
    SCORE_DIR,
    f"classifier_scores_{timestamp}.txt"
)

with open(score_file, "w") as f:

    f.write("Validation Results\n")
    f.write("------------------\n\n")

    for model, score in results.items():
        f.write(f"{model}: {score:.4f}\n")

    f.write("\nBest Classifier: " + best_model_name)
    f.write(f"\nValidation Balanced Accuracy: {results[best_model_name]:.4f}")
    f.write(f"\nTest Balanced Accuracy: {test_score:.4f}")

print("\nScores saved to:", score_file)


# ==========================================
# 9 Save comparison plot
# ==========================================

model_names = list(results.keys())
scores = list(results.values())

plt.figure(figsize=(8, 5))

plt.bar(model_names, scores)

plt.ylabel("Balanced Accuracy")
plt.title("Classifier Comparison (Validation Set)")

plt.xticks(rotation=30)

plt.tight_layout()

plot_file = os.path.join(
    PLOT_DIR,
    f"classifier_comparison_{timestamp}.png"
)

plt.savefig(plot_file)

print("Plot saved to:", plot_file)

plt.show()