# ==============================
# Main Pipeline for Emotion Classification
# ==============================
import os
import datetime
from src.load_data import load_dataset
from src.preprocess import preprocess_dataframe

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

import matplotlib.pyplot as plt


# ==============================
# 1. Load datasets
# ==============================

print("Loading datasets...")

train = load_dataset("data/training.csv")
test = load_dataset("data/test.csv")

print("Train size:", len(train))
print("Test size:", len(test))


# ==============================
# 2. Preprocess text
# ==============================

print("\nPreprocessing text...")

train = preprocess_dataframe(train)
test = preprocess_dataframe(test)


# ==============================
# 3. Feature Extraction (TF-IDF)
# ==============================

print("\nExtracting TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)  # unigrams + bigrams
)

X_train = vectorizer.fit_transform(train["clean_text"])
X_test = vectorizer.transform(test["clean_text"])

y_train = train["label"]
y_test = test["label"]

print("Feature matrix shape:", X_train.shape)


# ==============================
# 4. Define Classifiers
# ==============================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Naive Bayes": MultinomialNB()
}
# ==============================
# Create Result Folders
# ==============================

plot_path = r"C:\Users\dell\PycharmProjects\MasterThesis\Result\Plot"
score_path = r"C:\Users\dell\PycharmProjects\MasterThesis\Result\Score"

os.makedirs(plot_path, exist_ok=True)
os.makedirs(score_path, exist_ok=True)

# ==============================
# 5. Train and Evaluate Models
# ==============================

results = {}

print("\nTraining and evaluating models...\n")

for name, model in models.items():

    print("Training:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    score = balanced_accuracy_score(y_test, predictions)

    results[name] = score

    print(name, "Balanced Accuracy:", round(score, 4))
    print()


# ==============================
# 6. Display Final Comparison
# ==============================

print("\nFinal Model Comparison")
print("-----------------------")

for model, score in results.items():
    print(f"{model}: {score:.4f}")


# ==============================
# 7. Plot Results
# ==============================

# ==============================
# Save Plot
# ==============================

model_names = list(results.keys())
scores = list(results.values())

plt.figure(figsize=(8,5))
plt.bar(model_names, scores)

plt.ylabel("Balanced Accuracy")
plt.title("Classifier Performance Comparison")

plt.xticks(rotation=30)
plt.tight_layout()

# timestamp to avoid overwriting
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

plot_file = os.path.join(plot_path, f"classifier_comparison_{timestamp}.png")

plt.savefig(plot_file)

print("\nPlot saved to:", plot_file)

plt.show()

# ==============================
# 8. Best Model
# ==============================

best_model = max(results, key=results.get)

print("\nBest Classifier:", best_model)
print("Best Balanced Accuracy:", round(results[best_model], 4))

# ==============================
# Save Scores
# ==============================

score_file = os.path.join(score_path, f"classifier_scores_{timestamp}.txt")

with open(score_file, "w") as f:

    f.write("Final Model Comparison\n")
    f.write("----------------------\n\n")

    for model, score in results.items():
        f.write(f"{model}: {score:.4f}\n")

    f.write("\nBest Classifier: " + best_model)
    f.write(f"\nBest Balanced Accuracy: {results[best_model]:.4f}\n")

print("Scores saved to:", score_file)