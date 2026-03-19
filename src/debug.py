import joblib

path = r"C:\Users\dell\PycharmProjects\LLM-Evaluation-MasterThesis\Result\Score\tfidf_fold_results.pkl"

data = joblib.load(path)

print("LinearSVC scores:", data["LinearSVC"])
print("Number of folds:", len(data["LinearSVC"]))