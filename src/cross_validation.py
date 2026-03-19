from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import numpy as np


def run_kfold(models, X, y):

    rkf = RepeatedStratifiedKFold(
        n_splits=2,
        n_repeats=5,
        random_state=42
    )

    results = {name: [] for name in models.keys()}

    for fold, (train_idx, test_idx) in enumerate(rkf.split(X, y)):

        print(f"\nFold {fold+1}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for name, model in models.items():

            import numpy as np

            # Convert only if sparse
            if name == "GaussianNB":
                if hasattr(X_train, "toarray"):  # TF-IDF case
                    X_train_tmp = X_train.toarray()
                    X_test_tmp = X_test.toarray()
                else:  # SBERT case
                    X_train_tmp = np.array(X_train)
                    X_test_tmp = np.array(X_test)
            else:
                X_train_tmp = X_train
                X_test_tmp = X_test

            model.fit(X_train_tmp, y_train)

            preds = model.predict(X_test_tmp)

            score = balanced_accuracy_score(y_test, preds)

            results[name].append(score)

            print(name, ":", round(score, 4))

    return results