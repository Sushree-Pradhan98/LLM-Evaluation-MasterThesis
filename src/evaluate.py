from sklearn.metrics import balanced_accuracy_score


def evaluate_model(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    score = balanced_accuracy_score(y_test, predictions)

    return score