from sklearn.metrics import balanced_accuracy_score


def evaluate_model(model, X_train, X_eval, y_train, y_eval):

    model.fit(X_train, y_train)

    predictions = model.predict(X_eval)

    score = balanced_accuracy_score(y_eval, predictions)

    return score