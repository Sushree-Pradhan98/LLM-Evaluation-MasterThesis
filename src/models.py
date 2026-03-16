from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB


def get_models():

    models = {

        "Logistic Regression":
            LogisticRegression(max_iter=1000),

        "Support Vector Machine":
            LinearSVC(C=2),

        "Random Forest":
            RandomForestClassifier(n_estimators=200),

        "Naive Bayes":
            MultinomialNB()
    }

    return models