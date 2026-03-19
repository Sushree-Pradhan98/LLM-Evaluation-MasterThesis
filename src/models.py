from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC


def get_models():

    return {

        "LinearSVC": LinearSVC(C=2),

        "RandomForest":
            RandomForestClassifier(n_estimators=200),

        "LogisticRegression":
            LogisticRegression(max_iter=1000),

        "MLP":
            MLPClassifier(hidden_layer_sizes=(100,)),

        "MultinomialNB":
            MultinomialNB()

        # ❌ REMOVE THIS FOR NOW
        # "GaussianNB": GaussianNB()
    }