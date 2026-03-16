from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_features(train_texts, test_texts):

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1,2)
    )

    X_train = vectorizer.fit_transform(train_texts)

    X_test = vectorizer.transform(test_texts)

    return X_train, X_test, vectorizer