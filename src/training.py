from sklearn.ensemble import RandomForestClassifier


def fit_model(X_train, y_train):
    """Train a RandomForestClassifier model."""
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    return clf


def model_predict(clf, X_test):
    """Make predictions using the trained model."""
    return clf.predict(X_test)
