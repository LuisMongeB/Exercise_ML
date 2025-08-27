from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(clf, X_test, y_test, y_pred):
    """
    Evaluate the trained model and return evaluation metrics.

    Parameters:
    clf (RandomForestClassifier): The trained RandomForestClassifier model.
    X_test (pd.DataFrame): The test features.
    y_test (pd.Series): The true labels for the test set.
    y_pred (np.ndarray): The predicted labels.

    Returns:
    dict: A dictionary containing the classification report, confusion matrix, and accuracy.
    """
    report = classification_report(y_test, y_pred)  # target_names=labels)
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = clf.score(X_test, y_test)

    return {
        "classification_report": report,
        "confusion_matrix": conf_matrix,
        "accuracy": accuracy,
    }
