from src.evaluate import evaluate_model
from src.preprocessing import preprocess_dataset
from src.streamlit_utils import display_model_summary
from src.training import fit_model, model_predict


def training_pipeline():
    filename = "wine.csv"
    subset_columns = ["X7", "X12", "X6", "X4", "Y"]
    new_column_names = [
        "Flavanoids",
        "OD280/OD315 of diluted wines",
        "Total phenols",
        "Alcalinity of ash",
        "Wine Class",
    ]
    target_column = "Wine Class"

    # Preprocess
    X, y, X_train, X_test, y_train, y_test = preprocess_dataset(
        filename=filename,
        subset_columns=subset_columns,
        new_column_names=new_column_names,
        target_column=target_column,
    )

    trained_model = fit_model(X_train, y_train)
    y_pred = model_predict(trained_model, X_test)

    report_metrics = evaluate_model(trained_model, X_test, y_test, y_pred)

    print(
        f"Classification Report:\n{report_metrics['classification_report']}\nConfusion Matrix:\n{report_metrics['confusion_matrix']}\nAccuracy: {report_metrics['accuracy']:.2f}"
    )

    display_model_summary(X, X_train, X_test, y_pred, trained_model, report_metrics)


if __name__ == "__main__":
    training_pipeline()
