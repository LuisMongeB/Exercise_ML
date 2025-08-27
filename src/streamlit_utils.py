import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def display_model_summary(X, X_train, X_test, y_pred, clf, report):
    """
    Display the model summary, classification report, and feature importance plot in a Streamlit app.

    Parameters:
    X (pd.DataFrame): The full dataset features.
    X_train (pd.DataFrame): The training features.
    X_test (pd.DataFrame): The test features.
    accuracy (float): The model accuracy.
    clf (RandomForestClassifier): The trained RandomForestClassifier model.
    report (str): The classification report as a string.
    """
    # Streamlit App
    st.set_page_config(page_title="Wine Classifier", layout="wide")

    st.subheader("🧠 Model Summary")
    model_summary_text = f"""
    Model: RandomForestClassifier
    Training set size: {len(X_train)}
    Test set size: {len(X_test)}
    Model accuracy: {report['accuracy']:.2f}
    """
    st.text(model_summary_text)

    # Classification Report
    st.subheader("📄 Classification Report")
    st.text(report["classification_report"])
    st.text(report["confusion_matrix"])
    st.text(report["accuracy"])

    # Feature Importance Plot
    st.subheader("📌 Feature Importance")
    fig_feat, ax_feat = plt.subplots()
    pd.Series(clf.feature_importances_, index=X.columns).sort_values().plot(
        kind="barh", color="teal", ax=ax_feat
    )
    st.pyplot(fig_feat)

    # Sidebar Input
    st.sidebar.header("🔍 Predict New Sample")
    Flavanoids = st.sidebar.slider(
        "Flavanoids",
        float(X.min().iloc[0]),
        float(X.max().iloc[0]),
        float(X.mean().iloc[0]),
    )
    D280_0D315_of_diluted_wines = st.sidebar.slider(
        "D280_0D315_of_diluted_wines",
        float(X.min().iloc[1]),
        float(X.max().iloc[1]),
        float(X.mean().iloc[1]),
    )
    Total_phenols = st.sidebar.slider(
        "Total_phenols",
        float(X.min().iloc[2]),
        float(X.max().iloc[2]),
        float(X.mean().iloc[2]),
    )
    Alcalinity_of_ash = st.sidebar.slider(
        "Alcalinity_of_ash",
        float(X.min().iloc[3]),
        float(X.max().iloc[3]),
        float(X.mean().iloc[3]),
    )

    sample = [
        [Flavanoids, D280_0D315_of_diluted_wines, Total_phenols, Alcalinity_of_ash]
    ]
    prediction = clf.predict(sample)
    st.sidebar.success(f"🌸 Prediction: {y_pred[prediction[0]]}")
