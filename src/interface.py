import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# Load data
wine = pd.read_csv("data/raw/wine.csv")

#labels = wine.target_names

#creating a subset
subset = wine[['X7', 'X12', 'X6', 'X4', 'Y']]
print(subset.head())

# renamed the columns
subset.columns = ['Flavanoids', 'OD280/OD315 of diluted wines', 'Total phenols', 'Alcalinity of ash', 'Wine Class']

## independent variables, feature matrix
X = subset[['Flavanoids', 'OD280/OD315 of diluted wines', 'Total phenols','Alcalinity of ash']]

### The predicted variable
y = subset['Wine Class']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluation
report = classification_report(y_test, y_pred) #target_names=labels)
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = clf.score(X_test, y_test)

# Streamlit App
st.set_page_config(page_title="wine Classifier", layout="wide")
st.title("wine Classification - web app")

# Show raw data
if st.checkbox("📊 Show Raw Data"):
    st.write(X)

# Model Summary
model_summary = f"""
Model: RandomForestClassifier
Training set size: {len(X_train)}
Test set size: {len(X_test)}
Model accuracy: {accuracy:.2f}
"""
st.subheader("🧠 Model Summary")
st.text(model_summary)

# Classification Report
st.subheader("📄 Classification Report")
st.text(report)

# Feature Importance Plot
st.subheader("📌 Feature Importance")
fig_feat, ax_feat = plt.subplots()
pd.Series(clf.feature_importances_, index=X.columns).sort_values().plot(kind='barh', color='teal', ax=ax_feat)
st.pyplot(fig_feat)

# Sidebar Input
st.sidebar.header("🔍 Predict New Sample")
Flavanoids = st.sidebar.slider("Flavanoids", float(X.min()[0]), float(X.max()[0]), float(X.mean()[0]))
D280_0D315_of_diluted_wines = st.sidebar.slider("D280_0D315_of_diluted_wines", float(X.min()[1]), float(X.max()[1]), float(X.mean()[1]))
Total_phenols = st.sidebar.slider("Total_phenols", float(X.min()[2]), float(X.max()[2]), float(X.mean()[2]))
Alcalinity_of_ash = st.sidebar.slider("Alcalinity_of_ash", float(X.min()[3]), float(X.max()[3]), float(X.mean()[3]))

sample = [[Flavanoids, D280_0D315_of_diluted_wines, Total_phenols, Alcalinity_of_ash]]
prediction = clf.predict(sample)
st.sidebar.success(f"🌸 Prediction: {y_pred[prediction[0]]}")

