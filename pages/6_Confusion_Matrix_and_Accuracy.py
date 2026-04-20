import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

st.title("Experiment 6: Confusion Matrix and Accuracy Evaluation")

st.write("### Aim")
st.write("To understand how classification model performance is evaluated using confusion matrix and accuracy metrics.")

st.write("### Theory")
st.write(
    "A confusion matrix shows how many predictions are correct and how many are incorrect. "
    "Accuracy, precision, recall, and F1-score are common evaluation metrics used in classification problems."
)

st.write("### About the Dataset")
st.write(
    "This experiment uses sample actual and predicted class labels. "
    "Based on these values, the confusion matrix and evaluation scores are calculated."
)

# Sample actual and predicted values
y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1, 1, 1])

st.write("### Actual Labels")
st.write(y_true)

st.write("### Predicted Labels")
st.write(y_pred)

if st.button("Run Evaluation"):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    st.write("### Output")
    st.success("Evaluation completed successfully.")
    st.balloons()

    st.write("### Evaluation Metrics")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Accuracy", f"{acc:.2f}")
        st.metric("Precision", f"{prec:.2f}")

    with col2:
        st.metric("Recall", f"{rec:.2f}")
        st.metric("F1-Score", f"{f1:.2f}")

    st.write("### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    st.pyplot(fig)

    st.write("### Interpretation")
    st.info(
        "The confusion matrix shows the number of correct and incorrect predictions for each class. "
        "Higher diagonal values indicate better model performance."
    )

    st.write("### Result")
    st.write(
        "The classification model was evaluated using confusion matrix, accuracy, precision, recall, and F1-score."
    )

st.write("### Learning Outcome")
st.write(
    "Students can understand how to measure classification performance and interpret model evaluation results."
)