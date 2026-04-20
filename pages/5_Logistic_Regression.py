import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

st.title("Experiment 5: Logistic Regression")

st.write("### Aim")
st.write("To understand how Logistic Regression is used for binary classification.")

st.write("### Theory")
st.write(
    "Logistic Regression is a supervised machine learning algorithm used for classification. "
    "It predicts the probability of an input belonging to a class and then assigns the final class label."
)

st.write("### About the Dataset")
st.write(
    "This experiment uses a simple dataset of study hours and pass/fail outcome. "
    "The model predicts whether a student is likely to pass or fail."
)

# Sample dataset
X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # 0 = Fail, 1 = Pass

# Train model
model = LogisticRegression()
model.fit(X, y)

st.write("### Dataset")
st.write("Study Hours:", X.flatten())
st.write("Outcome:", y)

study_hours = st.slider("Enter study hours", 1, 10, 4)

if st.button("Run Logistic Regression"):
    prediction = model.predict([[study_hours]])[0]
    probability = model.predict_proba([[study_hours]])[0]

    class_label = "Pass" if prediction == 1 else "Fail"

    st.write("### Output")
    st.success(f"Predicted Class: {class_label}")

    st.write("### Prediction Probability")
    st.write(f"Probability of Fail: {probability[0]:.2f}")
    st.write(f"Probability of Pass: {probability[1]:.2f}")

    st.info(
        "The model predicts the probability of each class and assigns the class with higher probability."
    )

    # Plot
    x_test = np.linspace(1, 10, 200).reshape(-1, 1)
    y_prob = model.predict_proba(x_test)[:, 1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(X, y, label="Training Data")
    ax.plot(x_test, y_prob, label="Logistic Curve")
    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Probability of Passing")
    ax.set_title("Logistic Regression")
    ax.legend()

    st.pyplot(fig)

    st.write("### Result")
    st.write(
        f"For {study_hours} study hours, the model predicts that the student is likely to **{class_label}**."
    )

st.write("### Learning Outcome")
st.write(
    "Students can understand how Logistic Regression is used to predict binary outcomes based on input data."
)