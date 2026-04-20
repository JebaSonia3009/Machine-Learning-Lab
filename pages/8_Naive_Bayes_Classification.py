import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris

st.title("Experiment 8: Naive Bayes Classification")

st.write("### Aim")
st.write("To understand how Naive Bayes predicts classes using probability.")

st.write("### Theory")
st.write(
    "Naive Bayes is a supervised machine learning algorithm based on Bayes' theorem. "
    "It predicts the class of a sample using probability and assumes that features are independent."
)

st.write("### About the Dataset")
st.write(
    "This experiment uses the Iris dataset. The model predicts the flower class "
    "from sepal and petal measurements."
)

iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

model = GaussianNB()
model.fit(X, y)

st.write("### Input Features")
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5, 0.1)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2, 0.1)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button("Run Naive Bayes Classification"):
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    st.write("### Output")
    st.success(f"Predicted Flower Class: {target_names[prediction].title()}")
    st.snow()

    st.write("### Prediction Probability")
    for i, prob in enumerate(probabilities):
        st.write(f"{target_names[i].title()}: {prob:.2f}")

    st.write("### Interpretation")
    st.info(
        "The Naive Bayes model calculated the probability of the input belonging "
        "to each class and selected the class with the highest probability."
    )

    st.write("### Simple Visualization")
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(X[:, 0], X[:, 2], c=y, s=40)
    ax.scatter(input_data[0, 0], input_data[0, 2], marker='X', s=200)
    ax.set_xlabel("Sepal Length")
    ax.set_ylabel("Petal Length")
    ax.set_title("Naive Bayes Classification Visualization")
    st.pyplot(fig)

    st.write("### Result")
    st.write(
        f"The given sample was classified as **{target_names[prediction].title()}** "
        "using the Naive Bayes algorithm."
    )

st.write("### Learning Outcome")
st.write(
    "Students can understand how probability is used in machine learning classification."
)