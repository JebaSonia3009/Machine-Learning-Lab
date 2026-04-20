import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

st.title("Experiment 7: K-Nearest Neighbors Classification")

st.write("### Aim")
st.write("To understand how K-Nearest Neighbors (KNN) classifies data samples based on the nearest neighbors.")

st.write("### Theory")
st.write(
    "K-Nearest Neighbors is a supervised machine learning algorithm. "
    "It classifies a new sample based on the majority class among its nearest neighbors."
)

st.write("### About the Dataset")
st.write(
    "This experiment uses the Iris dataset. The model predicts the flower type "
    "based on sepal and petal measurements."
)

iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

k_value = st.slider("Select value of K", 1, 10, 3)

st.write("### Input Features")
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5, 0.1)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2, 0.1)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button("Run KNN Classification"):
    model = KNeighborsClassifier(n_neighbors=k_value)
    model.fit(X, y)

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    st.write("### Output")
    st.success(f"Predicted Flower Class: {target_names[prediction].title()}")
    st.snow()

    st.write("### Prediction Confidence")
    for i, prob in enumerate(probabilities):
        st.write(f"{target_names[i].title()}: {prob:.2f}")

    st.write("### Interpretation")
    st.info(
        f"The model checked the {k_value} nearest neighbors of the given sample "
        "and predicted the class based on majority voting."
    )

    st.write("### Simple Visualization")
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(X[:, 2], X[:, 3], c=y, s=40)
    ax.scatter(input_data[0, 2], input_data[0, 3], marker='X', s=200)
    ax.set_xlabel("Petal Length")
    ax.set_ylabel("Petal Width")
    ax.set_title("KNN Classification Visualization")
    st.pyplot(fig)

    st.write("### Result")
    st.write(
        f"The given sample was classified as **{target_names[prediction].title()}** "
        f"using the K-Nearest Neighbors algorithm with K = {k_value}."
    )

st.write("### Learning Outcome")
st.write(
    "Students can understand how KNN uses distance and nearest samples to classify new data points."
)