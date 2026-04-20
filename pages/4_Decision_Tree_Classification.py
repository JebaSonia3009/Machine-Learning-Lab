import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris

st.title("Experiment 4: Decision Tree Classification")

st.write("### Aim")
st.write("To understand how a Decision Tree Classifier predicts the class of a data sample.")

st.write("### Theory")
st.write(
    "Decision Tree is a supervised machine learning algorithm used for classification. "
    "It splits the data into branches based on feature values and finally predicts a class."
)

st.write("### About the Dataset")
st.write(
    "This experiment uses the Iris dataset. The dataset contains flower measurements "
    "and the model predicts the flower type."
)

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Train model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

st.write("### Input Features")
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5, 0.1)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2, 0.1)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button("Run Decision Tree Classification"):
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    st.write("### Output")
    st.success(f"Predicted Flower Class: {target_names[prediction].title()}")
    st.balloons()

    st.write("### Prediction Confidence")
    for i, prob in enumerate(probabilities):
        st.write(f"{target_names[i].title()}: {prob:.2f}")

    st.write("### Interpretation")
    st.info(
        "The Decision Tree model examined the input feature values and followed a sequence "
        "of rules to predict the final class."
    )

    st.write("### Decision Tree Visualization")
    fig, ax = plt.subplots(figsize=(14, 8))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=target_names,
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax
    )
    st.pyplot(fig)

    st.write("### Result")
    st.write(
        f"The given sample was classified as **{target_names[prediction].title()}** "
        "using the Decision Tree algorithm."
    )

st.write("### Learning Outcome")
st.write(
    "Students can understand how Decision Trees use feature-based rules to classify data "
    "into different categories."
)