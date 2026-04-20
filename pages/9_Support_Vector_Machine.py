import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.title("Experiment 9: Support Vector Machine Classification")

st.write("### Aim")
st.write("To understand how Support Vector Machine (SVM) classifies data by finding the best decision boundary.")

st.write("### Theory")
st.write(
    "Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification. "
    "It tries to find the best boundary that separates different classes. "
    "The closest points to this boundary are called support vectors, and they play an important role in defining the classifier."
)

with st.expander("Read simple explanation of SVM"):
    st.write(
        """
        Imagine two groups of points on a graph.  
        SVM tries to draw the best dividing line between them.  
        It does not choose just any line — it chooses the line with the maximum margin, 
        which means the line that keeps the classes as far apart as possible.
        
        **Why is it useful?**
        - Works well for classification problems
        - Effective even when classes are close
        - Can handle linear and non-linear separation
        
        **Important terms:**
        - **Decision boundary**: the line or curve that separates classes
        - **Margin**: distance between boundary and nearest data points
        - **Support vectors**: the nearest data points that define the boundary
        """
    )

st.write("### Dataset Used")
st.write(
    "This experiment uses the Iris dataset. It contains measurements of iris flowers. "
    "The model predicts the flower class based on sepal and petal values."
)

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df["target"] = y
df["flower_name"] = df["target"].map({
    0: target_names[0],
    1: target_names[1],
    2: target_names[2]
})

tab1, tab2, tab3 = st.tabs(["Dataset Preview", "Class Distribution", "Experiment"])

with tab1:
    st.write("### First 10 Rows of the Dataset")
    st.dataframe(df.head(10), use_container_width=True)

    st.write("### Dataset Information")
    st.write(f"Number of samples: {df.shape[0]}")
    st.write(f"Number of features: {len(feature_names)}")
    st.write("Target classes:")
    for i, name in enumerate(target_names):
        st.write(f"- Class {i}: {name.title()}")

with tab2:
    class_counts = df["flower_name"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(class_counts.index, class_counts.values)
    ax.set_title("Class Distribution in Iris Dataset")
    ax.set_xlabel("Flower Class")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.info("This chart shows how many samples belong to each flower class.")

with tab3:
    st.write("### Choose SVM Settings")
    kernel = st.selectbox("Select kernel type", ["linear", "rbf", "poly"], index=0)
    c_value = st.slider("Select C value (regularization)", 0.1, 5.0, 1.0, 0.1)

    st.write("### Enter Input Features")
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5, 0.1)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2, 0.1)

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = SVC(kernel=kernel, C=c_value, probability=True, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    if st.button("Run SVM Classification"):
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        st.write("### Output")
        st.success(f"Predicted Flower Class: {target_names[prediction].title()}")
        st.balloons()

        st.write("### Prediction Confidence")
        for i, prob in enumerate(probabilities):
            st.write(f"{target_names[i].title()}: {prob:.2f}")

        st.write("### Model Performance on Test Data")
        st.metric("Test Accuracy", f"{acc:.2f}")

        st.write("### Interpretation")
        st.info(
            f"The SVM model used a **{kernel} kernel** and found the best decision boundary "
            "to classify the flower sample."
        )

        st.write("### Input Sample Entered by Student")
        input_df = pd.DataFrame(
            input_data,
            columns=feature_names
        )
        st.dataframe(input_df, use_container_width=True)

        st.write("### Simple Visualization")
        fig, ax = plt.subplots(figsize=(8, 5))
        scatter = ax.scatter(X[:, 2], X[:, 3], c=y, s=40)
        ax.scatter(input_data[0, 2], input_data[0, 3], marker="X", s=200)
        ax.set_xlabel("Petal Length")
        ax.set_ylabel("Petal Width")
        ax.set_title("SVM Classification Visualization")
        st.pyplot(fig)

        with st.expander("See classification report"):
            report = classification_report(
                y_test, y_pred, target_names=target_names, output_dict=False
            )
            st.text(report)

        st.write("### Result")
        st.write(
            f"The input sample was classified as **{target_names[prediction].title()}** "
            f"using the Support Vector Machine algorithm with **{kernel} kernel**."
        )

st.write("### Learning Outcome")
st.write(
    "Students can understand how SVM separates classes using the best decision boundary "
    "and how kernel selection affects classification."
)