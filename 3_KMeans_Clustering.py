import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.title("Experiment 3: K-Means Clustering")

st.write("### Aim")
st.write("To understand how K-Means clustering groups similar data points into clusters.")

st.write("### Theory")
st.write(
    "K-Means is an unsupervised machine learning algorithm. "
    "It divides data into K clusters based on similarity."
)

# Sample dataset
X = np.array([
    [1, 2], [2, 3], [3, 3], [8, 8], [9, 10], [10, 9],
    [25, 30], [24, 29], [26, 31]
])

st.write("### Dataset")
st.write("Sample 2D points used for clustering:")
st.dataframe(X)

k = st.slider("Select number of clusters (K)", min_value=2, max_value=5, value=3)

if st.button("Run K-Means Clustering"):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)

    labels = model.labels_
    centers = model.cluster_centers_

    st.write("### Output")
    st.write("Cluster labels for each point:")
    st.write(labels)

    st.write("Cluster centers:")
    st.write(centers)

    fig, ax = plt.subplots()
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, s=100)
    ax.scatter(centers[:, 0], centers[:, 1], marker='X', s=200)
    ax.set_title("K-Means Clustering")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    for i, point in enumerate(X):
        ax.text(point[0] + 0.2, point[1] + 0.2, f"P{i+1}", fontsize=9)

    st.pyplot(fig)

    st.write("### Result")
    st.write(
        f"The dataset was grouped into {k} clusters using the K-Means algorithm."
    )