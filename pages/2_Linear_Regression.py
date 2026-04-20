import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title("Experiment 2: Linear Regression")

st.write("### Aim")
st.write("To understand how linear regression predicts output values.")

st.write("### Theory")
st.write("Linear regression finds the best-fit straight line between input and output values.")

# Sample dataset
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5, 7, 8, 9, 10, 12])

# Train model
model = LinearRegression()
model.fit(X, y)

st.write("### Dataset")
st.write("Input X:", X.flatten())
st.write("Output y:", y)

input_value = st.number_input("Enter a new X value for prediction", value=6)

if st.button("Run Linear Regression"):
    prediction = model.predict([[input_value]])[0]

    st.write("### Output")
    st.success(f"Predicted Y value for X = {input_value} is {prediction:.2f}")

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(X, y, label="Data Points")
    ax.plot(X, model.predict(X), label="Regression Line")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Linear Regression")
    ax.legend()

    st.pyplot(fig)

    st.write("### Result")
    st.write("The model predicted the output using the best-fit linear relationship.")