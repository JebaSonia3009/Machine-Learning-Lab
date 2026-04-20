import streamlit as st

st.title("Experiment 1: Addition Experiment")

st.write("### Aim")
st.write("To add two numbers interactively.")

a = st.number_input("Enter first number", value=0)
b = st.number_input("Enter second number", value=0)

if st.button("Run Experiment"):
    result = a + b
    st.write("### Output")
    st.success(f"Result = {result}")

    st.write("### Result")
    st.write("The addition experiment was performed successfully.")