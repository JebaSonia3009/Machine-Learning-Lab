import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

st.title("Experiment 10: Feature Scaling and Normalization")

st.write("### Aim")
st.write("To understand how feature scaling and normalization transform data before machine learning.")

st.write("### Theory")
st.write(
    "Feature scaling is a preprocessing step used to bring numerical features into a comparable range. "
    "This is important because some machine learning algorithms work better when the features are on similar scales."
)

with st.expander("Read simple explanation"):
    st.write(
        """
        Suppose one feature has values like **1, 2, 3** and another has values like **1000, 2000, 3000**.  
        The second feature is much larger in magnitude, so some algorithms may give it more importance.

        **Standardization** changes data so that:
        - mean becomes close to 0
        - standard deviation becomes close to 1

        **Min-Max Scaling** changes data so that:
        - all values come between 0 and 1

        These transformations help models learn better and faster.
        """
    )

st.write("### Sample Dataset")
data = {
    "Study Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Marks": [35, 40, 50, 60, 65, 70, 80, 90],
    "Family Income": [15000, 18000, 22000, 26000, 30000, 35000, 40000, 50000]
}

df = pd.DataFrame(data)

tab1, tab2, tab3, tab4 = st.tabs([
    "Original Dataset",
    "Standardization",
    "Min-Max Scaling",
    "Comparison"
])

with tab1:
    st.write("### Original Data")
    st.dataframe(df, use_container_width=True)

    st.write("### Observation")
    st.info(
        "Notice that the feature values are in very different ranges. "
        "For example, Study Hours is very small, while Family Income is much larger."
    )

with tab2:
    scaler_std = StandardScaler()
    std_scaled = scaler_std.fit_transform(df)
    std_df = pd.DataFrame(std_scaled, columns=df.columns)

    st.write("### Standardized Data")
    st.dataframe(std_df, use_container_width=True)

    st.write("### Explanation")
    st.write(
        "After standardization, the values are centered around 0 and scaled to have similar spread."
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    std_df.plot(kind="bar", ax=ax)
    ax.set_title("Standardized Features")
    ax.set_ylabel("Transformed Value")
    st.pyplot(fig)

with tab3:
    scaler_mm = MinMaxScaler()
    mm_scaled = scaler_mm.fit_transform(df)
    mm_df = pd.DataFrame(mm_scaled, columns=df.columns)

    st.write("### Min-Max Scaled Data")
    st.dataframe(mm_df, use_container_width=True)

    st.write("### Explanation")
    st.write(
        "After Min-Max Scaling, all feature values are converted into the range 0 to 1."
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    mm_df.plot(kind="bar", ax=ax)
    ax.set_title("Min-Max Scaled Features")
    ax.set_ylabel("Scaled Value")
    st.pyplot(fig)

with tab4:
    st.write("### Before and After Comparison")

    feature_name = st.selectbox(
        "Select feature for comparison",
        df.columns
    )

    scaler_std = StandardScaler()
    std_scaled = scaler_std.fit_transform(df)
    std_df = pd.DataFrame(std_scaled, columns=df.columns)

    scaler_mm = MinMaxScaler()
    mm_scaled = scaler_mm.fit_transform(df)
    mm_df = pd.DataFrame(mm_scaled, columns=df.columns)

    comparison_df = pd.DataFrame({
        "Original": df[feature_name],
        "Standardized": std_df[feature_name],
        "Min-Max Scaled": mm_df[feature_name]
    })

    st.dataframe(comparison_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(comparison_df.index, comparison_df["Original"], marker="o", label="Original")
    ax.plot(comparison_df.index, comparison_df["Standardized"], marker="o", label="Standardized")
    ax.plot(comparison_df.index, comparison_df["Min-Max Scaled"], marker="o", label="Min-Max Scaled")
    ax.set_title(f"Comparison of {feature_name}")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Value")
    ax.legend()
    st.pyplot(fig)

    st.success("Comparison completed successfully.")
    st.balloons()

st.write("### Key Points")
st.write(
    """
    - Feature scaling is important when features have very different ranges.
    - Standardization is useful when data should be centered around 0.
    - Min-Max Scaling is useful when values should be restricted between 0 and 1.
    - Many machine learning algorithms perform better after proper preprocessing.
    """
)

st.write("### Learning Outcome")
st.write(
    "Students can understand why feature scaling is needed and how Standardization and Min-Max Scaling change the original dataset."
)