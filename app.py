import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import r2_score, mean_absolute_error

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Salary Prediction",
    page_icon="💼",
    layout="wide"
)

# --------------------------------------------------
# Load Dataset & Model
# --------------------------------------------------
df = pd.read_csv("Salary_Data.csv")
model = joblib.load("salary_model.pkl")   # Your saved model

# --------------------------------------------------
# Model Evaluation
# --------------------------------------------------
X = df[["YearsExperience"]]
y = df["Salary"]

predictions = model.predict(X)

r2 = r2_score(y, predictions)
mae = mean_absolute_error(y, predictions)

# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------
st.sidebar.title("📂 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "💼 Salary Prediction",
        "📈 Data Visualization",
        "🤖 Model Information",
        "📊 Dataset Statistics",
        "ℹ️ About Project",
        "👨‍💻 Developer"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Salary Prediction using\nSimple Linear Regression"
)

# ==================================================
# HOME PAGE
# ==================================================
if page == "🏠 Home":

    st.title("💼 Salary Prediction using Simple Linear Regression")

    st.write(
        """
        Welcome to the **Salary Prediction Web App**.

        This application predicts an employee's salary
        based on their **Years of Experience** using a
        **Simple Linear Regression** machine learning model.
        """
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📌 Project Overview")

        st.write("""
        ✔ Predict Salary

        ✔ Machine Learning Model

        ✔ Data Visualization

        ✔ Model Performance

        ✔ Dataset Statistics
        """)

    with col2:

        st.subheader("📂 Dataset Information")

        st.write(f"**Dataset Name:** Salary_Data.csv")

        st.write(f"**Total Records:** {df.shape[0]}")

        st.write(f"**Total Features:** {df.shape[1]}")

        st.write(f"**Input Feature:** YearsExperience")

        st.write(f"**Target Variable:** Salary")

    st.markdown("---")

    st.subheader("📄 Dataset Preview")

    st.dataframe(df.head(10), use_container_width=True)

    with st.expander("📖 Project Description"):

        st.write("""
        This project demonstrates the implementation of
        **Simple Linear Regression** using Scikit-learn.

        The model learns the relationship between
        **Years of Experience** and **Salary**.

        Users simply enter their years of experience,
        and the trained model predicts the expected salary.
        """)
# ==================================================
# SALARY PREDICTION PAGE
# ==================================================
elif page == "💼 Salary Prediction":

    st.title("💼 Salary Prediction")

    st.write("Enter your years of experience to predict the estimated salary.")

    experience = st.number_input(
        "Years of Experience",
        min_value=0.0,
        max_value=50.0,
        value=1.0,
        step=0.1
    )

    if st.button("🚀 Predict Salary"):

        with st.spinner("Predicting Salary..."):

            input_data = np.array([[experience]])

            predicted_salary = model.predict(input_data)[0]

        st.success("Prediction Completed Successfully!")

        st.metric(
            label="💰 Predicted Salary",
            value=f"₹ {predicted_salary:,.2f}"
        )

        st.balloons()

        st.info(
            f"""
            For **{experience:.1f} years**
            of experience,

            the estimated salary is

            **₹ {predicted_salary:,.2f}**
            """
        )

# ==================================================
# DATA VISUALIZATION PAGE
# ==================================================
elif page == "📈 Data Visualization":

    st.title("📈 Data Visualization")

    st.subheader("Scatter Plot with Regression Line")

    fig, ax = plt.subplots(figsize=(8,5))

    # Scatter Plot
    ax.scatter(
        df["YearsExperience"],
        df["Salary"],
        label="Actual Data"
    )

    # Regression Line
    ax.plot(
        df["YearsExperience"],
        predictions,
        linewidth=2,
        label="Regression Line"
    )

    # Predicted Point (only if experience exists)
    try:
        ax.scatter(
            experience,
            predicted_salary,
            s=120,
            marker="*",
            label="Predicted Salary"
        )
    except:
        pass

    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary")
    ax.set_title("Salary Prediction using Linear Regression")
    ax.legend()

    st.pyplot(fig)

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(df, use_container_width=True)

# ==================================================
# DATASET STATISTICS PAGE
# ==================================================
elif page == "📊 Dataset Statistics":

    st.title("📊 Dataset Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Records", df.shape[0])
        st.metric("Total Features", df.shape[1])

    with col2:
        st.metric("Missing Values", int(df.isnull().sum().sum()))
        st.metric("Duplicate Records", int(df.duplicated().sum()))

    st.markdown("---")

    st.subheader("Statistical Summary")

    st.dataframe(df.describe(), use_container_width=True)

    st.markdown("---")

    st.subheader("Missing Values")

    st.dataframe(df.isnull().sum().to_frame("Missing Values"))

    st.markdown("---")

    st.subheader("Column Information")

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str)
    })

    st.dataframe(info_df, use_container_width=True)
# ==================================================
# MODEL INFORMATION PAGE
# ==================================================
elif page == "🤖 Model Information":

    st.title("🤖 Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Algorithm", "Linear Regression")
        st.metric("R² Score", f"{r2:.4f}")

    with col2:
        st.metric("Mean Absolute Error", f"{mae:.2f}")
        st.metric("Train-Test Split", "80 : 20")

    st.markdown("---")

    st.subheader("Model Summary")

    st.info("""
    • Algorithm : Simple Linear Regression

    • Input Feature : YearsExperience

    • Target Variable : Salary

    • The model predicts salary based on years of experience.

    • A higher R² Score indicates better prediction performance.
    """)

# ==================================================
# ABOUT PROJECT PAGE
# ==================================================
elif page == "ℹ️ About Project":

    st.title("ℹ️ About Project")

    st.subheader("Project Overview")

    st.write("""
    This project demonstrates the implementation of
    **Simple Linear Regression** using Python and
    Scikit-learn.

    The model learns the relationship between
    **Years of Experience** and **Salary** and predicts
    salary for new experience values.
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Details")

        st.write("**Dataset Name:** Salary_Data.csv")
        st.write(f"**Total Records:** {df.shape[0]}")
        st.write(f"**Total Features:** {df.shape[1]}")
        st.write("**Input Feature:** YearsExperience")
        st.write("**Target Variable:** Salary")

    with col2:
        st.subheader("Libraries Used")

        st.write("""
        ✔ Pandas

        ✔ NumPy

        ✔ Matplotlib

        ✔ Scikit-learn

        ✔ Streamlit

        ✔ Joblib
        """)

    st.markdown("---")

    with st.expander("Project Workflow"):

        st.write("""
        1. Load Dataset

        2. Data Preprocessing

        3. Train-Test Split

        4. Train Linear Regression Model

        5. Evaluate Model

        6. Predict Salary

        7. Visualize Results
        """)

# ==================================================
# DEVELOPER PAGE
# ==================================================
elif page == "👨‍💻 Developer":

    st.title("👨‍💻 Developer")

    st.success("Thank you for using this application!")

    st.markdown("---")

    st.subheader("Developer Information")

    st.write("**Name:** Your Name")
    st.write("**Role:** Machine Learning Enthusiast")
    st.write("**Project:** Salary Prediction using Simple Linear Regression")

    st.markdown("---")

    st.subheader("Connect")

    st.write("📧 Email: your_email@example.com")
    st.write("🐙 GitHub: https://github.com/yourusername")
    st.write("💼 LinkedIn: https://linkedin.com/in/yourusername")

    st.markdown("---")

    st.caption("© 2026 Salary Prediction App | Built with Streamlit & Scikit-learn")
