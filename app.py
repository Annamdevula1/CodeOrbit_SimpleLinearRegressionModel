import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import r2_score, mean_absolute_error

# -----------------------------------------------------
# Page Configuration
# -----------------------------------------------------
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💼",
    layout="wide"
)

# -----------------------------------------------------
# Load Dataset
# -----------------------------------------------------
df = pd.read_csv("Salary_Data.csv")

# Load Trained Model
model = joblib.load("salary_model.pkl")

# -----------------------------------------------------
# Model Evaluation
# -----------------------------------------------------
X = df[["YearsExperience"]]
y = df["Salary"]

predictions = model.predict(X)

r2 = r2_score(y, predictions)
mae = mean_absolute_error(y, predictions)

# -----------------------------------------------------
# Sidebar
# -----------------------------------------------------
st.sidebar.title("📂 Navigation")

page = st.sidebar.radio(
    "Select a Page",
    (
        "🏠 Home",
        "💼 Salary Prediction",
        "📈 Data Visualization",
        "🤖 Model Information",
        "📊 Dataset Statistics",
        "ℹ️ About Project",
        "👨‍💻 Developer"
    )
)

st.sidebar.markdown("---")
st.sidebar.success("Salary Prediction using Linear Regression")

# =====================================================
# HOME PAGE
# =====================================================
if page == "🏠 Home":
    st.image("Banner image.jpeg", width="stretch")
    st.title("💼 Salary Prediction using Simple Linear Regression")

    st.write("""
Welcome to the **Salary Prediction Web Application**.

This application predicts salary based on
**Years of Experience** using a trained
**Simple Linear Regression** model.
""")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 Project Overview")

        st.write("""
✔ Salary Prediction

✔ Machine Learning Model

✔ Data Visualization

✔ Model Performance

✔ Dataset Statistics
""")

    with col2:
        st.subheader("📂 Dataset Information")

        st.write(f"**Dataset Name:** Salary_Data.csv")
        st.write(f"**Rows:** {df.shape[0]}")
        st.write(f"**Columns:** {df.shape[1]}")
        st.write("**Input:** YearsExperience")
        st.write("**Output:** Salary")

    st.markdown("---")

    st.subheader("📄 Dataset Preview")

    st.dataframe(df.head(10), width="stretch")

    with st.expander("📖 Project Description"):

        st.write("""
This project uses **Simple Linear Regression**
to predict salary based on years of experience.

""")
# =====================================================
# SALARY PREDICTION PAGE
# =====================================================

elif page == "💼 Salary Prediction":

    st.title("💼 Salary Prediction")

    st.write("Enter your years of experience to predict your estimated salary.")

    experience = st.number_input(
        "Years of Experience",
        min_value=0.0,
        max_value=50.0,
        value=1.0,
        step=0.1
    )

    if st.button("🚀 Predict Salary"):

        with st.spinner("Predicting Salary..."):

            
            input_data = pd.DataFrame({
                  "YearsExperience": [experience]
          })

            salary = model.predict(input_data)[0]

        st.success("Prediction Completed Successfully!")

        st.metric(
            label="💰 Predicted Salary",
            value=f"₹ {salary:,.2f}"
        )

        st.balloons()

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.info(f"**Experience:** {experience} Years")

        with col2:
            st.success(f"**Predicted Salary:** ₹ {salary:,.2f}")

        st.markdown("---")

        st.subheader("📈 Prediction Visualization")

        fig, ax = plt.subplots(figsize=(9,5))

        # Scatter Plot
        ax.scatter(
            df["YearsExperience"],
            df["Salary"],
            label="Actual Data",
            s=60
        )

        # Regression Line
        ax.plot(
            df["YearsExperience"],
            predictions,
            linewidth=3,
            label="Regression Line"
        )

        # Predicted Point
        ax.scatter(
            experience,
            salary,
            color="red",
            marker="*",
            s=300,
            label="Your Prediction"
        )

        ax.set_xlabel("Years of Experience")
        ax.set_ylabel("Salary")
        ax.set_title("Salary Prediction using Linear Regression")
        ax.legend()

        st.pyplot(fig)

        st.markdown("---")

        with st.expander("📄 Show Dataset"):
             st.dataframe(df, width="stretch")
# =====================================================
# DATA VISUALIZATION PAGE
# =====================================================

elif page == "📈 Data Visualization":

    st.title("📈 Data Visualization")

    st.subheader("Scatter Plot")

    fig, ax = plt.subplots(figsize=(9,5))

    ax.scatter(
        df["YearsExperience"],
        df["Salary"],
        s=60,
        label="Actual Data"
    )

    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary")
    ax.set_title("Salary Dataset")

    ax.legend()

    st.pyplot(fig)

    st.markdown("---")

    st.subheader("Regression Line")

    fig2, ax2 = plt.subplots(figsize=(9,5))

    ax2.scatter(
        df["YearsExperience"],
        df["Salary"],
        s=60,
        label="Actual Data"
    )

    ax2.plot(
        df["YearsExperience"],
        predictions,
        linewidth=3,
        label="Regression Line"
    )

    ax2.set_xlabel("Years of Experience")
    ax2.set_ylabel("Salary")
    ax2.set_title("Linear Regression Model")

    ax2.legend()

    st.pyplot(fig2)

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(df,width="stretch")


# =====================================================
# DATASET STATISTICS PAGE
# =====================================================

elif page == "📊 Dataset Statistics":

    st.title("📊 Dataset Statistics")

    col1, col2 = st.columns(2)

    with col1:

        st.metric("Total Records", df.shape[0])

        st.metric("Total Features", df.shape[1])

    with col2:

        st.metric(
            "Missing Values",
            df.isnull().sum().sum()
        )

        st.metric(
            "Duplicate Records",
            df.duplicated().sum()
        )

    st.markdown("---")

    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Dataset Information")

    info = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str)
    })

    st.dataframe(
        info,
        use_container_width=True
    )
# =====================================================
# MODEL INFORMATION PAGE
# =====================================================

elif page == "🤖 Model Information":

    st.title("🤖 Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Algorithm", "Simple Linear Regression")
        st.metric("R² Score", f"{r2:.4f}")

    with col2:
        st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
        st.metric("Train-Test Split", "80 : 20")

    st.markdown("---")

    st.subheader("📌 Model Summary")

    st.info("""
• Algorithm : Simple Linear Regression

• Input Feature : YearsExperience

• Target Variable : Salary

• The model predicts salary based on years of experience.

• Higher R² Score indicates better model performance.
""")

    st.markdown("---")

    st.subheader("📊 Performance Metrics")

    st.write(f"✅ R² Score : **{r2:.4f}**")

    st.write(f"✅ Mean Absolute Error : **{mae:.2f}**")

    st.success("Model trained successfully using Scikit-learn.")


# =====================================================
# ABOUT PROJECT PAGE
# =====================================================

elif page == "ℹ️ About Project":

    st.title("ℹ️ About Project")

    st.write("""
This project predicts employee salary based on
Years of Experience using
Simple Linear Regression.
""")

    st.markdown("---")

    st.subheader("📂 Dataset Details")

    st.write("**Dataset Name :** Salary_Data.csv")
    st.write(f"**Rows :** {df.shape[0]}")
    st.write(f"**Columns :** {df.shape[1]}")

    st.markdown("---")

    st.subheader("📌  Workflow")

    st.write("""
1️⃣ Problem Definition

⬇️

2️⃣ Data Collection

⬇️

3️⃣ Data Preprocessing

⬇️

4️⃣ Exploratory Data Analysis (EDA)

⬇️

5️⃣ Feature Selection

⬇️

6️⃣ Train-Test Split

⬇️

7️⃣ Model Building (Linear Regression)

⬇️

8️⃣ Model Evaluation

⬇️

9️⃣ Salary Prediction

⬇️

🔟 Data Visualization

⬇️

1️⃣1️⃣ Streamlit Web Application

⬇️

1️⃣2️⃣ Deployment
""")

    st.markdown("---")

    st.subheader("🛠 Libraries Used")

    st.write("""
✅ Pandas

✅ NumPy

✅ Matplotlib

✅ Scikit-learn

✅ Streamlit

✅ Joblib
""")


# =====================================================
# DEVELOPER PAGE
# =====================================================

elif page == "👨‍💻 Developer":

    st.title("👨‍💻 Developer")

    st.success("Thank you for visiting this project!")

    st.markdown("---")

    st.subheader("Developer Details")

    st.write("**Name :**DURGA PRASAD ANNAMDEVULA")

    st.write("**Project :** Salary Prediction using Simple Linear Regression")

    st.write("**Role :** Data Science & Machine Learning Student")

    st.markdown("---")

    st.subheader("Connect With Me")

    st.write("📧 Email :durgaprasadannamdevula41@gmail.com")

    st.write("🐙 GitHub : https://github.com/Annamdevula1/Simple-Linear-Regression-Task-3.git")

    st.write("💼 LinkedIn :https://www.linkedin.com/in/durga-prasad-annamdevula-232538341")

    st.markdown("---")

    st.caption("© 2026 | Salary Prediction using Linear Regression | Built with Streamlit")
