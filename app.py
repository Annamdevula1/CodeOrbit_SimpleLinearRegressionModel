import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------
df = pd.read_csv("Salary_Data.csv")

# Features and Target
X = df[['YearsExperience']]
y = df['Salary']

# Train-Test Split (same as notebook)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -------------------------------------------------
# LOAD TRAINED MODEL
# -------------------------------------------------
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Predictions
y_pred = model.predict(X_test)

# Model Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Dataset Information
rows, cols = df.shape
missing_values = df.isnull().sum().sum()

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------
st.markdown("""
<style>

.main{
    background-color:#f8fafc;
}

section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#1e3c72,#2a5298);
}

section[data-testid="stSidebar"] *{
    color:white;
}

.title-box{
    background:linear-gradient(90deg,#0f4c81,#3b82f6);
    padding:22px;
    border-radius:12px;
    text-align:center;
    color:white;
    box-shadow:0px 4px 12px rgba(0,0,0,0.25);
}

.card{
    background:white;
    padding:18px;
    border-radius:12px;
    box-shadow:0px 3px 10px rgba(0,0,0,0.12);
    margin-bottom:15px;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:30px;
}

.stButton>button{
    background:#2563eb;
    color:white;
    border:none;
    border-radius:8px;
    padding:0.6em 1.2em;
    font-size:16px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#1d4ed8;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("📂 Navigation")

page = st.sidebar.radio(
    "Select a Page",
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
# -------------------------------------------------
# HOME PAGE
# -------------------------------------------------

if page == "🏠 Home":

    # Header
    st.markdown("""
    <div class="title-box">
        <h1>💼 Salary Prediction using Simple Linear Regression</h1>
        <p>Predict Employee Salary based on Years of Experience</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # Project Overview
    st.markdown("## 📖 Project Overview")

    st.markdown("""
    <div class="card">
    This Machine Learning project predicts an employee's salary based on their
    <b>Years of Experience</b> using the <b>Simple Linear Regression</b> algorithm.
    The model is trained on historical salary data and provides an estimated salary
    for a given experience value.
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📊 Total Records",
            value=rows
        )

    with col2:
        st.metric(
            label="📑 Features",
            value=cols-1
        )

    with col3:
        st.metric(
            label="🤖 Algorithm",
            value="Linear Regression"
        )

    with col4:
        st.metric(
            label="📈 R² Score",
            value=f"{r2:.2f}"
        )

    st.write("")

    # Dataset Information
    left, right = st.columns([2,1])

    with left:

        st.markdown("## 📂 Dataset Information")

        st.markdown("""
        <div class="card">

        <b>Dataset Name</b><br>
        Salary_Data.csv

        <br><br>

        <b>Input Feature</b><br>
        ✔ YearsExperience

        <br><br>

        <b>Target Variable</b><br>
        ✔ Salary

        <br><br>

        <b>Machine Learning Algorithm</b><br>
        ✔ Simple Linear Regression

        </div>
        """, unsafe_allow_html=True)

    with right:

        st.markdown("## ⚡ Quick Facts")

        st.info(f"""
📊 Dataset Rows : {rows}

📑 Columns : {cols}

❌ Missing Values : {missing_values}

📈 R² Score : {r2:.2f}

📉 MAE : {mae:.2f}
""")

    st.write("")

    # Project Workflow

    st.markdown("## 🔄 Project Workflow")

    st.markdown("""
    <div class="card">

    📁 Load Salary Dataset

    ⬇

    🧹 Data Preprocessing

    ⬇

    ✂ Train-Test Split (80:20)

    ⬇

    🤖 Train Linear Regression Model

    ⬇

    📈 Predict Salary

    ⬇

    📊 Evaluate Model Performance

    ⬇

    🚀 Deploy using Streamlit

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # Libraries Used

    st.markdown("## 🛠 Technologies Used")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.success("""
✔ Python

✔ Pandas

✔ NumPy
""")

    with c2:
        st.success("""
✔ Scikit-learn

✔ Matplotlib

✔ Pickle
""")

    with c3:
        st.success("""
✔ Streamlit

✔ VS Code / Google Colab

✔ GitHub
""")

    st.write("")

    # Expandable Section

    with st.expander("📌 Click to Know More About This Project"):

        st.write("""
This application demonstrates a Simple Linear Regression model that predicts
salary based on years of experience.

The project includes:

• Salary Prediction

• Data Visualization

• Model Performance

• Dataset Statistics

• Developer Information

The application is developed using Streamlit and Machine Learning.
""")
# -------------------------------------------------
# SALARY PREDICTION PAGE
# -------------------------------------------------

elif page == "💼 Salary Prediction":

    st.markdown("""
    <div class="title-box">
        <h1>💼 Salary Prediction</h1>
        <p>Predict Salary using Years of Experience</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    left, right = st.columns([1,1])

    # ---------------------------
    # INPUT SECTION
    # ---------------------------

    with left:

        st.markdown("## 📝 Enter Details")

        years = st.number_input(
            "Years of Experience",
            min_value=0.0,
            max_value=50.0,
            value=1.0,
            step=0.1,
            help="Enter total years of work experience."
        )

        predict = st.button(
            "🔮 Predict Salary",
            use_container_width=True
        )

    # ---------------------------
    # PREDICTION SECTION
    # ---------------------------

    with right:

        st.markdown("## 💰 Prediction Result")

        if predict:

            with st.spinner("Predicting Salary... Please wait"):

                salary = model.predict([[years]])[0]

            st.success("✅ Salary Predicted Successfully!")

            st.metric(
                label="Predicted Salary",
                value=f"₹ {salary:,.2f}"
            )

            st.balloons()

            st.info(f"""
**Prediction Summary**

• Years of Experience : **{years:.1f}**

• Estimated Salary : **₹ {salary:,.2f}**

• Model : **Linear Regression**
""")

    st.write("")

    # ---------------------------
    # QUICK INFORMATION
    # ---------------------------

    st.markdown("## 📌 Prediction Information")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Algorithm",
            "Linear Regression"
        )

    with c2:
        st.metric(
            "Input Feature",
            "YearsExperience"
        )

    with c3:
        st.metric(
            "Output",
            "Salary"
        )

    st.write("")

    with st.expander("ℹ️ How is the salary predicted?"):

        st.write("""
The model predicts salary using **Simple Linear Regression**.

### Steps:
1. Enter the Years of Experience.
2. Click **Predict Salary**.
3. The trained model estimates the salary.
4. The predicted salary is displayed instantly.

The prediction is based on the relationship learned from the Salary_Data.csv dataset.
""")
# -------------------------------------------------
# DATA VISUALIZATION PAGE
# -------------------------------------------------

elif page == "📈 Data Visualization":

    st.markdown("""
    <div class="title-box">
        <h1>📈 Data Visualization</h1>
        <p>Visual Analysis of Salary Dataset</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # User input for predicted point
    years = st.number_input(
        "Enter Years of Experience to Highlight",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.1,
        key="graph_years"
    )

    predicted_salary = model.predict([[years]])[0]

    # ----------------------------
    # Scatter Plot + Regression Line
    # ----------------------------

    st.subheader("📊 Scatter Plot with Regression Line")

    fig, ax = plt.subplots(figsize=(9,6))

    # Actual Data
    ax.scatter(
        df["YearsExperience"],
        df["Salary"],
        color="royalblue",
        s=70,
        label="Actual Data"
    )

    # Regression Line
    ax.plot(
        df["YearsExperience"],
        model.predict(df[["YearsExperience"]]),
        color="red",
        linewidth=3,
        label="Regression Line"
    )

    # Predicted Point
    ax.scatter(
        years,
        predicted_salary,
        color="green",
        s=180,
        marker="*",
        label="Predicted Salary"
    )

    ax.set_title("Salary vs Years of Experience")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary")
    ax.legend()

    st.pyplot(fig, use_container_width=True)

    st.success(
        f"Predicted Salary for {years:.1f} Years Experience = ₹ {predicted_salary:,.2f}"
    )

    st.write("")

    # ----------------------------
    # Dataset Preview
    # ----------------------------

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df,
        use_container_width=True,
        height=250
    )

    st.write("")

    # ----------------------------
    # Dataset Statistics
    # ----------------------------

    st.subheader("📊 Dataset Summary")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.write("")

    # ----------------------------
    # Additional Information
    # ----------------------------

    with st.expander("ℹ️ Interpretation of Graph"):

        st.write("""
### Scatter Plot
- Blue dots represent the actual salary records.
- Each point corresponds to one employee.

### Regression Line
- The red line is the best-fit line created by the Linear Regression model.
- It shows the relationship between Years of Experience and Salary.

### Predicted Point
- The green star represents the salary predicted by the model based on the entered years of experience.

### Observation
- As Years of Experience increases, Salary also increases.
- This indicates a strong positive linear relationship between the two variables.
""")
# -------------------------------------------------
# MODEL INFORMATION
# -------------------------------------------------

elif page == "🤖 Model Information":

    st.markdown("""
    <div class="title-box">
        <h1>🤖 Model Information</h1>
        <p>Performance of the Trained Machine Learning Model</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("🤖 Algorithm", "Linear Regression")
        st.metric("📈 R² Score", f"{r2:.4f}")
        st.metric("📉 Mean Absolute Error", f"{mae:.2f}")

    with c2:
        st.metric("📂 Train-Test Split", "80 : 20")
        st.metric("📊 Training Samples", len(X_train))
        st.metric("📋 Testing Samples", len(X_test))

    st.info("""
The Linear Regression model learns the relationship between
Years of Experience and Salary and predicts salary for new values.
""")

    with st.expander("📘 Model Explanation"):
        st.write("""
**Algorithm:** Linear Regression

**Input Feature**
- YearsExperience

**Target Variable**
- Salary

**Evaluation Metrics**
- R² Score
- Mean Absolute Error (MAE)

The model is simple, fast and suitable for predicting salary
from years of experience.
""")


# -------------------------------------------------
# DATASET STATISTICS
# -------------------------------------------------

elif page == "📊 Dataset Statistics":

    st.markdown("""
    <div class="title-box">
        <h1>📊 Dataset Statistics</h1>
        <p>Summary of Salary Dataset</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    a, b, c, d = st.columns(4)

    a.metric("Total Records", rows)
    b.metric("Total Features", cols-1)
    c.metric("Target Column", 1)
    d.metric("Missing Values", missing_values)

    st.write("")

    st.subheader("Dataset Preview")
    st.dataframe(df, use_container_width=True)

    st.write("")

    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

    st.write("")

    st.subheader("Missing Values")

    st.dataframe(df.isnull().sum().to_frame("Missing Values"))

    with st.expander("Dataset Information"):
        st.write(f"""
Rows : **{rows}**

Columns : **{cols}**

Features : **YearsExperience**

Target : **Salary**
""")


# -------------------------------------------------
# ABOUT PROJECT
# -------------------------------------------------

elif page == "ℹ️ About Project":

    st.markdown("""
    <div class="title-box">
        <h1>ℹ️ About Project</h1>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
### 📌 Project Title

Salary Prediction using Simple Linear Regression

### 📂 Dataset

Salary_Data.csv

### 🎯 Objective

Predict employee salary based on years of experience using Machine Learning.

### 📥 Input Feature

- YearsExperience

### 📤 Output

- Salary

### 🤖 Machine Learning Algorithm

- Linear Regression

### 🛠 Libraries Used

- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Pickle

### 🚀 Deployment

- Streamlit
""")

    with st.expander("Project Workflow"):
        st.write("""
1. Load Dataset

2. Data Preprocessing

3. Train-Test Split

4. Train Linear Regression Model

5. Predict Salary

6. Evaluate Model

7. Deploy using Streamlit
""")


# -------------------------------------------------
# DEVELOPER
# -------------------------------------------------

elif page == "👨‍💻 Developer":

    st.markdown("""
    <div class="title-box">
        <h1>👨‍💻 Developer</h1>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
### 👤 Name

Durga Prasad Annamdevula

### 🎓 Qualification

BCA Graduate

### 💡 Skills

- Python
- Machine Learning
- Data Science
- Artificial Intelligence
- Cloud Computing

### 🔗 GitHub

https://github.com/Annamdevula1

### 💼 LinkedIn

(Add your LinkedIn Profile URL)

### 📧 Email

(Add your Email Address)
""")

    st.success("Thank you for visiting the Salary Prediction Application!")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.markdown("---")

st.markdown(
"""
<div class='footer'>
Developed by <b>Durga Prasad Annamdevula</b> ❤️ <br>
Salary Prediction using Simple Linear Regression <br>
Powered by Streamlit & Scikit-learn
</div>
""",
unsafe_allow_html=True
)
