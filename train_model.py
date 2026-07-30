import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------------------------------
# Load Dataset
# ---------------------------------------
df = pd.read_csv("Salary_Data.csv")

# ---------------------------------------
# Features and Target
# ---------------------------------------
X = df[["YearsExperience"]]
y = df["Salary"]

# ---------------------------------------
# Train-Test Split
# ---------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------
# Train Model
# ---------------------------------------
model = LinearRegression()

model.fit(X_train, y_train)

# ---------------------------------------
# Predictions
# ---------------------------------------
y_pred = model.predict(X_test)

# ---------------------------------------
# Model Evaluation
# ---------------------------------------
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("=" * 40)
print("Model Training Completed Successfully")
print("=" * 40)

print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:.2f}")

# ---------------------------------------
# Save Model
# ---------------------------------------
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nmodel.pkl saved successfully.")
