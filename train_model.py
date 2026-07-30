import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv("Salary_Data.csv")

# Features and target
X = df[["YearsExperience"]]
y = df["Salary"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "salary_prediction_model.pkl")

print("Model trained successfully!")
print("Model saved as salary_prediction_model.pkl")
