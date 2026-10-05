import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("dataset/gestures.csv")

# Separate features and labels
X = data.drop("gesture", axis=1)

y = data["gesture"]

print("Dataset size:", len(data))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training model...")

# Create Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print()
print("Accuracy:", accuracy * 100, "%")

print()
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "model/gesture_model.pkl")

print()
print("Model saved successfully!")