import os
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# 1. Generate synthetic "Heart Disease" data for demonstration
# Features: Age, Cholesterol, RestingBP, MaxHeartRate
X, y = make_classification(
    n_samples=1000, 
    n_features=4, 
    n_informative=3, 
    n_redundant=0, 
    random_state=42,
    weights=[0.7, 0.3] # Simulating class imbalance (70% healthy, 30% risk)
)

feature_names = ['age', 'cholesterol', 'resting_bp', 'max_heart_rate']
df = pd.DataFrame(X, columns=feature_names)

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

# 3. Train a basic Random Forest model
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 5. Save the model
os.makedirs('../model', exist_ok=True)
model_path = '../model/disease_risk_model.joblib'
joblib.dump(model, model_path)
print(f"Model saved successfully to {model_path}")