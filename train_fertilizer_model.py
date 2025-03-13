import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib

# Set seed for reproducibility
np.random.seed(42)
data_size = 2000  # Increased data size for better generalization

# Generate Synthetic Data with Some Realistic Patterns
nitrogen = np.random.uniform(0, 100, data_size)
phosphorus = np.random.uniform(0, 100, data_size)
potassium = np.random.uniform(0, 100, data_size)
ph = np.random.uniform(4.5, 8.5, data_size)
moisture = np.random.uniform(10, 80, data_size)

# Fertilizer logic: Assign fertilizers based on rough nutrient needs
fertilizer_conditions = [
    (nitrogen > 60) & (phosphorus < 40),  # High nitrogen, low phosphorus → Urea
    (phosphorus > 50) & (potassium > 30),  # High P & K → DAP
    (potassium > 60),  # High K → MOP
    (ph < 6) & (moisture > 50),  # Acidic soil & high moisture → Compost
    (ph.between(6, 7)) & (moisture.between(30, 50))  # Neutral pH & medium moisture → NPK
]

fertilizer_choices = ["Urea", "DAP", "MOP", "Compost", "NPK"]
fertilizer = np.select(fertilizer_conditions, fertilizer_choices, default="Urea")  # Default to Urea

# Create DataFrame
df = pd.DataFrame({
    "Nitrogen": nitrogen,
    "Phosphorus": phosphorus,
    "Potassium": potassium,
    "pH": ph,
    "Moisture": moisture,
    "Fertilizer": fertilizer
})

# Convert Fertilizer Labels to Numeric
fertilizer_mapping = {name: idx for idx, name in enumerate(fertilizer_choices)}
df["Fertilizer"] = df["Fertilizer"].map(fertilizer_mapping)

# Prepare Data
X = df.drop(columns=["Fertilizer"])
y = df["Fertilizer"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **Hyperparameter Tuning for Higher Accuracy**
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best model
best_rf = grid_search.best_estimator_

# Save the Optimized Model
joblib.dump(best_rf, "fertilizer_model.pkl")

print("✅ Model training complete! Best parameters found:", grid_search.best_params_)
