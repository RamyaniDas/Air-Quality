#Week 1
#Importing Libraries
import pandas as pd

# Loading the dataset
file_path = "world_air_quality.csv"   # <-- change to your path if needed
df = pd.read_csv(file_path, sep=";", on_bad_lines="skip")

# Checking structure
print("Columns in dataset:\n", df.columns)
print("\nFirst 5 rows:\n", df.head())

# ------------------------------------------------------------

#Pivoting pollutants into columns
df_pivot = df.pivot_table(
    index=["Country Code", "City", "Location", "Coordinates", "Last Updated"],
    columns="Pollutant",
    values="Value",
    aggfunc="mean"
).reset_index()

print("\nDataset after pivoting:\n", df_pivot.head())

# ------------------------------------------------------------

#Creating AQI category based on PM2.5 values
def categorize_air_quality(pm25):
    if pd.isna(pm25):
        return "Unknown"
    elif pm25 <= 50:
        return "Good"
    elif pm25 <= 100:
        return "Moderate"
    elif pm25 <= 150:
        return "Unhealthy for Sensitive"
    elif pm25 <= 200:
        return "Unhealthy"
    elif pm25 <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

if "PM2.5" in df_pivot.columns:
    df_pivot["AQI_Category"] = df_pivot["PM2.5"].apply(categorize_air_quality)
else:
    df_pivot["AQI_Category"] = "Unknown"

print("\nDataset with AQI Category:\n", df_pivot[["City", "PM2.5", "AQI_Category"]].head())

# ------------------------------------------------------------

#Saving processed dataset
df_pivot.to_csv("air_quality_processed.csv", index=False)
print("\n Preprocessing complete. Saved as 'air_quality_processed.csv'")

#Week 2
# =========================================================
# STEP 5: Exploratory Data Analysis (EDA)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Checking missing values
print("\nMissing values in dataset:\n", df_pivot.isnull().sum())

# Filling missing values with median (safer than mean)
df_pivot.fillna(df_pivot.median(numeric_only=True), inplace=True)

# Distribution of PM2.5
plt.figure(figsize=(6,4))
sns.histplot(df_pivot["PM2.5"], bins=30, kde=True, color="blue")
plt.title("Distribution of PM2.5 Levels")
plt.xlabel("PM2.5 Value")
plt.ylabel("Frequency")
plt.show()

# Count of AQI categories
plt.figure(figsize=(6,4))
sns.countplot(x="AQI_Category", data=df_pivot, palette="Set2")
plt.title("Air Quality Categories Count")
plt.xticks(rotation=30)
plt.show()

# Country-wise PM2.5 comparison
plt.figure(figsize=(10,5))
df_pivot.groupby("Country Code")["PM2.5"].mean().sort_values(ascending=False).head(15).plot(kind="bar", color="orange")
plt.title("Top 15 Countries by Average PM2.5")
plt.ylabel("PM2.5 Value")
plt.show()

# =========================================================
# STEP 6: Feature Engineering
# =========================================================
# Selecting feature columns (pollutants only)
pollutants = ["PM2.5", "PM10", "NO2", "O3", "CO", "SO2"]
X = df_pivot[pollutants]

# Encode target
le = LabelEncoder()
y = le.fit_transform(df_pivot["AQI_Category"])

# Train-test split (70-30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# =========================================================
# STEP 7: Baseline Models
# =========================================================

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000, solver='lbfgs')
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

print("\n--- Logistic Regression Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log, target_names=le.classes_, zero_division=0))

# Decision Tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("\n--- Decision Tree Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt, target_names=le.classes_))

# Confusion Matrix Plot (Decision Tree)
cm = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix (Decision Tree)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#Week 3
# =========================================================
# STEP 8: Advanced Models
# =========================================================
from sklearn.ensemble import RandomForestClassifier

# Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n--- Random Forest Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf, target_names=le.classes_))


# =========================================================
# STEP 9: Hyperparameter Tuning
# =========================================================
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42),
                    param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print("\n--- Best Random Forest Model ---")
print("Best Params:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)

best_rf = grid.best_estimator_
y_pred_best = best_rf.predict(X_test)

print("Tuned RF Accuracy:", accuracy_score(y_test, y_pred_best))


# =========================================================
# STEP 10: Feature Importance
# =========================================================


importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10,5))
plt.title("Feature Importance (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.show()


# =========================================================
# STEP 11: Saving the Model
# =========================================================
import joblib

joblib.dump(best_rf, "air_quality_model.pkl")
print("\n Final model saved as 'air_quality_model.pkl'")

# =========================================================
# STEP 12: Predicting Air Quality with User Input
# =========================================================

import joblib

# Loading the saved model & label encoder
best_rf = joblib.load("air_quality_model.pkl")

def predict_air_quality(pm25, pm10, no2, co, o3, so2):
    new_data = pd.DataFrame([[pm25, pm10, no2, co, o3, so2]],
                            columns=X.columns)
    prediction = best_rf.predict(new_data)
    predicted_label = le.inverse_transform(prediction)
    return predicted_label[0]

print("\n--- Air Quality Prediction System ---")

# Asking user for input
pm25 = float(input("Enter PM2.5 value: "))
pm10 = float(input("Enter PM10 value: "))
no2 = float(input("Enter NO2 value: "))
co   = float(input("Enter CO value: "))
o3   = float(input("Enter O3 value: "))
so2  = float(input("Enter SO2 value: "))

# Predict and show result
result = predict_air_quality(pm25, pm10, no2, co, o3, so2)
print("\n Predicted Air Quality Category:", result)

