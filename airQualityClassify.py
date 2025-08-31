# 📌 Step 1: Import Libraries
import pandas as pd

# Load the dataset
file_path = "world_air_quality.csv"   # <-- change to your path if needed
df = pd.read_csv(file_path, sep=";", on_bad_lines="skip")

# Check structure
print("Columns in dataset:\n", df.columns)
print("\nFirst 5 rows:\n", df.head())

# ------------------------------------------------------------

# 📌 Step 2: Pivot pollutants into columns
df_pivot = df.pivot_table(
    index=["Country Code", "City", "Location", "Coordinates", "Last Updated"],
    columns="Pollutant",
    values="Value",
    aggfunc="mean"
).reset_index()

print("\nDataset after pivoting:\n", df_pivot.head())

# ------------------------------------------------------------

# 📌 Step 3: Create AQI category based on PM2.5 values
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

# 📌 Step 4: Save processed dataset
df_pivot.to_csv("air_quality_processed.csv", index=False)
print("\n✅ Preprocessing complete. Saved as 'air_quality_processed.csv'")
