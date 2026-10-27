import os
import numpy as np
import pandas as pd

root = os.path.abspath(os.path.dirname(__file__))

# Load your existing Cleveland dataset (already cleaned as heart.csv in data folder)
cleveland = pd.read_csv(os.path.join(root, "data", "heart.csv"))
# Align column naming with .data conversion target
if "target" in cleveland.columns:
    cleveland = cleveland.rename(columns={"target": "num"})

# Helper to load broken-line-space-joined .data files and reshape into 76 columns
def load_uci_data(file_path):
    text = open(file_path, "r").read().strip()
    tokens = text.split()
    if len(tokens) % 76 != 0:
        raise ValueError(f"Unexpected token count in {file_path}: {len(tokens)}")
    raw_table = np.array(tokens).reshape(-1, 76)
    df = pd.DataFrame(raw_table)
    return df

# Load the other three datasets from the data/ subfolder
hungary = load_uci_data(os.path.join(root, "data", "hungarian.data"))
switzerland = load_uci_data(os.path.join(root, "data", "switzerland.data"))
long_beach = load_uci_data(os.path.join(root, "data", "long-beach-va.data"))

# Select only the 14 important columns from the .data files
selected_cols = [2, 3, 8, 9, 11, 15, 18, 31, 37, 39, 40, 43, 50, 57]

hungary = hungary.iloc[:, selected_cols].apply(pd.to_numeric, errors='coerce')
switzerland = switzerland.iloc[:, selected_cols].apply(pd.to_numeric, errors='coerce')
long_beach = long_beach.iloc[:, selected_cols].apply(pd.to_numeric, errors='coerce')

# Rename columns to match Cleveland dataset
columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

for df in [hungary, switzerland, long_beach]:
    df.columns = columns

# Combine all datasets
combined = pd.concat([cleveland, hungary, switzerland, long_beach], axis=0)

# Save final dataset in main folder
combined.to_csv("heart_disease_combined.csv", index=False)

print("✅ Combined dataset created successfully!")
print("Shape:", combined.shape)
print("\nPreview of merged dataset:")
print(combined.head())