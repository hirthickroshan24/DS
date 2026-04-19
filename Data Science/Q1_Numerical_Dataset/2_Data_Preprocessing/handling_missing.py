import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.read_csv("delhi_air.csv")

# ========== 1. CHECK MISSING VALUES ==========
print("Missing Values:\n", df.isnull().sum())
print("\nTotal Missing:", df.isnull().sum().sum())

# ========== 2. HANDLE MISSING VALUES (Imputation) ==========
# If there are null values, fill them with mean
imputer = SimpleImputer(strategy='mean')
df[['PM2.5', 'PM10']] = imputer.fit_transform(df[['PM2.5', 'PM10']])

print("\nAfter Imputation:\n", df.isnull().sum())

# ========== 3. CHECK DUPLICATES ==========
print("\nDuplicate Rows:", df.duplicated().sum())
df = df.drop_duplicates()
print("After Removing Duplicates:", df.shape)

# ========== 4. STANDARDIZATION (StandardScaler) ==========
# Converts to mean=0, std=1
numerical_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'AQI']

std_scaler = StandardScaler()
df_standardized = pd.DataFrame(
    std_scaler.fit_transform(df[numerical_cols]),
    columns=numerical_cols
)
print("\nAfter StandardScaler (first 5 rows):\n", df_standardized.head())

# ========== 5. NORMALIZATION (MinMaxScaler) ==========
# Converts to range 0-1
mm_scaler = MinMaxScaler()
df_normalized = pd.DataFrame(
    mm_scaler.fit_transform(df[numerical_cols]),
    columns=numerical_cols
)
print("\nAfter MinMaxScaler (first 5 rows):\n", df_normalized.head())

# ========== INFERENCE ==========
"""
INFERENCE:
1. Missing values were checked using df.isnull().sum().
2. SimpleImputer with strategy='mean' fills null values with column mean.
3. Duplicate rows were identified and removed using drop_duplicates().
4. StandardScaler standardizes features (mean=0, std=1) — useful for algorithms like SVM, KNN.
5. MinMaxScaler normalizes features to 0-1 range — useful for Neural Networks, Clustering.
6. Preprocessing is essential before model training to ensure data quality.
"""