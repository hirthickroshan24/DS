============================================================
  Q1 - PART 2: DATA PREPROCESSING (10 marks) — CO2
============================================================

FILES:
  1. handling_missing.py  — Missing value detection & imputation

WHAT TO STUDY:
  - df.isnull().sum() — Check for missing values
  - SimpleImputer(strategy='mean'/'median'/'most_frequent')
  - StandardScaler — Standardization (mean=0, std=1)
  - MinMaxScaler — Normalization (0 to 1)
  - Handling duplicates: df.drop_duplicates()
  - Encoding categorical: LabelEncoder, OneHotEncoder
  - Feature selection: df.drop(), df.corr()

INFERENCE TIPS:
  - State how many null values were found and in which columns
  - Explain why you chose mean/median/mode for imputation
  - Explain why scaling is needed (different ranges of features)
  - Mention if duplicates were found and removed
============================================================
