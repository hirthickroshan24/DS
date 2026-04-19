import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("delhi_air.csv")

def get_aqi_category(pm25):
    if pm25 <= 50:
        return 0
    else:
        return 1

df['target'] = df['PM2.5'].apply(get_aqi_category)
X = df.drop(['target', 'PM2.5'], axis=1)  # FIX: Also drop PM2.5 to avoid data leakage
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # FIX: test_size, not train_size
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ========== INFERENCE ==========
"""
INFERENCE:
1. Logistic Regression is used for binary classification (0 = Good AQI, 1 = Poor AQI).
2. PM2.5 is dropped from features since the target is derived from it (avoids data leakage).
3. 80% data is used for training and 20% for testing.
4. Accuracy tells overall correct predictions.
5. Confusion Matrix shows TP, TN, FP, FN counts.
6. Classification Report gives precision, recall, and F1-score per class.
"""
