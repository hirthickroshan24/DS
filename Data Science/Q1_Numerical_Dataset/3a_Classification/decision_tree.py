import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("delhi_air.csv")

df['target'] = (df['PM2.5'] > 50).astype(int)
X = df.drop(['target', 'PM2.5'], axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # FIX: test_size, not train_size
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ========== INFERENCE ==========
"""
INFERENCE:
1. Decision Tree Classifier splits data based on feature thresholds.
2. max_depth=5 limits tree depth to prevent overfitting.
3. Target: PM2.5 > 50 means poor air quality (1), else good (0).
4. Accuracy, Confusion Matrix, and Classification Report evaluate performance.
5. Decision Trees are easy to interpret and visualize.
"""
