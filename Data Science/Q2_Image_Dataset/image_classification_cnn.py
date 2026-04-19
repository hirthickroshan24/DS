import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import models, layers

# ========== 1. LOAD IMAGES ==========
data = []
labels = []

data_path = "."
for category in os.listdir(data_path):
    category_path = os.path.join(data_path, category)

    # Skip files (only process directories like "cats", "dogs")
    if not os.path.isdir(category_path):
        continue

    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)

        try:
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 128))
            img = cv2.GaussianBlur(img, (5, 5), 0)

            data.append(img)
            labels.append(category)

        except:
            pass

# ========== 2. PREPROCESSING ==========
X = np.array(data) / 255.0  # Normalize pixel values to 0-1
y = np.array(labels)

le = LabelEncoder()
y = le.fit_transform(y)  # Convert labels to numbers (0, 1)

print(f"Total images: {len(X)}")
print(f"Classes: {le.classes_}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== 3. BUILD CNN MODEL ==========
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# ========== 4. TRAIN MODEL ==========
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# ========== 5. EVALUATE ==========
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)
print("Test Loss:", loss)

# ========== 6. PREDICT ON NEW IMAGE ==========
img = cv2.imread("cats/cat_1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (128, 128))
img = img / 255.0

img = np.expand_dims(img, axis=0)

prediction = model.predict(img)

if prediction[0][0] > 0.5:
    print("Dog 🐶")
else:
    print("Cat 🐱")

# ========== INFERENCE ==========
"""
INFERENCE:
1. Images are loaded from directories where each folder name is the class label.
2. Preprocessing: Resize to 128x128, GaussianBlur for noise reduction, normalize to 0-1.
3. LabelEncoder converts text labels (cat/dog) to numbers (0/1).
4. CNN Architecture: Conv2D extracts features, MaxPooling reduces dimensions, Dense layers classify.
5. Binary crossentropy loss is used since this is a 2-class problem.
6. model.fit() trains the CNN, validation_data monitors overfitting.
7. Test accuracy indicates how well the model generalizes to unseen images.
"""