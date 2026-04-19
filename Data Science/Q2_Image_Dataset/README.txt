============================================================
  Q2: IMAGE DATASET (10 marks) — CO4
============================================================

FILES:
  1. image_classification_cnn.py  — CNN for Cat vs Dog classification

WHAT TO STUDY:
  - cv2.imread(), cv2.resize(), cv2.cvtColor()
  - cv2.GaussianBlur() — Image preprocessing
  - Normalization: img / 255.0
  - LabelEncoder — Convert labels to numbers
  - CNN Architecture:
      Conv2D → MaxPooling2D → Conv2D → MaxPooling2D → Flatten → Dense → Output
  - model.compile(optimizer='adam', loss='binary_crossentropy')
  - model.fit() — Training
  - model.evaluate() — Testing
  - model.predict() — Prediction on new image

INFERENCE TIPS:
  - Report training accuracy and validation accuracy
  - Discuss loss convergence over epochs
  - Explain CNN layers (convolution extracts features, pooling reduces size)
  - Mention preprocessing steps (resize, blur, normalize)
  - Discuss overfitting if train acc >> val acc
  - Show prediction result on sample image
============================================================
