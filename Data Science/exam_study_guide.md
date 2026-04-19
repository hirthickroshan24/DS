# 📚 DS Model Lab Exam — Study Guide (50 Marks, 3 Hours)

## 📁 Final Folder Structure

```
D:\Data Science\
│
├── Q1_Numerical_Dataset\           ← 30 marks (CO1, CO2, CO3)
│   ├── 1_EDA_Visualization\        ← 10 marks
│   │   ├── descriptive_statistics.py   ✅ Your code (mean, median, mode, etc.)
│   │   ├── visualization.py            🆕 NEW (histograms, boxplots, heatmap, etc.)
│   │   └── README.txt
│   │
│   ├── 2_Data_Preprocessing\       ← 10 marks
│   │   ├── handling_missing.py         ✅ Your code (null check, imputation)
│   │   └── README.txt
│   │
│   ├── 3a_Classification\          ← 10 marks (Choose ONE: Classification OR Clustering)
│   │   ├── logistic_regression.py      ✅ Your code
│   │   ├── decision_tree.py            ✅ Your code
│   │   └── README.txt
│   │
│   └── 3b_Clustering\              ← 10 marks (Alternative to Classification)
│       ├── kmeans_clustering.py        ✅ Your code (KMeans + Agglomerative)
│       └── README.txt
│
├── Q2_Image_Dataset\               ← 10 marks (CO4)
│   ├── image_classification_cnn.py     ✅ Your code (CNN - Cat vs Dog)
│   └── README.txt
│
├── Q3_Text_Dataset\                ← 10 marks (CO5)
│   ├── text_analysis.py               🆕 NEW (TF-IDF + Naive Bayes spam detection)
│   └── README.txt
│
└── (original files still here as backup)
```

---

## 📝 Question-wise Breakdown

### Q1: Numerical Dataset (30 marks)

| Component | Marks | Code File | Key Concepts |
|-----------|-------|-----------|--------------|
| **EDA** | 10 | `descriptive_statistics.py` | Mean, Median, Mode, Variance, Std Dev, Skewness, Kurtosis |
| **Visualization** | 10 | `visualization.py` | Histogram, Boxplot, Heatmap, Pairplot, Bar, Line |
| **Preprocessing** | 10 | `handling_missing.py` | Null check, Imputation, Scaling |
| **Classification** | 10 | `logistic_regression.py` / `decision_tree.py` | Train-test split, Accuracy, Confusion Matrix, Classification Report |
| **Clustering** | 10 | `kmeans_clustering.py` | Elbow method, KMeans, Agglomerative, Dendrogram, Silhouette Score |

> [!IMPORTANT]
> For Q1 Part 3, you only need **ONE** of: Classification, Clustering, or Regression. Prepare whichever you're most comfortable with!

### Q2: Image Dataset (10 marks)

| What | Details |
|------|---------|
| **Code** | `image_classification_cnn.py` |
| **Algorithm** | CNN (Convolutional Neural Network) |
| **Dataset** | Cat vs Dog images |
| **Steps** | Load → Resize → Normalize → CNN → Train → Evaluate → Predict |

### Q3: Text Dataset (10 marks)

| What | Details |
|------|---------|
| **Code** | `text_analysis.py` |
| **Algorithm** | Naive Bayes with TF-IDF |
| **Dataset** | SMS Spam Collection (`spam.csv` from Kaggle) |
| **Steps** | Load → Clean text → TF-IDF → Naive Bayes → Evaluate → Predict |

> [!WARNING]
> **Q3 is new code!** You didn't have any text dataset code before. Download the `spam.csv` dataset from [Kaggle SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) before the exam.

---

## ✍️ Inference Writing Tips (Marks depend on this!)

Your exam says: *"We allot marks only for detailed inference reports."*

For **every question**, write inferences covering:

1. **Dataset Description** — Size, features, target variable
2. **What you found** — Patterns, correlations, outliers
3. **Model Performance** — Accuracy, precision, recall, F1-score
4. **Why this approach** — Why you chose the algorithm
5. **Conclusion** — What the results mean practically

> [!TIP]
> Each code file already has an `INFERENCE` section at the bottom with sample inference text. Use those as a template for your answer sheet!

---

## 🎯 Quick Revision Checklist

- [ ] Can you write EDA code from scratch? (describe, info, isnull, corr)
- [ ] Can you create 4+ different plots? (hist, box, heatmap, pair, bar, line)
- [ ] Can you handle missing values? (SimpleImputer)
- [ ] Can you do train-test split and run a classifier?
- [ ] Can you calculate accuracy, confusion matrix, and classification report?
- [ ] Can you set up a CNN for image classification?
- [ ] Can you do text preprocessing and TF-IDF?
- [ ] Can you write detailed inferences for each step?
