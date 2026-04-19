============================================================
  Q3: TEXT DATASET (10 marks) — CO5
============================================================

FILES:
  1. text_analysis.py  — SMS Spam Classification using TF-IDF + Naive Bayes

DATASET:
  - Download "spam.csv" from Kaggle (SMS Spam Collection Dataset)
  - URL: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

WHAT TO STUDY:
  - Text Preprocessing:
      .str.lower() — Convert to lowercase
      re.sub(r'[^a-z\s]', '', text) — Remove special characters
      Removing extra whitespace
  - TfidfVectorizer — Converts text to numerical TF-IDF features
      TF = Term Frequency, IDF = Inverse Document Frequency
  - MultinomialNB — Naive Bayes classifier (perfect for text)
  - accuracy_score, confusion_matrix, classification_report
  - Visualization: Bar chart (label distribution), Histogram (word count)

INFERENCE TIPS:
  - Report accuracy of spam detection
  - Explain TF-IDF (words appearing in few docs get higher weight)
  - Explain Naive Bayes (probabilistic classifier, assumes independence)
  - Discuss precision & recall (important: false positives = legit msg marked spam)
  - Mention preprocessing impact on accuracy
  - Show custom text prediction results
============================================================
