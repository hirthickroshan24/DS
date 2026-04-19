============================================================
  Q1 - PART 3c: REGRESSION (10 marks) — CO3
============================================================

FILES:
  1. linear_regression.py  — Linear Regression (predicts continuous AQI)

KEY DIFFERENCE FROM CLASSIFICATION:
  - Classification → predicts CATEGORIES (0 or 1, spam or ham)
  - Regression → predicts CONTINUOUS VALUES (AQI = 150.3, price = 2500)

WHAT TO STUDY:
  - LinearRegression()
  - train_test_split(X, y, test_size=0.2)
  - Metrics (VERY DIFFERENT from classification!):
      R² Score      — How well model explains variance (0 to 1)
      MAE           — Average absolute error
      MSE           — Average squared error
      RMSE          — Square root of MSE
  - model.coef_       — Feature importance (coefficients)
  - model.intercept_  — y-intercept of the line
  - Actual vs Predicted plot
  - Residual plot

INFERENCE TIPS:
  - Report R² score (closer to 1 = better fit)
  - Report RMSE in context ("average error of X AQI units")
  - Discuss which features have highest coefficients
  - Comment on residual plot (random = good, pattern = bad)
  - Compare with classification: here we predict VALUES, not labels

NOTE: Logistic Regression is CLASSIFICATION (despite the name!)
      Linear Regression is REGRESSION
============================================================
