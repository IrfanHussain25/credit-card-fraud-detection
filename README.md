# Explainable Credit Card Fraud Detection

Machine learning project for detecting fraudulent credit card transactions and explaining model predictions using **SHAP**.

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Random Forest
* XGBoost
* Logistic Regression
* SMOTE
* SHAP
* Matplotlib, Seaborn

## Approach

* Preprocessed and engineered transaction features
* Handled class imbalance using **SMOTE**
* Compared Logistic Regression, Random Forest, and XGBoost
* Evaluated models using Precision, Recall, F1-score, and ROC-AUC
* Used **SHAP** to explain feature importance and individual predictions

## Results

* **XGBoost ROC-AUC:** 99.64%
* **XGBoost Recall:** 86.34%
* **Random Forest F1-Score:** 79.39%

## Key Highlight

The project combines **fraud detection with Explainable AI (XAI)**, helping understand why a transaction is classified as fraudulent.
