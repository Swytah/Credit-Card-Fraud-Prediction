# Credit-Card-Fraud-Prediction

This project builds a machine learning system to detect fraudulent credit card transactions using advanced classification techniques.  
It combines **exploratory data analysis, feature scaling, class imbalance handling (SMOTE), model comparison, hyperparameter tuning, and deployment-ready model saving** to create a high-performance fraud detection pipeline.

---

## 📂 Dataset Overview
**Source:** Credit Card Fraud Detection Dataset. Link:https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
**Transactions:** 280,000+ anonymized card transactions  

**Key Features:**
- Time and Amount of transaction  
- PCA-transformed variables (V1–V28) for confidentiality  
- Target Variable:  
  - **0 → Legitimate Transaction**  
  - **1 → Fraudulent Transaction**

📌 The dataset is **highly imbalanced**, with fraudulent transactions representing a very small percentage of total activity.

---

## 🔎 Exploratory Data Analysis (EDA)
- Verified dataset structure, summary statistics, and missing values (none found).
- Identified extreme class imbalance.
- Visualized transaction amount distribution.
- Analyzed correlations between features and fraud labels.
- Observed strong relationships between fraud and features such as **V14, V17, V12, and V10**.

**Key Insight:**  
Fraudulent transactions often exhibit distinct statistical patterns despite anonymized features.

---

## ⚙️ Data Preprocessing
- Separated features and target variable.
- Standardized **Time** and **Amount** using `StandardScaler`.
- Split dataset into training and testing sets with stratification.
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance the training data.

📌 **Outcome:** Improved model ability to detect minority fraud cases without bias toward legitimate transactions.

---

## 🤖 Model Development

### 1️⃣ Logistic Regression (Baseline)
- Implemented with class weighting to address imbalance.
- Achieved high recall but lower precision.

**Approx Performance:**
- Recall: ~0.92  
- AUC-ROC: ~0.98  

📌 Insight: Good for catching fraud but increases false positives.

---

### 2️⃣ XGBoost Classifier (Primary Model)
- Trained on SMOTE-balanced data.
- Utilized `scale_pos_weight` to further adjust class imbalance.
- Delivered significantly stronger precision-recall balance.

**Approx Performance:**
- Precision: ~0.88  
- Recall: ~0.85  
- F1 Score: ~0.86  
- AUC-ROC: ~0.98  

📌 Insight: XGBoost provided the best trade-off between detecting fraud and minimizing false alarms.

---

## 🎯 Hyperparameter Tuning
Used **GridSearchCV** to optimize:
- `n_estimators`
- `max_depth`
- `learning_rate`

📌 Result: Improved model generalization and predictive stability.

---

## 📊 Feature Importance
XGBoost feature importance analysis revealed top predictors:

- V14  
- V10  
- V17  
- V12  
- V11  

📌 Insight: These variables strongly influence fraud probability despite anonymization.

---

## 💾 Model Persistence
Saved production-ready artifacts:

- `best_xgb_model.pkl` → Trained fraud detection model  
- `scaler.pkl` → Feature scaler  

This enables rapid reuse without retraining.

---

## 🧠 Real-Time Fraud Prediction Script
Built an interactive prediction system that:

- Accepts transaction feature inputs  
- Scales required variables  
- Generates fraud probability  
- Explains prediction using top contributing features  

📌 **Business Value:** Demonstrates how the model can support real-time fraud monitoring systems.

---

## 🛠 Tech Stack
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- XGBoost  
- Imbalanced-learn (SMOTE)  
- Joblib  

---

## 📁 Repository Structure
```
├── creditcard.csv
├── Fraud prediction.ipynb
├── best_xgb_model.pkl
├── scaler.pkl
└── README.md
```

---

## 🚀 Skills Demonstrated
- Fraud Detection Modeling  
- Imbalanced Data Handling  
- Feature Scaling & Engineering  
- Machine Learning Model Comparison  
- Hyperparameter Tuning  
- Model Persistence  
- Predictive System Design  

📌 This project demonstrates how machine learning can proactively identify fraudulent financial activity, reduce monetary losses, and strengthen transaction security.

---

## ▶️ How to Execute the Project

### 1. Clone the Repository
```bash
git clone <your-repo-link>
cd credit-card-fraud-prediction
```

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn joblib jupyter
```

### 3. Run the Notebook
```bash
jupyter notebook "Fraud prediction.ipynb"
```
Execute all cells to reproduce EDA, preprocessing, model training, and evaluation.

### 4. Use the Saved Model for Predictions
After training, the model and scaler will be saved automatically.

Run your prediction script and enter feature values when prompted to classify transactions as **fraudulent or legitimate**.

---

## 🔮 Future Improvements
- Deploy the model using FastAPI or Flask  
- Build a real-time fraud detection pipeline  
- Implement deep learning models for comparison  
- Optimize threshold tuning for business risk tolerance  
- Create a monitoring dashboard for fraud alerts  

---
