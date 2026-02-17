# Bank Marketing Campaign Prediction using Machine Learning

## 📌 Project Overview
This project focuses on predicting whether a customer will subscribe to a bank term deposit based on demographic, financial, and marketing campaign data.  
By leveraging machine learning techniques, the goal is to help banks optimize marketing strategies and reduce unnecessary customer outreach.

The project demonstrates an **end-to-end machine learning workflow**, from data preprocessing and exploratory analysis to model deployment using a Streamlit web application.

---

## 📂 Dataset
- **Source**: Bank Marketing Dataset
- **Records**: ~45,000 customers
- **Target Variable**: `y` (Subscription: yes / no)
- **Features**:
  - Demographic: age, job, marital status, education
  - Financial: balance, loans, credit default
  - Campaign-related: contact type, campaign count, previous outcome

⚠️ The `duration` column was removed to prevent **data leakage**, as it is only known after the call is completed.

---

## 🔍 Exploratory Data Analysis (EDA)
Key insights from EDA:
- The dataset is **highly imbalanced**, with fewer subscribers than non-subscribers.
- Customers with **higher account balances** tend to subscribe more.
- Subscription rates vary significantly across **job categories**.
- Correlation analysis showed **low multicollinearity** among features.

Outliers were **not removed** because they represent genuine customer behavior, and the final model is robust to such values.

---

## ⚙️ Data Preprocessing
- Binary categorical variables were converted to **0/1**.
- Categorical features were encoded using **One-Hot Encoding**.
- Feature scaling was applied only for models that require it (Logistic Regression, KNN, SVM).
- Data was split into training and testing sets using **stratified sampling**.

---

## 🤖 Models Implemented
The following machine learning models were trained and evaluated:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

---

## 📊 Model Evaluation
Due to class imbalance, models were evaluated using:
- **ROC–AUC**
- **Recall (positive class)**
- **F1-score**

Accuracy alone was not considered sufficient.

---

## 🏆 Final Model
- **Tuned Random Forest Classifier**
- Selected based on:
  - Better generalization
  - Improved minority class detection
  - Robust handling of non-linear relationships and outliers

Hyperparameter tuning was performed using **GridSearchCV** to reduce overfitting.

---

## 🚀 Deployment
The final model was deployed using **Streamlit**.

Saved components:
- Trained Random Forest model
- One-Hot Encoder
- Feature schema

The Streamlit app allows users to:
- Enter customer and campaign details
- Get real-time predictions
- View subscription probability and recommendations
