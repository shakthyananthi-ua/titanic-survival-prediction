# 🚢 Titanic Survival Prediction

A machine learning project that predicts whether a Titanic passenger survived, based on features like class, sex, age, fare, and family size. Built as a hands-on exercise in the full ML workflow — from raw data to a trained, evaluated model.

## 📌 Problem Statement
Given passenger data from the Titanic (class, sex, age, fare, etc.), predict whether that passenger survived the disaster. This is a classic binary classification problem, useful for practicing data cleaning, feature engineering, and model evaluation.

## 🧰 Tech Stack
- Python 3
- pandas, numpy — data handling
- matplotlib, seaborn — visualization
- scikit-learn — modeling (Logistic Regression, Random Forest)

## 🔍 Approach
1. **Data Cleaning** — handled missing `Age` (median imputation) and `Embarked` (mode imputation); dropped `Cabin` (too sparse to be useful).
2. **Feature Engineering** — created `FamilySize` and `IsAlone` features; label-encoded categorical fields (`Sex`, `Embarked`).
3. **EDA** — visualized survival rate by sex, class, age, and family size to understand key patterns.
4. **Modeling** — trained and compared Logistic Regression and Random Forest classifiers.
5. **Evaluation** — assessed accuracy, precision/recall, and feature importance.

## 📊 Results
| Model | Accuracy |
|---|---|
| Logistic Regression | ~79% |
| Random Forest | ~81% |

**Key insight:** `Sex`, `Fare`, and `Age` were the most influential features in predicting survival — consistent with the historical "women and children first" boarding policy and the fact that wealthier passengers (higher fare) had better access to lifeboats.

### Sample Visualization
![EDA Summary](eda_summary.png)

## 🚀 How to Run
```bash
git clone https://github.com/YOUR_USERNAME/titanic-survival-prediction.git
cd titanic-survival-prediction
pip install -r requirements.txt
python titanic_analysis.py
```

## 📁 Project Structure
```
titanic-survival-prediction/
├── data/
│   └── titanic.csv
├── titanic_analysis.py
├── requirements.txt
├── eda_summary.png
├── feature_importance.png
└── README.md
```

## 💡 Future Improvements
- Try gradient boosting (XGBoost/LightGBM) for higher accuracy
- Hyperparameter tuning with GridSearchCV
- Deploy as a simple Streamlit app for interactive predictions
