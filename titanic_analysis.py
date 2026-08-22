"""
Titanic Survival Prediction
----------------------------
A beginner-friendly ML project that predicts passenger survival
on the Titanic using classic tabular features (class, sex, age, fare, etc.)

Steps:
1. Load & explore the data
2. Clean missing values
3. Engineer simple features
4. Visualize key survival patterns
5. Train a classification model
6. Evaluate performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sns.set_style("whitegrid")

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
df = pd.read_csv("data/titanic.csv")
print("Dataset shape:", df.shape)
print(df.head())

# ---------------------------------------------------------
# 2. Clean missing values
# ---------------------------------------------------------
print("\nMissing values before cleaning:\n", df.isnull().sum())

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df.drop(columns=["Cabin"], inplace=True)  # too many missing values to be useful

print("\nMissing values after cleaning:\n", df.isnull().sum())

# ---------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

le_sex = LabelEncoder()
le_embarked = LabelEncoder()
df["Sex_enc"] = le_sex.fit_transform(df["Sex"])
df["Embarked_enc"] = le_embarked.fit_transform(df["Embarked"])

# ---------------------------------------------------------
# 4. Exploratory visualizations
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.barplot(x="Sex", y="Survived", data=df, ax=axes[0, 0])
axes[0, 0].set_title("Survival Rate by Sex")

sns.barplot(x="Pclass", y="Survived", data=df, ax=axes[0, 1])
axes[0, 1].set_title("Survival Rate by Passenger Class")

sns.histplot(data=df, x="Age", hue="Survived", multiple="stack", bins=30, ax=axes[1, 0])
axes[1, 0].set_title("Age Distribution by Survival")

sns.barplot(x="FamilySize", y="Survived", data=df, ax=axes[1, 1])
axes[1, 1].set_title("Survival Rate by Family Size")

plt.tight_layout()
plt.savefig("eda_summary.png", dpi=150)
print("\nSaved EDA chart to eda_summary.png")

# ---------------------------------------------------------
# 5. Train/test split
# ---------------------------------------------------------
features = ["Pclass", "Sex_enc", "Age", "Fare", "FamilySize", "IsAlone", "Embarked_enc"]
X = df[features]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 6. Train models
# ---------------------------------------------------------
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
log_preds = log_reg.predict(X_test)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# ---------------------------------------------------------
# 7. Evaluate
# ---------------------------------------------------------
print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, log_preds))
print(classification_report(y_test, log_preds))

print("\n--- Random Forest ---")
print("Accuracy:", accuracy_score(y_test, rf_preds))
print(classification_report(y_test, rf_preds))

# Feature importance (Random Forest)
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature importances (Random Forest):\n", importances)

plt.figure(figsize=(8, 5))
sns.barplot(x=importances.values, y=importances.index)
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
print("Saved feature importance chart to feature_importance.png")
