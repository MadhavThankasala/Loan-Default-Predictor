#!/usr/bin/env python
# coding: utf-8

# SBA Loan Default Prediction Project
# -----------------------------------
# This script cleans, processes, and models SBA loan data
# to predict whether a loan will default or be paid in full.

# ------------------------------
# Import Libraries
# ------------------------------
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# ------------------------------
# Load and Clean Dataset
# ------------------------------
data = pd.read_csv('SBAnational.csv', low_memory=False)

# Drop rows with missing essential fields
data.dropna(subset=['Name', 'City', 'State', 'Bank', 'BankState', 'NewExist',
                    'RevLineCr', 'LowDoc', 'DisbursementDate', 'MIS_Status'], inplace=True)

# Convert currency fields to numeric
money_cols = ['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']
for col in money_cols:
    data[col] = data[col].astype(str).str.replace('$', '').str.replace(',', '').str.strip()
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Clean ApprovalFY (e.g., remove trailing 'A')
def clean_str(x):
    if isinstance(x, str):
        return x.replace('A', '')
    return x

data['ApprovalFY'] = data['ApprovalFY'].apply(clean_str).astype('int64')

# Fix column types
data = data.astype({'Zip': 'str', 'NewExist': 'int64', 'UrbanRural': 'str'})

# ------------------------------
# Feature Engineering
# ------------------------------
# Industry from first two NAICS digits
data['Industry'] = data['NAICS'].astype('str').str[:2].map({
    '11': 'Ag/For/Fish/Hunt', '21': 'Min/Quar/Oil_Gas_ext', '22': 'Utilities', '23': 'Construction',
    '31': 'Manufacturing', '32': 'Manufacturing', '33': 'Manufacturing', '42': 'Wholesale_trade',
    '44': 'Retail_trade', '45': 'Retail_trade', '48': 'Trans/Ware', '49': 'Trans/Ware',
    '51': 'Information', '52': 'Finance/Insurance', '53': 'RE/Rental/Lease', '54': 'Prof/Science/Tech',
    '55': 'Mgmt_comp', '56': 'Admin_sup/Waste_Mgmt_Rem', '61': 'Educational',
    '62': 'Healthcare/Social_assist', '71': 'Arts/Entertain/Rec', '72': 'Accom/Food_serv',
    '81': 'Other_no_pub', '92': 'Public_Admin'
})
data.dropna(subset=['Industry'], inplace=True)

# Franchise flag
data['IsFranchise'] = np.where(data['FranchiseCode'] > 1, 1, 0)

# Business age flag
data = data[(data['NewExist'] == 1) | (data['NewExist'] == 2)]
data['NewBusiness'] = np.where(data['NewExist'] == 2, 1, 0)

# Clean binary Y/N columns
data = data[data['RevLineCr'].isin(['Y', 'N'])]
data = data[data['LowDoc'].isin(['Y', 'N'])]
data['RevLineCr'] = np.where(data['RevLineCr'] == 'Y', 1, 0)
data['LowDoc'] = np.where(data['LowDoc'] == 'Y', 1, 0)

# Loan outcome (target)
data['LoanDefault'] = np.where(data['MIS_Status'] == 'P I F', 0, 1)

# ------------------------------
# Date Features
# ------------------------------
# Convert to datetime safely
data['ApprovalDate'] = pd.to_datetime(data['ApprovalDate'], errors='coerce')
data['DisbursementDate'] = pd.to_datetime(data['DisbursementDate'], errors='coerce')

# Create a numeric feature: number of days between approval and disbursement
data['DaysToDisburse'] = (data['DisbursementDate'] - data['ApprovalDate']).dt.days

# Drop unnecessary columns (including raw date fields)
for col in ['LoanNr_ChkDgt', 'Name', 'City', 'Zip', 'Bank', 'NAICS',
            'NewExist', 'FranchiseCode', 'MIS_Status',
            'ApprovalDate', 'DisbursementDate']:
    if col in data.columns:
        data.drop(columns=[col], inplace=True)

# ------------------------------
# Encode Categorical Columns
# ------------------------------
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data[col] = data[col].astype('category').cat.codes

# Drop any remaining datetime columns (final safety check)
datetime_cols = data.select_dtypes(include=['datetime64']).columns
if len(datetime_cols) > 0:
    print(f"Dropping datetime columns: {list(datetime_cols)}")
    data.drop(columns=datetime_cols, inplace=True)

# Verify all columns are numeric
non_numeric = data.select_dtypes(exclude=[np.number]).columns
print("Non-numeric columns (should be empty):", list(non_numeric))

# ------------------------------
# Modeling Section
# ------------------------------
X = data.drop('LoanDefault', axis=1)
y = data['LoanDefault']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- Model 1: Decision Tree ---
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)
print("\nDecision Tree Accuracy:", accuracy_score(y_test, tree_pred))

# --- Model 2: Logistic Regression Pipeline ---
log_model = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(k=10)),
    ('classifier', LogisticRegression(max_iter=1000))
])

log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

print("\nLogistic Regression Accuracy:", accuracy_score(y_test, log_pred))
print("\nClassification Report:")
print(classification_report(y_test, log_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, log_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix (Logistic Regression)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Cross-validation
cv_scores = cross_val_score(log_model, X, y, cv=5)
print("\nCross-Validation Scores:", cv_scores)
print("Average CV Accuracy:", cv_scores.mean())

# ------------------------------
# Single Prediction Example
# ------------------------------
sample_idx = 150
single_business = X_test.iloc[[sample_idx]]
pred_class = tree_model.predict(single_business)
pred_prob = tree_model.predict_proba(single_business)[0][1]

print(f"\nPredicted Loan Default Class: {pred_class[0]} (0 = Paid, 1 = Default)")
print(f"Probability of Loan Default: {pred_prob:.2f}")
