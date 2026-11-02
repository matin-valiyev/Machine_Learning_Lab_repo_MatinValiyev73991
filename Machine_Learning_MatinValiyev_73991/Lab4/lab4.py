import pandas as pd
import numpy as np
import os
import joblib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import hashlib

print("1. Ingestion Step: Downloading data...")
adult = fetch_openml(name='adult', version=2, as_frame=True)
X = adult.data 
y = adult.target 

print(f"--> Log: Row count {len(X)} and column count {len(X.columns)}")

print("\n2. Splitting data (70/30 Stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

print("\n2.5 Validation Checks (Reporting Rule Breakers)...")
print("--> Age violations (<17 or >90):", len(X_train[(X_train['age'] < 17) | (X_train['age'] > 90)]))
print("--> Hours violations (<1 or >99):", len(X_train[(X_train['hours-per-week'] < 1) | (X_train['hours-per-week'] > 99)]))
print("--> Capital-gain violations (<0 or >99999):", len(X_train[(X_train['capital-gain'] < 0) | (X_train['capital-gain'] > 99999)]))
print("--> Capital-loss violations (<0 or >4356):", len(X_train[(X_train['capital-loss'] < 0) | (X_train['capital-loss'] > 4356)]))
print("--> Missing values ('?'):", (X_train == '?').sum().sum())

print("\n2.6 Cleaning: Duplicates & Outliers...")
duplicates = X_train.duplicated()
print(f"--> Dropping {duplicates.sum()} exact duplicate rows from training data.")
X_train = X_train[~duplicates]
y_train = y_train[~duplicates]

for col in ['age', 'hours-per-week', 'capital-gain', 'capital-loss']:
    bottom_limit = X_train[col].quantile(0.01)
    top_limit = X_train[col].quantile(0.99)
    # Using .astype(float) to prevent pandas warnings during clipping
    X_train.loc[:, col] = X_train[col].astype(float).clip(lower=bottom_limit, upper=top_limit)
    print(f"--> Clipped {col} limits: min {bottom_limit}, max {top_limit}")

print("\n3. Building the cleaning pipeline...")
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), 
    ('scaler', StandardScaler()) 
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), 
    ('onehot', OneHotEncoder(handle_unknown='ignore')) 
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_train_cleaned = preprocessor.fit_transform(X_train)
print("Data cleaned successfully!")

print("\n4. Saving the transformer...")
os.makedirs('results', exist_ok=True)
joblib.dump(preprocessor, 'results/preprocessor.joblib')
print("--> Saved to results/preprocessor.joblib")

print("\n5. Generating SHA256 Fingerprint...")
data_string = X_train.to_string().encode('utf-8')
fingerprint = hashlib.sha256(data_string).hexdigest()
print(f"--> Dataset SHA256 Fingerprint: {fingerprint}")

# --- PASTE THIS AT THE VERY BOTTOM OF lab4.py ---
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

print("\n6. Label-Quality Check (Finding conflicts)...")
# We combine features and answers to find liars (identical rows with different answers)[cite: 1]
audit_df = X_train.copy()
audit_df['income_label'] = y_train
# Find rows where all features are exactly the same, but there are multiple unique income labels[cite: 1]
conflicts = audit_df.groupby(list(X_train.columns))['income_label'].nunique()
conflict_count = (conflicts > 1).sum()
print(f"--> Found {conflict_count} duplicate rows that have conflicting income answers.")

print("\n7. Feature Engineering (Making new columns)...")
# First, let's see how smart the robot is WITHOUT our new columns[cite: 1]
# We clean the test data using the machine we already built
X_test_cleaned = preprocessor.transform(X_test)

# Train a basic robot[cite: 1]
robot_basic = LogisticRegression(max_iter=1000)
robot_basic.fit(X_train_cleaned, y_train)
y_pred_basic = robot_basic.predict(X_test_cleaned)
# Since income has two classes ('<=50K', '>50K'), we specify pos_label for F1 score
pos_label = '>50K' if '>50K' in y_train.values else y_train.unique()[1]
print("--> WITHOUT new features:")
print(f"    Accuracy: {accuracy_score(y_test, y_pred_basic):.4f}")
print(f"    F1 Score: {f1_score(y_test, y_pred_basic, pos_label=pos_label):.4f}")

# Now, let's create the new columns requested by the professor[cite: 1]
# 1. capital_net = capital-gain - capital-loss[cite: 1]
X_train['capital_net'] = X_train['capital-gain'] - X_train['capital-loss']
X_test['capital_net'] = X_test['capital-gain'] - X_test['capital-loss']

# 2. hours_per_age = hours-per-week / (age - 16)[cite: 1]
# We add a tiny number (0.0001) to avoid dividing by zero if someone is exactly 16!
X_train['hours_per_age'] = X_train['hours-per-week'] / (X_train['age'] - 16 + 0.0001)
X_test['hours_per_age'] = X_test['hours-per-week'] / (X_test['age'] - 16 + 0.0001)

# We must update our cleaning machine to know about these two new number columns
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
preprocessor.transformers_[0] = ('num', numeric_transformer, numeric_features)

# Clean the data again, now including our new columns
X_train_engineered = preprocessor.fit_transform(X_train)
X_test_engineered = preprocessor.transform(X_test)

# Train a new robot with the extra information[cite: 1]
robot_smart = LogisticRegression(max_iter=1000)
robot_smart.fit(X_train_engineered, y_train)
y_pred_smart = robot_smart.predict(X_test_engineered)
print("--> WITH new features:")
print(f"    Accuracy: {accuracy_score(y_test, y_pred_smart):.4f}")
print(f"    F1 Score: {f1_score(y_test, y_pred_smart, pos_label=pos_label):.4f}")

print("\n8. Preparing Manual Spot-Check...")
# Pick 50 random rows and save them to a file so you can look at them
spot_check_data = adult.data.sample(n=50, random_state=42)
spot_check_data.to_csv("manual_spot_check.csv", index=False)
print("--> Saved 50 rows to manual_spot_check.csv. Please open it and read it with your eyes!")