import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Step 1: Data Collection ---

data_path = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'

try:
    df = pd.read_csv(data_path)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print(f"Error: {data_path} not found.")
    print("Please ensure the file is in the 'data/' directory within your project.")
    print("Or provide the correct full path to the file.")
    exit()

print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset Info (Initial):")
df.info()

print("\nDataset Shape (rows, columns):", df.shape)

# --- Step 2: EDA & Preprocessing ---

# 2.1 Initial Data Inspection and Cleaning

print("\n--- Cleaning Phase ---")

print("\n'TotalCharges' before cleaning:")
print(df['TotalCharges'].value_counts(dropna=False).head())
print("Dtype:", df['TotalCharges'].dtype)

df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)

print("\nMissing values after replacing empty strings in 'TotalCharges':")
print(df.isnull().sum()['TotalCharges'])

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

nan_charges_tenure_check = df[df['TotalCharges'].isnull()][['tenure', 'MonthlyCharges', 'TotalCharges']]
print("\nRows with NaN 'TotalCharges' (expecting tenure 0):")
print(nan_charges_tenure_check)

df['TotalCharges'].fillna(0, inplace=True)

print("\nMissing values after imputing 'TotalCharges':")
print(df.isnull().sum()['TotalCharges'])
print("\n'TotalCharges' dtype after cleaning:", df['TotalCharges'].dtype)
print("First 5 rows with 'TotalCharges' after cleaning:")
print(df[['tenure', 'MonthlyCharges', 'TotalCharges']].head())

if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)
    print("\n'customerID' column dropped.")
else:
    print("\n'customerID' column not found (already dropped or not present).")
print("Current columns:", df.columns.tolist())

print("\n'tenure' column details:")
print("Dtype:", df['tenure'].dtype)
print("Unique values count:", df['tenure'].nunique())
print("Min tenure:", df['tenure'].min())
print("Max tenure:", df['tenure'].max())

print("\n--- EDA & Encoding Phase ---")

sns.set_style("whitegrid")

# Target Variable Analysis (Churn)
# Ensure Churn is numeric (0/1) for correlation later, but keep original for EDA plots if needed
# For simplicity, convert here permanently for rest of script.
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
print("\n'Churn' column converted to numeric (0/1):")
print(df['Churn'].value_counts())
print(df['Churn'].dtype)


plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df)
plt.title('Distribution of Churn')
plt.xlabel('Churn')
plt.ylabel('Number of Customers')
plt.show()

# Univariate Analysis of Categorical Features
categorical_cols_for_eda = df.select_dtypes(include='object').columns.tolist()

print("\nCategorical Columns for EDA:", categorical_cols_for_eda)

fig, axes = plt.subplots(len(categorical_cols_for_eda), 2, figsize=(15, 5 * len(categorical_cols_for_eda)))
axes = axes.flatten()

for i, col in enumerate(categorical_cols_for_eda):
    sns.countplot(x=col, data=df, ax=axes[i*2], palette='viridis')
    axes[i*2].set_title(f'Distribution of {col}')
    axes[i*2].tick_params(axis='x', rotation=45) # <-- CHANGED HERE: Removed ha='right'

    sns.countplot(x=col, hue='Churn', data=df, ax=axes[i*2+1], palette='plasma')
    axes[i*2+1].set_title(f'{col} vs. Churn')
    axes[i*2+1].tick_params(axis='x', rotation=45) # <-- CHANGED HERE: Removed ha='right'

plt.tight_layout()
plt.show()

# Univariate Analysis of Numerical Features
numerical_cols_for_eda = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'SeniorCitizen' in numerical_cols_for_eda:
    numerical_cols_for_eda.remove('SeniorCitizen')
if 'Churn' in numerical_cols_for_eda:
    numerical_cols_for_eda.remove('Churn')

print("\nNumerical Columns for EDA:", numerical_cols_for_eda)

fig, axes = plt.subplots(len(numerical_cols_for_eda), 2, figsize=(15, 5 * len(numerical_cols_for_eda)))
axes = axes.flatten()

for i, col in enumerate(numerical_cols_for_eda):
    sns.histplot(df[col], kde=True, ax=axes[i*2], bins=30, color='skyblue')
    axes[i*2].set_title(f'Distribution of {col}')

    sns.boxplot(x=col, data=df, ax=axes[i*2+1], color='lightcoral')
    axes[i*2+1].set_title(f'Box Plot of {col}')

plt.tight_layout()
plt.show()

numerical_cols_vs_churn = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']

fig, axes = plt.subplots(len(numerical_cols_vs_churn), 1, figsize=(10, 5 * len(numerical_cols_vs_churn)))

for i, col in enumerate(numerical_cols_vs_churn):
    sns.violinplot(x='Churn', y=col, data=df, ax=axes[i], palette='pastel')
    axes[i].set_title(f'{col} vs. Churn')

plt.tight_layout()
plt.show()

# Correlation Matrix (Numerical Features)
numeric_df_for_corr = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df_for_corr.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features (including Churn)')
plt.show()

# 2.3 Preprocessing - Encoding Remaining Categorical Features

categorical_features_to_encode = [col for col in df.columns if df[col].dtype == 'object']
print("\nCategorical features to be one-hot encoded (remaining):", categorical_features_to_encode)

df_encoded = pd.get_dummies(df, columns=categorical_features_to_encode, drop_first=True, dtype=int)

print("\nShape after one-hot encoding:", df_encoded.shape)
print("\nFirst 5 rows of encoded dataset (df_encoded):")
print(df_encoded.head())
print("\nColumns after encoding (df_encoded):")
print(df_encoded.columns.tolist())
print("\nFinal DataFrame Info after encoding (df_encoded):")
df_encoded.info()

# --- Save the Cleaned and Encoded DataFrame ---
print("\n--- Saving Cleaned and Encoded Data ---")
output_path_csv = 'data/data.csv'
df_encoded.to_csv(output_path_csv, index=False)
print(f"Cleaned and Encoded DataFrame saved to: {output_path_csv}")