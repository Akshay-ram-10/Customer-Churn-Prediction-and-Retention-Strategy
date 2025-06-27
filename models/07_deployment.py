import pandas as pd
import numpy as np
import joblib
import os

# --- Step 8: Deployment (Conceptual) ---

print("\n--- Step 8: Model Deployment (Conceptual) ---")

model_dir = 'models/'
data_dir = 'data/'
final_features_data_path = 'data/final_features.csv'

# Load the trained model and scaler
try:
    best_model = joblib.load(os.path.join(model_dir, 'random_forest_model.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    df_final = pd.read_csv(final_features_data_path)
    print("Trained model and scaler loaded successfully for deployment.")
except FileNotFoundError:
    print("Error: Model or scaler files not found in 'models/' directory.")
    print("Please ensure '03_model_building.py' was run to save these assets.")
    exit()

# --- Feature Engineering Function (must match 02_feature_engineering.py) ---
def apply_feature_engineering(df_raw_or_preprocessed):
    # This function should replicate the feature engineering steps from 02_feature_engineering.py
    # It assumes df_raw_or_preprocessed already has 'Churn' converted to 0/1 and customerID dropped
    # and has been one-hot encoded for original categorical features.

    # Ensure Churn is numeric (0/1) just in case for internal consistency
    if 'Churn' in df_raw_or_preprocessed.columns and df_raw_or_preprocessed['Churn'].dtype == 'object':
        df_raw_or_preprocessed['Churn'] = df_raw_or_preprocessed['Churn'].map({'Yes': 1, 'No': 0})

    df_temp = df_raw_or_preprocessed.copy()

    # MonthlyToTotalRatio
    df_temp['MonthlyToTotalRatio'] = df_temp.apply(
        lambda row: row['MonthlyCharges'] / row['TotalCharges'] if row['TotalCharges'] != 0 else 0,
        axis=1
    )

    # TenureGroup
    bins = [0, 12, 24, 48, 60, np.inf]
    labels = ['0-12', '13-24', '25-48', '49-60', '>60']
    df_temp['TenureGroup'] = pd.cut(df_temp['tenure'], bins=bins, labels=labels, right=False)
    # Re-create dummy columns for TenureGroup (ensure consistency with training)
    # For prediction, you might need to use a pre-fitted OneHotEncoder if categories might be missing
    # But for simplicity, we assume all possible categories appear or are handled by get_dummies.
    df_temp = pd.get_dummies(df_temp, columns=['TenureGroup'], drop_first=True, dtype=int)


    # HasMultipleServices
    service_cols = [
        'MultipleLines_Yes', 'OnlineSecurity_Yes', 'OnlineBackup_Yes',
        'DeviceProtection_Yes', 'TechSupport_Yes', 'StreamingTV_Yes',
        'StreamingMovies_Yes'
    ]
    existing_service_cols = [col for col in service_cols if col in df_temp.columns]
    if existing_service_cols:
        df_temp['HasMultipleServices'] = (df_temp[existing_service_cols].sum(axis=1) > 0).astype(int)
    else:
        df_temp['HasMultipleServices'] = 0 # Default if cols are missing

    # TotalServiceScore
    value_added_services = [
        'OnlineSecurity_Yes', 'OnlineBackup_Yes', 'DeviceProtection_Yes',
        'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes',
        'MultipleLines_Yes'
    ]
    existing_value_added_services = [col for col in value_added_services if col in df_temp.columns]
    if existing_value_added_services:
        df_temp['TotalServiceScore'] = df_temp[existing_value_added_services].sum(axis=1)
    else:
        df_temp['TotalServiceScore'] = 0 # Default if cols are missing


    # Important: Realign columns to match training data columns
    # Load feature names from the original training set (X_train) to ensure consistency
    # A robust way is to save X_train.columns during model building and load it here.
    # For now, we'll load the processed data to get the reference columns.
    reference_df = pd.read_csv(os.path.join(data_dir, 'final_features.csv'))
    reference_columns = reference_df.drop('Churn', axis=1).columns.tolist()

    # Add missing columns with 0, drop extra columns
    missing_cols = set(reference_columns) - set(df_temp.columns)
    for c in missing_cols:
        df_temp[c] = 0
    extra_cols = set(df_temp.columns) - set(reference_columns)
    df_temp = df_temp.drop(columns=list(extra_cols))

    df_engineered = df_temp[reference_columns] # Ensure column order matches training

    return df_engineered

# --- Prediction Function ---
def predict_churn(customer_data_df):
    # customer_data_df: A pandas DataFrame containing new customer data
    # (can be one row for a single customer or multiple rows for a batch)

    # 1. Apply necessary preprocessing (e.g., TotalCharges cleaning, ID drop)
    # This step should replicate logic from 01_data_preprocessing.py as well
    # For simplicity, this example assumes input is already somewhat preprocessed (no customerID, TotalCharges numeric)
    # In a real deployment, you'd add more robust input cleaning here.

    if 'customerID' in customer_data_df.columns:
        customer_data_df = customer_data_df.drop('customerID', axis=1)

    if 'TotalCharges' in customer_data_df.columns:
        # Handle cases where TotalCharges might be ' ' in new data
        customer_data_df['TotalCharges'] = customer_data_df['TotalCharges'].replace(' ', np.nan)
        customer_data_df['TotalCharges'] = pd.to_numeric(customer_data_df['TotalCharges'], errors='coerce')
        customer_data_df['TotalCharges'].fillna(0, inplace=True)

    # Ensure Churn column is not present if it's new data for prediction
    if 'Churn' in customer_data_df.columns:
        customer_data_df = customer_data_df.drop('Churn', axis=1)

    # Convert object columns to one-hot if they are still present
    categorical_cols_to_convert = [col for col in customer_data_df.columns if customer_data_df[col].dtype == 'object']
    customer_data_df = pd.get_dummies(customer_data_df, columns=categorical_cols_to_convert, drop_first=True, dtype=int)


    # 2. Apply Feature Engineering
    features_engineered = apply_feature_engineering(customer_data_df)

    # 3. Scale the features
    features_scaled = scaler.transform(features_engineered)
    features_scaled_df = pd.DataFrame(features_scaled, columns=features_engineered.columns, index=features_engineered.index)


    # 4. Make prediction
    churn_probability = best_model.predict_proba(features_scaled_df)[:, 1]
    churn_prediction = best_model.predict(features_scaled_df)

    results_df = pd.DataFrame({
        'Predicted_Churn_Probability': churn_probability,
        'Predicted_Churn': churn_prediction
    }, index=customer_data_df.index)

    return results_df

# --- Demonstration of Prediction ---
print("\n--- Demonstration: Predicting Churn for New Customers ---")

# Create some sample new customer data (replace with actual new data)
# This dummy data MUST have the same columns as your original raw data (minus customerID, plus new features if added).
# It's crucial for the preprocessing and feature engineering steps to work.

# Example structure of incoming data:
# You'd typically receive a new customer's details in a dictionary or a single-row DataFrame.
sample_new_customer_data = {
    'gender': ['Male'],
    'SeniorCitizen': [0],
    'Partner': ['Yes'],
    'Dependents': ['No'],
    'tenure': [10],
    'PhoneService': ['Yes'],
    'MultipleLines': ['No'],
    'InternetService': ['Fiber optic'],
    'OnlineSecurity': ['No'],
    'OnlineBackup': ['No'],
    'DeviceProtection': ['Yes'],
    'TechSupport': ['No'],
    'StreamingTV': ['Yes'],
    'StreamingMovies': ['Yes'],
    'Contract': ['Month-to-month'],
    'PaperlessBilling': ['Yes'],
    'PaymentMethod': ['Electronic check'],
    'MonthlyCharges': [89.95],
    'TotalCharges': [899.50]
}

new_customer_df = pd.DataFrame(sample_new_customer_data)
print("\nSample New Customer Data:")
print(new_customer_df)

# Make prediction
prediction_results = predict_churn(new_customer_df.copy()) # Use .copy() to avoid SettingWithCopyWarning
print("\nPrediction Results for Sample New Customer:")
print(prediction_results)

# --- Batch Prediction Example ---
# Load a small part of your original dataset to simulate a batch prediction scenario
# Exclude the actual 'Churn' column from this simulation
simulated_batch_data = df_final.drop('Churn', axis=1).sample(5, random_state=10).copy()

# For a true "new" batch, you would load raw data and apply initial preprocessing.
# Here we take engineered features for simplicity of demonstration for this script.
# In real deployment, if you get raw data, you'd apply _all_ steps.

# Let's simplify this batch prediction demo to only include the raw-like columns
# so the `predict_churn` function (which includes initial preprocessing) is fully utilized.

# Load original raw data, clean it for this demo purpose, and then use it for prediction
# This is to simulate receiving raw-like data.
raw_data_for_demo_path = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
try:
    demo_raw_df = pd.read_csv(raw_data_for_demo_path).sample(5, random_state=42).reset_index(drop=True)
    if 'customerID' in demo_raw_df.columns:
        demo_customer_ids = demo_raw_df['customerID']
        demo_raw_df = demo_raw_df.drop('customerID', axis=1)
    else:
        demo_customer_ids = [f"Cust_{i}" for i in range(len(demo_raw_df))]

    print("\nSimulated Batch Raw Data (first 5 rows):")
    print(demo_raw_df)

    # Make prediction for the batch
    batch_prediction_results = predict_churn(demo_raw_df.copy())
    batch_prediction_results['CustomerID'] = demo_customer_ids
    batch_prediction_results = batch_prediction_results[['CustomerID', 'Predicted_Churn_Probability', 'Predicted_Churn']]

    print("\nBatch Prediction Results:")
    print(batch_prediction_results)

except FileNotFoundError:
    print(f"Error: {raw_data_for_demo_path} not found for batch prediction demo.")

print("\n--- Deployment Considerations ---")
print("1. **API Integration:** For real-time, wrap the 'predict_churn' function in a web framework (e.g., Flask, FastAPI) to create an API endpoint.")
print("2. **Batch Processing:** For batch, schedule the script to run periodically on new data (e.g., using cron jobs, Airflow).")
print("3. **Monitoring:** Continuously monitor model performance (e.g., data drift, prediction accuracy) and retrain as needed.")
print("4. **Feedback Loop:** Establish a feedback mechanism to collect actual churn outcomes to improve future model versions.")
print("5. **Infrastructure:** Decide on deployment environment (e.g., local server, cloud platforms like AWS SageMaker, Azure ML, GCP AI Platform).")
print("6. **Error Handling:** Implement robust error handling for unexpected input formats or missing data.")

print("\nDeployment step conceptually outlined.")