## Project Overview

Customer churn, the loss of customers, is a significant challenge for businesses, particularly in subscription-based industries like telecommunications. It directly impacts revenue, growth, and market share. This project addresses this challenge by developing a comprehensive machine learning solution aimed at:

1.  **Predicting** which customers are most likely to churn.
2.  **Identifying** the key factors driving churn.
3.  **Proposing** data-driven, actionable strategies to improve customer retention.

By accurately identifying at-risk customers, companies can proactively intervene, personalize their retention efforts, and optimize resource allocation, ultimately leading to improved customer lifetime value and sustained business growth.

## Goals

The primary objectives of this project were to:

* Develop and evaluate a robust classification model capable of predicting customer churn with high accuracy and a strong ability to identify true churners (high recall/F1-score).
* Uncover the most influential features and patterns contributing to customer churn through thorough data analysis and model interpretation.
* Translate analytical insights into practical, actionable recommendations for customer retention campaigns.
* Demonstrate a complete end-to-end machine learning project pipeline, from raw data preprocessing to conceptual model deployment.

## Data Source

The dataset used for this project is the **IBM Watson Telco Customer Churn dataset**, publicly available on Kaggle. It comprises customer information for a telecommunications company, detailing various aspects such as:

* **Demographics:** Gender, Senior Citizen status, Partners, Dependents.
* **Account Information:** Tenure, Contract type, Paperless Billing, Payment Method, Monthly Charges, Total Charges.
* **Services:** Phone Service, Multiple Lines, Internet Service, Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies.
* **Target Variable:** Churn (Yes/No).

[Link to Kaggle dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Optional: Replace with actual link if you reference it directly)

## Methodology

This project follows a structured machine learning pipeline, implemented through a series of modular Python scripts:

1.  **Data Loading & Preprocessing (`scripts/01_data_preprocessing.py`):**
    * Initial loading of the raw dataset.
    * Handling of missing values (specifically for `TotalCharges`).
    * Correction of data types and standardization of column names for consistency.

2.  **Exploratory Data Analysis (EDA) & Initial Insights (`notebooks/Churn_EDA_and_Insights.ipynb`):**
    * In-depth analysis of data distributions, relationships between features, and initial correlations with the 'Churn' target variable.
    * **Refer to the dedicated [Jupyter Notebook](notebooks/Churn_EDA_and_Insights.ipynb) for a detailed visual walkthrough and insights gained during EDA.**

3.  **Feature Engineering (`scripts/02_feature_engineering.py`):**
    * Creation of new, more informative features to enhance model performance. Examples include:
        * `MonthlyToTotalRatio`: Ratio of monthly charges to total charges, capturing billing consistency.
        * `TotalServiceScore`: A numerical score representing the number of services a customer subscribes to.
        * `HasAddonServices`: Binary indicator if customer has any security/support addons.
    * Encoding of categorical variables (One-Hot Encoding).
    * Scaling of numerical features (StandardScaler).

4.  **Model Building & Training (`scripts/03_model_building.py`):**
    * Experimentation with various classification algorithms suitable for tabular data, including Logistic Regression, Random Forest, and Gradient Boosting Classifier.
    * Addressing class imbalance in the target variable ('Churn') using **SMOTE (Synthetic Minority Over-sampling Technique)** to prevent bias towards the majority class.

5.  **Model Evaluation (`scripts/04_model_evaluation.py`):**
    * Comprehensive evaluation of trained models using a suite of metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
    * Analysis of Confusion Matrices to understand true positives, false positives, true negatives, and false negatives.
    * Cross-validation to ensure model robustness and generalization.

6.  **Model Interpretation & Insights (`scripts/05_interpretation_insights.py`):**
    * Utilizing feature importance techniques (e.g., from tree-based models or permutation importance) to identify the most impactful features driving customer churn.
    * Translating statistical findings into clear, business-relevant explanations.

7.  **Retention Strategy Development (`scripts/06_retention_strategy.py`):**
    * Developing targeted retention strategies directly based on the identified churn drivers and model insights.

8.  **Conceptual Deployment (`scripts/07_deployment.py`):**
    * Demonstrating the process of loading the trained model and scaler to make predictions on new, unseen customer data. This script serves as a conceptual blueprint for a real-world deployment.

## Key Findings & Business Insights

Our analysis revealed several critical factors significantly influencing customer churn:

* **Contract Type is Paramount:** Customers on **Month-to-month contracts** exhibit a disproportionately high churn rate compared to those on 1-year or 2-year contracts. This indicates a strong desire for flexibility but also a lack of commitment.
* **Internet Service Impact:** Customers with **Fiber Optic internet service** show a notably higher propensity to churn. This could point towards underlying service quality issues, reliability concerns, or dissatisfaction with its cost-to-value ratio.
* **Importance of Value-Added Services:** The absence of **Online Security, Online Backup, Device Protection, and Tech Support** services is a strong predictor of churn. Customers lacking these protective or supportive services feel less engaged or secure.
* **Tenure and Total Charges:** Shorter customer tenure and lower `TotalCharges` (often correlated) are associated with higher churn, suggesting that customers may churn early due to initial dissatisfaction or a lack of strong initial engagement.
* **Payment Method:** Certain payment methods, particularly **Electronic Check**, are associated with higher churn, possibly indicating a segment of customers less integrated or satisfied with billing convenience.

## Actionable Retention Strategies

Based on the insights derived from the model and data analysis, here are targeted recommendations for customer retention:

* **Incentivize Long-Term Contracts:** Develop compelling offers (e.g., discounts, premium upgrades) to encourage month-to-month customers, especially those showing early signs of churn risk, to transition to 1-year or 2-year contracts.
* **Improve Fiber Optic Service & Communication:** Conduct a thorough review of Fiber Optic service quality. Proactively address common pain points, and enhance communication about service benefits and troubleshooting. Consider offering loyalty discounts to high-value Fiber Optic customers at risk.
* **Bundle and Promote Value-Added Services:** Actively market and offer free trials or introductory bundles of Online Security, Online Backup, Device Protection, and Tech Support to new and at-risk customers. Highlight the security and convenience benefits.
* **Enhanced Onboarding & Early Engagement:** For new customers, especially those without long-term contracts, implement a robust onboarding program. This could include personalized follow-ups, ensuring smooth service activation, and offering early access to support resources.
* **Targeted Outreach for Electronic Check Users:** Investigate the specific reasons for churn among Electronic Check users. This might involve surveys or focused outreach to understand if billing convenience or other factors are at play.

## Model Performance

The **Random Forest Classifier**, after balancing the dataset using SMOTE, emerged as the most suitable model for this prediction task due to its balanced performance across various metrics and its interpretability.

| Metric      | Value   |
| :---------- | :------ |
| Accuracy    | 0.7339  |
| Precision   | 0.4992  |
| Recall      | 0.7914  |
| F1-Score    | 0.6122  |
| ROC-AUC     | 0.8455  |

## Technical Stack

* **Languages:** Python 3.x
* **Libraries:**
    * `pandas`: For data manipulation and analysis.
    * `numpy`: For numerical operations.
    * `scikit-learn`: For machine learning models, preprocessing (scaling, encoding), and evaluation.
    * `matplotlib` & `seaborn`: For data visualization and exploratory data analysis.
    * `imblearn` (scikit-learn-contrib): For handling class imbalance (SMOTE).
    * `joblib`: For model persistence (saving and loading models).

## Project Structure

.
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Original raw dataset
│   ├── data.csv # Intermediate processed data (optional to include in repo if large)
│   └── final_features.csv # Final features used for model training (optional to include in repo if large)
├── models/
│   ├── random_forest_model.pkl                  # Trained Random Forest Model
│   └── scaler.pkl                               # Fitted StandardScaler object
│   ├── 01_data_preprocessing.py                 # Handles initial data cleaning and preparation
│   ├── 02_feature_engineering.py                # Creates new features and performs encoding/scaling
│   ├── 03_model_building.py                     # Trains and saves the ML model
│   ├── 04_model_evaluation.py                   # Evaluates model performance
│   ├── 05_interpretation_insights.py            # Extracts and presents model insights (e.g., feature importance)
│   ├── 06_retention_strategy.py                 # Outlines strategic recommendations based on insights
│   └── 07_deployment.py                         # Demonstrates model loading and prediction on new data
├── notebooks/
│   └── Churn_EDA_and_Insights.ipynb             # Detailed Jupyter Notebook for EDA and visual insights
├── .gitignore                                   # Specifies files and folders to be ignored by Git
└── README.md                                    # Project overview and documentation (this file)


## How to Run (Local Setup)

To set up and run this project on your local machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Akshay-ram-10/Customer-Churn-Prediction-Project.git](https://github.com/Akshay-ram-10/Customer-Churn-Prediction-Project.git)
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd Customer-Churn-Prediction-Project
    ```
3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```
    * **Activate the virtual environment:**
        * **Windows:** `.\venv\Scripts\activate`
        * **macOS/Linux:** `source venv/bin/activate`
4.  **Install the required libraries:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn imblearn joblib
    ```
5.  **Run the Python scripts in sequence:**
    ```bash
    python scripts/01_data_preprocessing.py
    python scripts/02_feature_engineering.py
    python scripts/03_model_building.py
    python scripts/04_model_evaluation.py
    python scripts/05_interpretation_insights.py
    python scripts/06_retention_strategy.py
    python scripts/07_deployment.py
    ```
6.  **Explore the EDA & Insights Notebook:**
    * Make sure you have Jupyter installed (`pip install jupyter`).
    * Run Jupyter from the project root: `jupyter notebook`
    * Navigate to and open `notebooks/Churn_EDA_and_Insights.ipynb` in your browser.

## Future Work & Enhancements

* Develop a production-ready API (e.g., using FastAPI or Flask) for real-time churn predictions, potentially containerized with Docker.
* Implement a robust MLOps pipeline for continuous model monitoring (data drift, model drift) and automated retraining.
* Integrate the prediction system with a CRM (Customer Relationship Management) platform for automated, personalized outreach.
* Conduct A/B testing of different retention strategies to empirically measure their effectiveness.
* Explore more advanced machine learning techniques, such as deep learning or complex ensemble methods, for potential marginal performance gains.

## Contact

Feel free to reach out via email at [kamujuakshayram@gmail.com](mailto:kamujuakshayram@gmail.com).
