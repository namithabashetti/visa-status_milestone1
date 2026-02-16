# Visa Status Prediction Dataset & Preprocessing

## Project Overview

This project uses a synthetic visa application dataset containing: -
Application Date - Decision Date - Applicant Details - Visa Status

The goal is to preprocess the dataset and prepare it for machine
learning.

## Files Included

-   visa_dataset_200_rows.csv → Raw dataset (200 rows with missing
    values)
-   visa_preprocessing.py → Python preprocessing script
-   visa_dataset_cleaned.csv → Cleaned dataset (generated after running
    script)

## Preprocessing Steps

1.  Convert date columns to datetime format.
2.  Create a new column called `processing_time` (decision_date -
    application_date).
3.  Handle missing values:
    -   Numerical → Filled with median
    -   Categorical → Filled with mode
4.  Encode categorical variables using Label Encoding.
5.  Drop unnecessary columns.
6.  Save cleaned dataset.

## How to Run

Make sure you have installed:

pip install pandas numpy scikit-learn

Then run:

python visa_preprocessing.py

## Output

The script will generate: visa_dataset_cleaned.csv

This dataset is ready for machine learning models like: - Logistic
Regression - Decision Tree - Random Forest - XGBoost
