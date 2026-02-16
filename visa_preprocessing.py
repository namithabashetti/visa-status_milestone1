
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("visastatus_dataset.csv")

# -------------------------
# Convert Date Columns
# -------------------------
df['application_date'] = pd.to_datetime(df['application_date'], errors='coerce')
df['decision_date'] = pd.to_datetime(df['decision_date'], errors='coerce')

# -------------------------
# Create Processing Time Column
# -------------------------
df['processing_time'] = (df['decision_date'] - df['application_date']).dt.days

# -------------------------
# Handle Missing Values
# -------------------------

# Numerical columns → fill with median
num_cols = ['age', 'processing_time']
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical columns → fill with mode
cat_cols = ['applicant_country', 'visa_type',
            'education_level', 'employment_status',
            'sponsor_company', 'visa_status']

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# -------------------------
# Encode Categorical Variables
# -------------------------

label_encoder = LabelEncoder()

for col in cat_cols:
    df[col] = label_encoder.fit_transform(df[col])

# -------------------------
# Drop Unnecessary Columns
# -------------------------
df = df.drop(['application_id', 'application_date', 'decision_date'], axis=1)

# -------------------------
# Save Cleaned Dataset
# -------------------------
df.to_csv("visa_dataset_cleaned.csv", index=False)

print("Preprocessing Completed Successfully!")
print(df.head())
