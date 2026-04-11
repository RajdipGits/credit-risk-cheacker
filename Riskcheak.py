import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('credit_risk_dataset.csv')
print(dataset.head())
print(dataset.info())
print(dataset.describe())
print('\n checking')
print(dataset.isnull().sum())

# Data cleaning part ->
print("\nDATA CLEANING PART")

# AGE - remove below 18 and above 70
dataset = dataset[(dataset['person_age'] >= 18) & (dataset['person_age'] <= 70)]

# INCOME - remove outliers and fill median
dataset['person_income'] = dataset['person_income'].clip(
    dataset['person_income'].quantile(0.01),
    dataset['person_income'].quantile(0.99)
)
dataset['person_income'] = dataset['person_income'].fillna(
    dataset['person_income'].median()
).astype(int)

# EMPLOYMENT LENGTH - fill median
dataset['person_emp_length'] = dataset['person_emp_length'].fillna(
    dataset['person_emp_length'].median()
).astype(int)

# HOUSE - convert to numbers
ownership_map = {
    'RENT': 0,
    'MORTGAGE': 1,
    'OWN': 2,
    'OTHER': 3
}
dataset['person_home_ownership'] = dataset['person_home_ownership'].map(ownership_map)

# LOAN PURPOSE - convert to numbers
loan_intent_map = {
    'DEBTCONSOLIDATION': 0,
    'PERSONAL': 1,
    'MEDICAL': 2,
    'EDUCATION': 3,
    'VENTURE': 4,
    'HOMEIMPROVEMENT': 5
}
dataset['loan_intent'] = dataset['loan_intent'].map(loan_intent_map)

# LOAN GRADE - convert to numbers
loan_grade_map = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6
}
dataset['loan_grade'] = dataset['loan_grade'].map(loan_grade_map)

# INTEREST RATE - fill mean
dataset['loan_int_rate'] = dataset['loan_int_rate'].fillna(
    dataset['loan_int_rate'].mean()
)

# EMPLOYMENT LENGTH - remove outliers using IQR
Q1 = dataset['person_emp_length'].quantile(0.25)
Q3 = dataset['person_emp_length'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
dataset = dataset[
    (dataset['person_emp_length'] >= lower) &
    (dataset['person_emp_length'] <= upper)
]

# DEFAULT ON FILE - convert Y/N to 1/0
dataset['cb_person_default_on_file'] = dataset['cb_person_default_on_file'].map({'Y': 1, 'N': 0})

# Drop any remaining nulls
dataset = dataset.dropna()
dataset = dataset.reset_index(drop=True)

print('\n', dataset.head())
print('\nFinal shape:', dataset.shape)
print('\nNull check after cleaning:')
print(dataset.isnull().sum())
print('\nData types:')
print(dataset.dtypes)

