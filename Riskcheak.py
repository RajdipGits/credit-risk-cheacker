import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


dataset = pd.read_csv('credit_risk_dataset.csv')
print("Raw shape:", dataset.shape)


# Data cleaning
dataset = dataset[
    (dataset['person_age'] >= 18) & (dataset['person_age'] <= 70)
]

dataset['person_income'] = dataset['person_income'].clip(
    dataset['person_income'].quantile(0.01),
    dataset['person_income'].quantile(0.99)
)
dataset['person_income'] = dataset['person_income'].fillna(
    dataset['person_income'].median()
).astype(int)

dataset['person_emp_length'] = dataset['person_emp_length'].fillna(
    dataset['person_emp_length'].median()
).astype(int)
Q1 = dataset['person_emp_length'].quantile(0.25)
Q3 = dataset['person_emp_length'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
dataset = dataset[
    (dataset['person_emp_length'] >= lower) &
    (dataset['person_emp_length'] <= upper)
]

ownership_map = {'RENT': 0, 'MORTGAGE': 1, 'OWN': 2, 'OTHER': 3}
dataset['person_home_ownership'] = dataset['person_home_ownership'].map(ownership_map)


loan_intent_map = {
    'DEBTCONSOLIDATION': 0, 
    'PERSONAL':           1,
    'MEDICAL':            2,
    'EDUCATION':          3,
    'VENTURE':            4,
    'HOMEIMPROVEMENT':   5, 
}
dataset['loan_intent'] = dataset['loan_intent'].map(loan_intent_map)

loan_grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
dataset['loan_grade'] = dataset['loan_grade'].map(loan_grade_map)

dataset['loan_int_rate'] = dataset['loan_int_rate'].fillna(
    dataset['loan_int_rate'].mean()
)
dataset['loan_percent_income'] = (dataset['loan_percent_income'] * 100).round(2)

dataset['cb_person_default_on_file'] = dataset['cb_person_default_on_file'].map(
    {'Y': 1, 'N': 0}
)


print('Cleaned Data---------\n',dataset.head())

# Train slipt part
X = dataset.drop('loan_status', axis=1)
y = dataset['loan_status']


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  
)


model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))