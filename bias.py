import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Load the Adult dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

data = pd.read_csv(url, names=column_names, sep=', ', engine='python')

# Convert target to binary
data['income'] = data['income'].apply(lambda x: 1 if x == '>50K' else 0)

# Print bias before mitigation
print("Original data distribution by sex:")
print(data.groupby(['sex', 'income']).size())

# Save sex column for analysis
sex_column = data['sex'].copy()

# One-hot encode categorical features
data_encoded = pd.get_dummies(data, columns=['workclass', 'education', 'marital-status',
                                              'occupation', 'relationship', 'race', 'native-country'])

# Define features and target
X = data_encoded.drop(['income', 'sex'], axis=1)
y = data_encoded['income']

# Convert sex_column to NumPy array to ensure alignment
sex_column_array = sex_column.values

# Train-test split (stratify by sex for fair gender distribution)
X_train, X_test, y_train, y_test, sex_train, sex_test = train_test_split(
    X, y, sex_column_array, test_size=0.2, random_state=42, stratify=sex_column_array)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train initial model (before bias mitigation)
model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train_scaled, y_train)

# Predict and evaluate before mitigation
y_pred = model.predict(X_test_scaled)

# Create masks for male and female
male_mask = sex_test == ' Male'
female_mask = sex_test == ' Female'

print("\nBefore mitigation:")
print(f"Overall accuracy: {accuracy_score(y_test, y_pred):.4f}")
if sum(male_mask) > 0:
    print(f"Male accuracy: {accuracy_score(y_test[male_mask], y_pred[male_mask]):.4f}")
if sum(female_mask) > 0:
    print(f"Female accuracy: {accuracy_score(y_test[female_mask], y_pred[female_mask]):.4f}")

# Bias mitigation via reweighting
male_count = sum(sex_train == ' Male')
female_count = sum(sex_train == ' Female')

print(f"\nTraining data distribution: {male_count} males, {female_count} females")

if male_count == 0 or female_count == 0:
    print("Warning: Missing gender samples. Using equal weights.")
    sample_weights = np.ones(len(y_train))
else:
    # Calculate weights to balance the classes
    male_weight = 1.0
    female_weight = male_count / female_count
    print(f"Using weights: Male={male_weight:.2f}, Female={female_weight:.2f}")
    sample_weights = np.array([
        male_weight if sex == ' Male' else female_weight for sex in sex_train
    ])

# Retrain model with bias mitigation
model_mitigated = LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced')
model_mitigated.fit(X_train_scaled, y_train, sample_weight=sample_weights)

# Predict and evaluate after mitigation
y_pred_mitigated = model_mitigated.predict(X_test_scaled)

print("\nAfter mitigation and improvement:")
print(f"Overall accuracy: {accuracy_score(y_test, y_pred_mitigated):.4f}")
if sum(male_mask) > 0:
    print(f"Male accuracy: {accuracy_score(y_test[male_mask], y_pred_mitigated[male_mask]):.4f}")
if sum(female_mask) > 0:
    print(f"Female accuracy: {accuracy_score(y_test[female_mask], y_pred_mitigated[female_mask]):.4f}")
