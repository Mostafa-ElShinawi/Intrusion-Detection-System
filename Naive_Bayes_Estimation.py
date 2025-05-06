import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.stats._continuous_distns import _distn_names
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data
file_path = 'C:\Users\Mostafa\Desktop\guc\random\Train_data.csv'
data = pd.read_csv(file_path)

# Data Preparation
X = data.drop('class', axis=1)
y = data['class']
Z = X.select_dtypes(include=[np.number])  # Numerical columns only

# Split into train/test sets
train_size = int(0.7 * len(data))
train_indices = np.random.choice(data.index, size=train_size, replace=False)
test_indices = data.index.difference(train_indices)

X_train = Z.loc[train_indices]
y_train = y.loc[train_indices]
X_test = Z.loc[test_indices]
y_test = y.loc[test_indices]

# Identify numerical and discrete columns
numerical_cols = X_train.columns
discrete_cols = [col for col in numerical_cols if X_train[col].nunique() <= 10]

# Functions for PDF and PMF Calculation
def fit_pdf(data):
    """Fit the best distribution to the data and return the distribution and parameters."""
    best_distribution = None
    best_params = None
    min_mse = np.inf
    y, x = np.histogram(data, bins=50, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    for dist_name in ['norm', 'lognorm', 'gamma', 'beta', 'expon']:
        try:
            dist = getattr(st, dist_name)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                params = dist.fit(data)
                pdf = dist.pdf(x, *params)
                mse = np.mean((y - pdf) ** 2)
                if mse < min_mse:
                    min_mse = mse
                    best_distribution = dist
                    best_params = params
        except Exception:
            pass
    return best_distribution, best_params

def compute_pdf_value(distribution, params, value):
    """Compute the PDF value for a given distribution and parameters."""
    try:
        return distribution.pdf(value, *params)
    except:
        return 1e-6  # To avoid zero probability

def compute_pmf(data, value):
    """Compute PMF for categorical/discrete columns."""
    counts = data.value_counts(normalize=True)
    return counts.get(value, 1e-6)

# Step 1: Fit PDFs for numerical columns conditioned on class
pdfs_anomaly = {}
pdfs_normal = {}
pdfs_total = {}

for col in numerical_cols:
    pdfs_anomaly[col] = fit_pdf(X_train[y_train == 'anomaly'][col].dropna())
    pdfs_normal[col] = fit_pdf(X_train[y_train == 'normal'][col].dropna())
    pdfs_total[col] = fit_pdf(X_train[col].dropna())

# Step 2: PMFs for discrete columns conditioned on class
pmfs_anomaly = {}
pmfs_normal = {}
pmfs_total = {}

for col in discrete_cols:
    pmfs_anomaly[col] = X_train[y_train == 'anomaly'][col].value_counts(normalize=True)
    pmfs_normal[col] = X_train[y_train == 'normal'][col].value_counts(normalize=True)
    pmfs_total[col] = X_train[col].value_counts(normalize=True)


# Step 3: Naïve Bayes Prediction Function
def predict_naive_bayes(row):
    # Start with class probabilities (assuming equal prior probabilities)
    pr_anomaly = len(X_test[y_test == 'anomaly']) / len(X_test)
    pr_no_anomaly = len(X_test[y_test == 'normal']) / len(X_test)
    denom = 1

    for col in numerical_cols:
        value = row[col]
        dist_anomaly, params_anomaly = pdfs_anomaly[col]
        dist_normal, params_normal = pdfs_normal[col]
        dist_total, params_total = pdfs_total[col]

        pr_anomaly *= compute_pdf_value(dist_anomaly, params_anomaly, value)
        pr_no_anomaly *= compute_pdf_value(dist_normal, params_normal, value)
        denom *= compute_pdf_value(dist_total, params_total, value)

    for col in discrete_cols:
        value = row[col]
        pr_anomaly *= pmfs_anomaly[col].get(value, 1e-6)
        pr_no_anomaly *= pmfs_normal[col].get(value, 1e-6)
        denom *= pmfs_total[col].get(value, 1e-6)
    
    pr = pr_anomaly / denom
    
    print(f"Probabilty of Anomaly at this row: {pr}")
    return 'anomaly' if pr_anomaly > pr_no_anomaly else 'normal'

# Step 4: Make Predictions on Test Data
predictions = X_test.apply(predict_naive_bayes, axis=1)

# Step 5: Performance Metrics
TP = np.sum((y_test == 'anomaly') & (predictions == 'anomaly'))
TN = np.sum((y_test == 'normal') & (predictions == 'normal'))
FP = np.sum((y_test == 'normal') & (predictions == 'anomaly'))
FN = np.sum((y_test == 'anomaly') & (predictions == 'normal'))

accuracy = (TP + TN) / len(y_test)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0

# Output Results
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')

#--------------------------------------------------------------------------------------------------------------------------

# Separate features and target
X = data.drop('class', axis=1)
y = data['class']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=[np.number]).columns

# One-hot encode the categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'  # Keep numerical columns as is
)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define Naïve Bayes models
models = {
    'GaussianNB': GaussianNB(),
    'MultinomialNB': MultinomialNB(),
    'BernoulliNB': BernoulliNB()
}

# Evaluate models
results = {}

for model_name, model in models.items():
    # Build a pipeline with the preprocessor and the model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Train the model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(X_train, y_train)

    # Predict on the test set
    y_pred = pipeline.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='anomaly')
    recall = recall_score(y_test, y_pred, pos_label='anomaly')

    # Store results
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall
    }

# Output Results
for model_name, metrics in results.items():
    print(f'\n{model_name}:')
    print(f'Accuracy: {metrics["Accuracy"]:.2f}')
    print(f'Precision: {metrics["Precision"]:.2f}')
    print(f'Recall: {metrics["Recall"]:.2f}')