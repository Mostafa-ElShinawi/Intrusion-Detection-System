import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.stats._continuous_distns import _distn_names

# Load the data
file_path = 'Train_data.csv'
data = pd.read_csv(file_path)

# Data Preparation
X = data.drop('class', axis=1)
y = data['class']
X = X.select_dtypes(include=[np.number])  # Numerical columns only

# Split into train/test sets
train_size = int(0.7 * len(data))
train_indices = np.random.choice(data.index, size=train_size, replace=False)
test_indices = data.index.difference(train_indices)

X_train = X.loc[train_indices]
y_train = y.loc[train_indices]
X_test = X.loc[test_indices]
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

for col in numerical_cols:
    pdfs_anomaly[col] = fit_pdf(X_train[y_train == 'anomaly'][col].dropna())
    pdfs_normal[col] = fit_pdf(X_train[y_train == 'normal'][col].dropna())

# Step 2: PMFs for discrete columns conditioned on class
pmfs_anomaly = {}
pmfs_normal = {}

for col in discrete_cols:
    pmfs_anomaly[col] = X_train[y_train == 'anomaly'][col].value_counts(normalize=True)
    pmfs_normal[col] = X_train[y_train == 'normal'][col].value_counts(normalize=True)

# Step 3: Naïve Bayes Prediction Function
def predict_naive_bayes(row):
    # Start with class probabilities (assuming equal prior probabilities)
    pr_anomaly = 1.0
    pr_no_anomaly = 1.0

    for col in numerical_cols:
        value = row[col]
        dist_anomaly, params_anomaly = pdfs_anomaly[col]
        dist_normal, params_normal = pdfs_normal[col]

        pr_anomaly *= compute_pdf_value(dist_anomaly, params_anomaly, value)
        pr_no_anomaly *= compute_pdf_value(dist_normal, params_normal, value)

    for col in discrete_cols:
        value = row[col]
        pr_anomaly *= pmfs_anomaly[col].get(value, 1e-6)
        pr_no_anomaly *= pmfs_normal[col].get(value, 1e-6)

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