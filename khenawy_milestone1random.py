import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom
from scipy.stats import f_oneway
from matplotlib.colors import LogNorm
import seaborn as sns

file_path = 'Train_data.csv'
data = pd.read_csv(file_path)

data_fields = data.columns.tolist()
print("Data Fields:", data_fields)

#-----------------------------------------------------------------

data_types = data.dtypes
print("\nData Types:\n", data_types)

#-----------------------------------------------------------------

missing_data = data.isnull().sum()
inf_data = data.select_dtypes(include=[np.number]).apply(np.isinf).sum()
print("\nMissing Data:\n", missing_data)
print("\nInfinite Data:\n", inf_data)

#-----------------------------------------------------------------

categorical_counts = data.select_dtypes(include=['object']).nunique()
print("\nNumber of Categories in Each Categorical Field:\n", categorical_counts)

#-----------------------------------------------------------------

stats = data.describe().T[['max', 'min', 'mean', 'std']]
stats['variance'] = stats['std'] ** 2
print("\nMaximum, Minimum, Average and Variance of Each Numeric Field:\n", stats)

#-----------------------------------------------------------------

numeric_cols = data.select_dtypes(include=[np.number]).columns
continous_cols = [col for col in numeric_cols if data[col].nunique() > 10]
discrete_cols = [col for col in numeric_cols if data[col].nunique() <= 10] + ['protocol_type','service','flag','class']
subset = data[data['class'] == 'anomaly']

#-----------------------------------------------------------------

for col in numeric_cols:
    # Use pd.cut to ensure exactly 4 bins
    data[col + '_quartiles'] = pd.cut(data[col], bins=4, labels=False, duplicates='drop')

    quarter_stats = data.groupby(col + '_quartiles')[col].agg(['max', 'min', 'mean', 'var'])
    print(f"\nStats for {col} by Quartiles:\n", quarter_stats)

#-----------------------------------------------------------------

for col in discrete_cols:
    values, counts = np.unique(data[col], return_counts=True)
    pmf = counts / counts.sum()
    plt.bar(values, pmf, color='b', label='all data')
    plt.title(f'PMF of {col}')
    plt.xlabel(col)
    plt.ylabel('Probability')
    plt.show()

#-----------------------------------------------------------------

for col in continous_cols:
    density = norm.pdf(sorted(data[col]), np.mean(data[col]), np.std(data[col]))
    plt.plot(sorted(data[col]), density, color='b', label='Without Attack')
    plt.title(f'PDF of {col}')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.show()

#-----------------------------------------------------------------

for col in numeric_cols:
    sorted_data = np.sort(data[col])
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, cdf)
    plt.title(f'CDF of {col}')
    plt.xlabel(col)
    plt.ylabel('CDF')
    plt.show()
    
#-----------------------------------------------------------------

for col in discrete_cols:
    values, counts = np.unique(data[col], return_counts=True)
    values_attacked, counts_attacked = np.unique(subset[col], return_counts=True)
    pmf = counts / counts.sum()
    pmf_attacked = counts_attacked / counts_attacked.sum()
    plt.bar(values, pmf, color='b', label='all data')
    plt.bar(values_attacked, pmf_attacked, color='r', label='attacked data')
    plt.title(f'PMF of {col}')
    plt.xlabel(col)
    plt.ylabel('Probability')
    plt.legend()
    plt.show()

#-----------------------------------------------------------------

for col in continous_cols:
    density = norm.pdf(sorted(data[col]), np.mean(data[col]), np.std(data[col]))
    density_attacked = norm.pdf(sorted(subset[col]), np.mean(subset[col]), np.std(subset[col]))
    plt.plot(sorted(data[col]), density, color='b', label='Without Attack')
    plt.plot(sorted(subset[col]), density_attacked, color='r', label='With Attack')
    plt.title(f'PDF of {col}')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.legend()
    plt.show()

#-----------------------------------------------------------------

# Scatter plot between 'src_bytes' and 'dst_bytes'
plt.figure(figsize=(8, 5))
plt.scatter(data['count'], data['srv_count'], alpha=0.5, color='blue')
plt.title('Scatter Plot: src_bytes vs dst_bytes')
plt.xlabel('Source Bytes (src_bytes)')
plt.ylabel('Destination Bytes (dst_bytes)')
plt.grid(True)
plt.show()

# Define ranges for 'count' and 'srv_count' to exclude extreme outliers
count_range = (0, data['count'].max())
srv_count_range = (0, data['srv_count'].max())

# Step 1: Create a 2D histogram with better color contrast using LogNorm
plt.figure(figsize=(8, 6))
counts, xedges, yedges, im = plt.hist2d(
    data['count'], 
    data['srv_count'], 
    bins=50,  # Number of bins for each axis
    range=[count_range, srv_count_range],  # Set the range for both axes
    density=True,  # Normalize the counts to form a probability density
    cmap='viridis',  # Use a more contrasting color map
    norm=LogNorm()  # Apply logarithmic normalization to make small values visible
)

# Step 2: Add a colorbar to represent the density
plt.colorbar(im, label='Density')

# Step 3: Add labels and title
plt.title('Joint PDF of count and srv_count (2D Histogram with LogNorm)')
plt.xlabel('Count (Connections to Same Destination)')
plt.ylabel('Service Count (Connections to Same Service)')

# Step 4: Show the plot
plt.show()

#----------------------------------------------------------------

# Define ranges for 'count' and 'srv_count' to exclude extreme outliers
count_range = (0, subset['count'].max())
srv_count_range = (0, subset['srv_count'].max())

# Step 1: Create a 2D histogram with better color contrast using LogNorm
plt.figure(figsize=(8, 6))
counts, xedges, yedges, im = plt.hist2d(
    subset['count'], 
    subset['srv_count'], 
    bins=50,  # Number of bins for each axis
    range=[count_range, srv_count_range],  # Set the range for both axes
    density=True,  # Normalize the counts to form a probability density
    cmap='viridis',  # Use a more contrasting color map
    norm=LogNorm()  # Apply logarithmic normalization to make small values visible
)

# Step 2: Add a colorbar to represent the density
plt.colorbar(im, label='Density')

# Step 3: Add labels and title
plt.title('Joint PDF of count and srv_count with attack (2D Histogram with LogNorm)')
plt.xlabel('Count (Connections to Same Destination)')
plt.ylabel('Service Count (Connections to Same Service)')

# Step 4: Show the plot
plt.show()

# Calculating Pearson correlation matrix for all numerical columns
correlation_matrix = data.select_dtypes(include=[np.number]).corr(method='pearson')

# Step 2: Display the correlation matrix
print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# Step 1: Select relevant continuous fields and the attack type
continuous_fields = continous_cols  # Add more if needed

# Step 2: Perform ANOVA for each field
for field in continuous_fields:
    groups = [group[field].values for name, group in data.groupby('class')]
    
    # Perform ANOVA
    f_stat, p_value = f_oneway(*groups)
    
    # Print results
    print(f"ANOVA for {field}:")
    print(f"F-Statistic: {f_stat}, p-value: {p_value}")
    if p_value < 0.05:
        print(f"{field} is dependent on the attack type (Reject H0)")
    else:
        print(f"{field} is independent of the attack type (Fail to Reject H0)")
    print("\n")