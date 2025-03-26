import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Define the numerical columns to plot
numerical_columns = ['Age', 'SibSp', 'Parch', 'Fare']

# Create a 2x2 grid of subplots for the train dataset
plt.figure(figsize=(14, 10))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(2, 2, i)
    sns.histplot(train_df[column].dropna(), bins=30, kde=True, color='blue', label='Train')
    plt.title(f'Histogram of {column} (Train)')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.legend()
plt.tight_layout()
plt.savefig('train_histograms.png', dpi=100)

# Create a 2x2 grid of subplots for the test dataset
plt.figure(figsize=(14, 10))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(2, 2, i)
    sns.histplot(test_df[column].dropna(), bins=30, kde=True, color='green', label='Test')
    plt.title(f'Histogram of {column} (Test)')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.legend()
plt.tight_layout()
plt.savefig('test_histograms.png', dpi=100)
