import pandas as pd

# Load the datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Define the numerical columns of interest
numerical_columns = ['Age', 'SibSp', 'Parch', 'Fare']

# Function to compute and print statistics for a given DataFrame
def print_statistics(df, name):
    print(f"Statistics for {name} dataset:")
    for column in numerical_columns:
        mean = df[column].mean()
        median = df[column].median()
        std = df[column].std()
        min_val = df[column].min()
        max_val = df[column].max()
        print(f"Column: {column}")
        print(f"  Mean: {mean}")
        print(f"  Median: {median}")
        print(f"  Standard Deviation: {std}")
        print(f"  Min: {min_val}")
        print(f"  Max: {max_val}")
    print("\n")

# Print statistics for the train dataset
print_statistics(train_df, 'Train')

# Print statistics for the test dataset
print_statistics(test_df, 'Test')
