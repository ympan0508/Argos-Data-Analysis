import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load the dataset
train_df = pd.read_csv('train.csv')

# Categorize 'Age' into bins
age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]
age_labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80']
train_df['AgeGroup'] = pd.cut(train_df['Age'], bins=age_bins, labels=age_labels, right=False)

# Function to perform chi-squared test
def chi_squared_test(df, var1, var2):
    contingency_table = pd.crosstab(df[var1], df[var2])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return chi2, p, dof

# Perform chi-squared tests
chi2_pclass, p_pclass, dof_pclass = chi_squared_test(train_df, 'Survived', 'Pclass')
chi2_sex, p_sex, dof_sex = chi_squared_test(train_df, 'Survived', 'Sex')
chi2_age, p_age, dof_age = chi_squared_test(train_df, 'Survived', 'AgeGroup')

# Print results
print(f"Chi-squared test between 'Survived' and 'Pclass':")
print(f"Chi-squared statistic: {chi2_pclass}, p-value: {p_pclass}, Degrees of freedom: {dof_pclass}\n")

print(f"Chi-squared test between 'Survived' and 'Sex':")
print(f"Chi-squared statistic: {chi2_sex}, p-value: {p_sex}, Degrees of freedom: {dof_sex}\n")

print(f"Chi-squared test between 'Survived' and 'AgeGroup':")
print(f"Chi-squared statistic: {chi2_age}, p-value: {p_age}, Degrees of freedom: {dof_age}\n")
