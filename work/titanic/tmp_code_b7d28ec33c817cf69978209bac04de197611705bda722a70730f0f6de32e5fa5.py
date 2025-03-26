import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
train_df = pd.read_csv('train.csv')

# Calculate the survival rates for each combination of 'Pclass' and 'Sex'
survival_rates = train_df.groupby(['Pclass', 'Sex'])['Survived'].mean().unstack()

# Create a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(survival_rates, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'label': 'Survival Rate'})
plt.title('Survival Rates by Pclass and Sex')
plt.xlabel('Sex')
plt.ylabel('Pclass')
plt.savefig('survival_heatmap.png', dpi=100)
