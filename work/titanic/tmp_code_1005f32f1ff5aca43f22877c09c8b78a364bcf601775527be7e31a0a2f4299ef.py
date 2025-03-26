import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
train_df = pd.read_csv('train.csv')

# Calculate survival rates for Pclass
pclass_survival = train_df.groupby('Pclass')['Survived'].mean().reset_index()
pclass_survival.columns = ['Pclass', 'SurvivalRate']

# Calculate survival rates for Sex
sex_survival = train_df.groupby('Sex')['Survived'].mean().reset_index()
sex_survival.columns = ['Sex', 'SurvivalRate']

# Bin the Age column and calculate survival rates for each age group
train_df['AgeGroup'] = pd.cut(train_df['Age'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80], labels=['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80'])
age_survival = train_df.groupby('AgeGroup')['Survived'].mean().reset_index()
age_survival.columns = ['AgeGroup', 'SurvivalRate']

# Plotting
plt.figure(figsize=(15, 5))

# Pclass survival rate bar plot
plt.subplot(1, 3, 1)
sns.barplot(x='Pclass', y='SurvivalRate', data=pclass_survival, palette='viridis')
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')

# Sex survival rate bar plot
plt.subplot(1, 3, 2)
sns.barplot(x='Sex', y='SurvivalRate', data=sex_survival, palette='plasma')
plt.title('Survival Rate by Sex')
plt.xlabel('Sex')
plt.ylabel('Survival Rate')

# AgeGroup survival rate bar plot
plt.subplot(1, 3, 3)
sns.barplot(x='AgeGroup', y='SurvivalRate', data=age_survival, palette='magma')
plt.title('Survival Rate by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Survival Rate')
plt.xticks(rotation=45)

# Save the figure
plt.tight_layout()
plt.savefig('survival_rates.png', dpi=100)
