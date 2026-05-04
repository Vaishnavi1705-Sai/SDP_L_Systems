import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Data
df = pd.read_csv('set3_cleaned_new.csv', header=None).dropna()
# Target the feature columns
features = df.iloc[:, -6:] 
features.columns = ['Variance', 'Slope', 'Jaggedness', 'Range', 'AvgRaw', 'AvgTemp']

# --- TEST A: Correlation Heatmap ---
# Calculates the Pearson correlation coefficient between features
plt.figure(figsize=(10, 8))
correlation_matrix = features.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap\n(1.0 = Perfect Link | 0.0 = No Link)")
plt.show()

# --- TEST B: Pair Plot ---
# Shows scatter plots for every combination of features
print("Generating Pair Plot... this may take a few seconds.")
sns.pairplot(features, diag_kind='kde')
plt.suptitle("Feature Interaction Matrix (Pair Plot)", y=1.02)
plt.show()