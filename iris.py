import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

# Load the dataset
df = pd.read_csv('Iris.csv')

print("=== BASIC STATISTICS FOR SEPAL LENGTH (ALL SPECIES) ===")
print(f"Mean: {df['SepalLengthCm'].mean():.3f} cm")
print(f"Median: {df['SepalLengthCm'].median():.3f} cm")
print(f"Standard Deviation: {df['SepalLengthCm'].std():.3f} cm")
print()

# 1. Box plot for petal length across species
plt.figure(figsize=(10, 6))
sns.boxplot(x='Species', y='PetalLengthCm', data=df)
plt.title('Distribution of Petal Length Across Iris Species')
plt.ylabel('Petal Length (cm)')
plt.xlabel('Species')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("=== BOX PLOT INSIGHTS ===")
print("The box plot reveals:")
print("- Iris-setosa has the smallest petal length (1-2 cm) with compact distribution")
print("- Iris-versicolor has medium petal length (3-5 cm)")
print("- Iris-virginica has the largest petal length (4.5-7 cm)")
print("- Little overlap between setosa and other species")
print("- Some overlap between versicolor and virginica")
print("- Petal length is a good feature for species classification")
print()

# 2. Pearson correlation
correlation = df['SepalLengthCm'].corr(df['PetalLengthCm'])
print(f"=== CORRELATION ANALYSIS ===")
print(f"Pearson correlation coefficient: {correlation:.4f}")
print("Strong positive correlation: as sepal length increases, petal length tends to increase")
print()

# 3. Scatter plot with regression line
plt.figure(figsize=(10, 6))
sns.regplot(x='SepalLengthCm', y='PetalLengthCm', data=df, 
            scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
plt.title('Sepal Length vs Petal Length with Regression Line')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 4. Hypothesis testing - t-test for sepal length: setosa vs versicolor
setosa_sepal = df[df['Species'] == 'Iris-setosa']['SepalLengthCm']
versicolor_sepal = df[df['Species'] == 'Iris-versicolor']['SepalLengthCm']

# Perform two-sample t-test (assuming unequal variances)
t_stat, p_value = stats.ttest_ind(setosa_sepal, versicolor_sepal, equal_var=False)

print("=== HYPOTHESIS TESTING ===")
print(f"Setosa mean sepal length: {setosa_sepal.mean():.3f} cm")
print(f"Versicolor mean sepal length: {versicolor_sepal.mean():.3f} cm")
print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.2e}")
print()

print("=== CONCLUSION ===")
alpha = 0.05
if p_value < alpha:
    print(f"Since p-value ({p_value:.2e}) < α ({alpha}), we REJECT the null hypothesis.")
    print("There is significant evidence that mean sepal lengths differ between Iris-setosa and Iris-versicolor.")
else:
    print(f"Since p-value ({p_value:.2e}) ≥ α ({alpha}), we FAIL TO REJECT the null hypothesis.")
    print("There is no significant evidence that mean sepal lengths differ between Iris-setosa and Iris-versicolor.")

# Additional visualization: Compare distributions
plt.figure(figsize=(12, 8))

# Subplot 1: Petal length box plot
plt.subplot(2, 2, 1)
sns.boxplot(x='Species', y='PetalLengthCm', data=df)
plt.title('Petal Length by Species')
plt.xticks(rotation=45)

# Subplot 2: Sepal length box plot
plt.subplot(2, 2, 2)
sns.boxplot(x='Species', y='SepalLengthCm', data=df)
plt.title('Sepal Length by Species')
plt.xticks(rotation=45)

# Subplot 3: Scatter plot colored by species
plt.subplot(2, 2, 3)
colors = {'Iris-setosa': 'red', 'Iris-versicolor': 'blue', 'Iris-virginica': 'green'}
for species in df['Species'].unique():
    species_data = df[df['Species'] == species]
    plt.scatter(species_data['SepalLengthCm'], species_data['PetalLengthCm'], 
               label=species, alpha=0.7, c=colors[species])
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.title('Sepal vs Petal Length by Species')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 4: Distribution of sepal lengths for setosa vs versicolor
plt.subplot(2, 2, 4)
plt.hist(setosa_sepal, alpha=0.7, label='Iris-setosa', bins=10)
plt.hist(versicolor_sepal, alpha=0.7, label='Iris-versicolor', bins=10)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.title('Sepal Length: Setosa vs Versicolor')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()