from sklearn.datasets import load_iris
import pandas as pd

# Load dataset
iris = load_iris()

# Convert to DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target 

# print(df.head())  - first few rows of the dataset


# Save DataFrame to CSV
df.to_csv("iris_dataset.csv", index=False)  
# Saves to your working directory

# ANALYSIS SCRIPT
# Load libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('iris_dataset.csv')


