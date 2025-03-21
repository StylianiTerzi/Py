import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#still working on it
dt=pd.read_csv('diabetes.csv')
print("Missing values in each column:")
print(dt.isnull().sum())
msdValue=dt.isnull().sum().sum()
if msdValue > 0:
    print("\nMissing values found in the dataset.")
    plt.figure(figsize=(10, 6))
    sns.heatmap(dt.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Values Heatmap")
    plt.show()
else:
    print("\nThere is no missing values found in the CSV file")
print("\nDataset Info:")
print(dt.info())
plt.figure(figsize=(12, 8))
correlation_matrix = dt.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True,cmap="YlGnBu")
plt.show()