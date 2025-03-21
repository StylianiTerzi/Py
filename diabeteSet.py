import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#still working on it
data = pd.read_csv('diabetes.csv')
print("Missing values in each column:")
print(data.isnull().sum())
print("\nDataset Info:")
print(data.info())
plt.figure(figsize=(12, 8))
correlation_matrix = data.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True,cmap="YlGnBu")

#sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
#plt.title("Missing Values Heatmap")
plt.show()