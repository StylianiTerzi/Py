import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file="penguins.csv"
df = pd.read_csv(file)
print(df) 
print(df.head())
print(df.columns)
print(df['species'].value_counts())
print(df['sex'].value_counts())
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='species')
plt.title('Penguin Species Distribution')
plt.xlabel('Species')
plt.ylabel('Count')
plt.xticks(rotation=45) 
sns.pairplot(df, hue="species")
plt.show()
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='sex')
plt.title('Penguin Sex Distribution')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.xticks(rotation=45) 
sns.pairplot(df, hue="sex") 
plt.show()
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='island')
plt.title('Penguins location')
plt.xlabel('islands')
plt.ylabel('Count')
plt.xticks(rotation=45) 
plt.show()
