# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Show first 5 rows
print("FIRST 5 ROWS")
print(df.head())

# Dataset information
print("\nDATASET INFO")
print(df.info())

# Missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Summary statistics
print("\nSUMMARY STATISTICS")
print(df.describe())

# Histogram
plt.figure(figsize=(6,4))
df['Age'].hist(bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.savefig("age_histogram.png")
plt.show()

# Boxplot
plt.figure(figsize=(6,4))
sns.boxplot(x=df['Fare'])
plt.title("Fare Boxplot")
plt.savefig("fare_boxplot.png")
plt.show()

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

# Pairplot
sns.pairplot(df[['Age', 'Fare', 'Pclass', 'Survived']])
plt.savefig("pairplot.png")
plt.show()

print("\nEDA COMPLETED SUCCESSFULLY")