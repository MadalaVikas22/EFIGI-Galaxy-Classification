# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the data
data = pd.read_csv(r'/Users/vikasvicky/PycharmProjects/pythonProject/EFIGI-Galaxy-Classification/EFIGI_attributes.csv')
data.drop(columns = ['Unnamed: 0'], axis =1, inplace=True)
data.head()

print(data.shape)
print(data.describe())
print(data.info())
print(data.columns)
print('No. of duplicate instances: ' + data.duplicated().sum())

## Histogram Plot for showing number of instances in each class
class_counts = data['T'].value_counts()

plt.figure(figsize = (10, 6))
bars = plt.bar(range(-6, 12), [class_counts.get(cls, 0) for cls in range(-6, 12)])
plt.xlabel("Target-class")
plt.ylabel("Number of Galaxies")
plt.title("Number of Galaxies in each Class")
plt.xticks(range(-6, 12), labels=range(-6, 12))
# Count on bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, yval, ha='center', va='bottom')
plt.show()

## Boxplot for each column
# Attributes to plot
columns_to_exclude = ['PGCname', 'T', 'T_inf', 'T_sup'] # T is our Target variable
columns_to_plot = data.drop(columns=columns_to_exclude)
# subplots
fig, axes = plt.subplots(16, 3, figsize=(12, 40))
axes = axes.flatten()

for i, col in enumerate(columns_to_plot.columns):
    axes[i].boxplot(columns_to_plot[col])
    axes[i].set_title(col)
    axes[i].set_ylabel('Value')
plt.tight_layout()
plt.show()

# Plotting Correlation Matrix
corr = data.drop("PGCname", axis = 1).corr()
plt.figure(figsize = (20, 20))
sns.heatmap(corr, cmap = 'Blues', fmt = '.1f', annot = True)

# Taking only the main attributes and excluding the inf
# and suf attributes

df_attributes = data.columns[4 : : 3]
print(df_attributes)