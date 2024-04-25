# Removing the Data of the Spiral Galaxies as it is causing bias and imbalance to the dataset
data_no_spiral = data.drop(data[data["category_label_name"] == "spiral"].index)
print(data_no_spiral.head())

# Count the number of galaxies in each new class
class_counts = data_no_spiral["category_label"].value_counts()

plt.figure(figsize = (10, 6))
bars = plt.bar(range(0, 5), [class_counts.get(cls, 0) for cls in range(0, 5)])
plt.xlabel("Target-class")
plt.ylabel("Number of Galaxies")
plt.title("Number of Galaxies in each Class")
plt.xticks(range(0, 5), labels=range(0, 5))
# Count on bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, yval, ha='center', va='bottom')
plt.show()

# Taking only the main attributes and excluding the inf and suf attributes
df_attributes_no_spiral = data_no_spiral.columns[4::3]
df_attributes_no_spiral = df_attributes_no_spiral.drop('category_label')
primt(df_attributes_no_spiral)

# Data Splitting
X= data_no_spiral[df_attributes_no_spiral]
y= data_no_spiral['category_label']

x_train_no_spiral, x_test_no_spiral, y_train_no_spiral, y_test_no_spiral = train_test_split(X, y, shuffle = True, random_state=42, test_size = 0.2)
print(x_train_no_spiral.shape)

# ANOVa F-value
selector_no_spiral = SelectKBest(f_classif, k=10)
selector_no_spiral.fit(x_train_no_spiral, y_train_no_spiral)
x_new = selector_no_spiral.transform(x_train_no_spiral)
print(x_new.shape)

## Comparison Plot for Data with and without Spiral Galaxies
# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# Plot selected features for ANOVA F-value without Spiral
axes[0].bar(range(len(selector_no_spiral.scores_)), selector_no_spiral.scores_)
axes[0].set_xticks(range(len(selector_no_spiral.scores_)))
axes[0].set_xticklabels(df_attributes, rotation='vertical')
axes[0].set_title("ANOVA F-value feature selection without Spiral")

# Plot selected features for ANOVA F-value with Spiral
axes[1].bar(range(len(initial_selector.scores_)), initial_selector.scores_)
axes[1].set_xticks(range(len(initial_selector.scores_)))
axes[1].set_xticklabels(df_attributes, rotation='vertical')
axes[1].set_title("ANOVA F-value feature selection with Spiral")
plt.tight_layout()
plt.show()
