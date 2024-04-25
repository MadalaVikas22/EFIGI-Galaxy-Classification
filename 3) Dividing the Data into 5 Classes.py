## Creating 5 classes by combining the existing classes

# Mapping original classes to new classes
def type_splitter(n: int):

    if -6 <= n <= -4:
        n = 0
    elif -3 <= n <= -1:
        n = 1
    elif 0 <= n <= 9:
        n = 2
    elif n == 10:
        n = 3
    else:
        n = 4
    return n

# Mapping new class indices to class names
def type_splitter_by_name(n):

    labels = [
        "elliptical",
        "lenticular",
        "spiral",
        "irregular",
        "dwarf_elliptical"
    ]
    return labels[n]

# Applying the type splitter function to each element in the list
def split_types_bycategory(type_list):
    return [type_splitter(n) for n in type_list]

# Mapping new class indices to class names
def get_category_label_names(category_label_list):
    return [type_splitter_by_name(n) for n in category_label_list]

# Append a new column to the CSV with the split data.
types_list_original = np.array(data["T"])

# Get new class indices for each galaxy
types_list_bycategory = split_types_bycategory(types_list_original)
data["category_label"] = types_list_bycategory

data["category_label_name"] = get_category_label_names(types_list_bycategory)

# Select relevant columns for EFIGI labels
efigi_labels = data[["PGCname", "T", "category_label", "category_label_name"]]

# Count the number of galaxies in each new class
class_counts = efigi_labels["category_label"].value_counts()

# Plot a histogram of the new classes
plt.figure(figsize = (10, 6))
bars = plt.bar(range(0, 5), [class_counts.get(cls, 0) for cls in range(0, 5)])

plt.xlabel("Target-class")
plt.ylabel("Number of Galaxies")
plt.title("Number of Galaxies in each Class")
plt.xticks(range(0, 5), labels=range(0, 5))
# count on each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, yval, ha='center', va='bottom')
plt.show()
