from imblearn.over_sampling import SMOTE

# Classification after feature selection with New Class Labels
final_data = data[['Bulge_to_Total', 'Arm_Curvature', 'Arm_Rotation',
       'Perturbation', 'Flocculence', 'category_label']] #, 'Hot_Spots', 'category_label'

final_data['Multiplicative_Feature'] = final_data['Arm_Curvature'] * final_data['Arm_Rotation']
final_data.drop(columns = ["Arm_Curvature", "Arm_Rotation"], inplace = True) # Replacing Arm_curvature and Arm_rotation attributes with a
                                                                             # multiplicative feature
print(final_data.shape)
print(final_data.head())

# __________________Classification after feature selection with New Class Labels_______________

# Separate the features and the target variable
X = final_data.drop('category_label', axis=1)
y = final_data['category_label']

# Apply SMOTE to address class imbalance
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)
class_counts = y_resampled.value_counts()

# Class-wise histogram after applying SMOTE
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

x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
print('Classification after feature selection with New Class Labels: \n')
# DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
y_test_pred = clf.predict(x_test)
print("Decision Tree Classifier: \n")
print(classification_report(y_test, y_test_pred))

# Confusion Matrix
plt.figure(figsize=(6, 6))
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix - Decision Tree Classifier')
plt.show()
# Cross Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X_resampled, y_resampled, cv=5)
print("Cross-validation scores :", scores)
print("Mean CV accuracy:", scores.mean())



# ______________________Classification after feature selection with Original Class Labels_____________________________

# Separate the features and the target variable
X = final_data.drop('T', axis=1)
y = final_data['T']

# Apply SMOTE to address class imbalance
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
print('Classification after feature selection with Original Class Labels: \n')

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
y_test_pred = clf.predict(x_test)

print("Decision Tree Classifier: \n")
print(classification_report(y_test, y_test_pred))

plt.figure(figsize=(6, 6))
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.xticks(ticks=np.arange(18), labels=np.arange(-6, 12))
plt.yticks(ticks=np.arange(18), labels=np.arange(-6, 12))
plt.title('Confusion Matrix - Decision Tree Classifier')
plt.show()

class_counts = y_resampled.value_counts()

# Class-wise histogram after applying SMOTE
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

