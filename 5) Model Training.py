from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

# __________________Classification after feature selection with New Class Labels_______________
final_data = data[['Bulge_to_Total', 'Arm_Curvature', 'Arm_Rotation',
       'Perturbation', 'Flocculence', 'category_label']] #, 'Hot_Spots', 'category_label'
        # Hot spots attribute is not considered

print(final_data.shape)
print(final_data.head())

X= final_data.drop(columns = ['category_label'])
y= final_data['category_label']
x_train, x_test, y_train, y_test = train_test_split(X, y, shuffle = True, random_state=42, test_size = 0.2)
print('Classification after feature selection with New Class Labels: \n')
print(x_train.shape)
print(y_train.shape)

# DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
y_test_pred = clf.predict(x_test)
print("Decision Tree Classifier: \n")
print(classification_report(y_test, y_test_pred))

# RandomForestClassifier
rf_clf = RandomForestClassifier()
rf_clf.fit(x_train, y_train)
y_test_pred_rf = rf_clf.predict(x_test)
print("Random Forest Classifier: \n" )
print(classification_report(y_test, y_test_pred_rf))

# GradientBoostingClassifier
gb_clf = GradientBoostingClassifier()
gb_clf.fit(x_train, y_train)
y_test_pred_gb = gb_clf.predict(x_test)
print("Gradient Boosting Classifier: \n")
print(classification_report(y_test, y_test_pred_gb))

# Cross Validation
clf = DecisionTreeClassifier()
scores = cross_val_score(clf, X, y, cv=5)
print("Cross-validation scores :", scores)
print("Mean CV accuracy:", scores.mean())

# ______________________Classification after feature selection with Original Class Labels_____________________________

final_data = data[['Bulge_to_Total', 'Arm_Curvature', 'Arm_Rotation',
       'Perturbation', 'Flocculence', 'T']] #, 'Hot_Spots', 'category_label'
print(final_data.shape)
print(final_data.head())

X= final_data.drop(columns = ['T'])
y= final_data['T']

x_train, x_test, y_train, y_test = train_test_split(X, y, shuffle = True, random_state=42, test_size = 0.2)
print('Classification after feature selection with Original Class Labels: \n')
print(x_train.shape)
print(y_train.shape)

# DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
y_test_pred = clf.predict(x_test)
print("Decision Tree Classifier: \n")
print(classification_report(y_test, y_test_pred))

# RandomForestClassifier
rf_clf = RandomForestClassifier()
rf_clf.fit(x_train, y_train)
y_test_pred_rf = rf_clf.predict(x_test)
print("Random Forest Classifier: \n")
print(classification_report(y_test, y_test_pred_rf))

# GradientBoostingClassifier
gb_clf = GradientBoostingClassifier()
gb_clf.fit(x_train, y_train)
y_test_pred_gb = gb_clf.predict(x_test)
print("Gradient Boosting Classifier: \n")
print(classification_report(y_test, y_test_pred_gb))

# Define the Decision Tree Classifier
clf = DecisionTreeClassifier()
# 5-fold cross-validation
scores = cross_val_score(clf, X, y, cv=5)
print("Cross-validation scores:", scores)
print("Mean CV accuracy:", scores.mean())