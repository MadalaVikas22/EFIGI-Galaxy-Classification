from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

# Splitting the data
X= data[df_attributes]
y= data['T']
x_train_initial, x_test_initial, y_train_initial, y_test_initial = train_test_split(X, y, shuffle = True, random_state=42, test_size = 0.2)
print(x_train_initial.shape)

# ANOVa F-value
initial_selector = SelectKBest(f_classif, k=10)
initial_selector.fit(x_train_initial, y_train_initial)
x_new_initial = initial_selector.transform(x_train_initial)
print(x_new_initial.shape)

#plot selected features for ANOVA F-value
plt.figure()
plt.bar(range(len(initial_selector.scores_)), initial_selector.scores_)
plt.xticks(range(len(initial_selector.scores_)), df_attributes, rotation = 'vertical')
plt.title("ANOVA F-value feature selection - Initial")
plt.show()

# Model
clf = DecisionTreeClassifier()
clf.fit(x_train_initial, y_train_initial)
y_test_pred_initial = clf.predict(x_test_initial)
print(classification_report(y_test_initial, y_test_pred_initial))

# F1 Score
F1 = f1_score(y_test_initial, y_test_pred_initial, average='macro') # Since there is a lot of imbalance in the Data, F1 Score is a good measure
print(f"F1-Score : {F1:.3f}")