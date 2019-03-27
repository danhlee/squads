# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import sys

# Load dataset
csv = "matches.csv"
names = ['b_top','b_jung','b_mid','b_bot','b_sup','r_top','r_jung','r_mid','r_bot','r_sup','class']
dataset = pandas.read_csv(csv, names=names)

# shape shows (numberOfTuples, numberOfFeatures)
print(dataset.shape)
# shows first 20 tuples
print(dataset.head(20))
print()

# descriptions
print('--== DESCRIPTION of DATASET ==--')
print(dataset.describe())
print('--== DESCRIPTION of DATASET ==--')
print()

# # class distribution
# print('--== CLASS DISTRIBUTION ==--')
# print(dataset.groupby('class').size())
# print('--== CLASS DISTRIBUTION ==--')
# print()

# # scatter plot matrix
# print('--== SCATTER PLOT MATRIX ==--')
# scatter_matrix(dataset)
# plt.show()
# print('--== SCATTER PLOT MATRIX ==--')
# print()

# Split-out validation dataset
# array = dataset.values
array = dataset.to_numpy()
print(array)
# arraySlice[start_row:end_row_exclusive, start_col:end_col_exclusive]
# X = tuples minus class
# Y = just the classes of each tuple
X = array[:,0:10]
Y = array[:,10]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
print('Here come the comparisons...')
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print('mean(stdev) = ',msg)

# # Compare Algorithms
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()

# # Make predictions on validation dataset (using KNN)
# knn = KNeighborsClassifier()
# knn.fit(X_train, Y_train)
# predictions = knn.predict(X_validation)

# Make predictions on validation dataset (using Decision Tree) -  0.91
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
predictions = dt.predict(X_validation)

# # Make predictions on validation dataset (using SVM) - 0.92
# svm = SVC(gamma='auto', probability=True)
# svm.fit(X_train, Y_train)
# predictions = svm.predict(X_validation)

# Print Prediction accuracy from using test set
print('prediction =', predictions)
print('accuracy = ', accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

#mock single tuple of features for classification
#single_match = [[13, 85, 22, 154, 4, 76, 23, 94, 12, 61]]

#trial_1 [13,85,22,154,4,76,23,94,12,61]
#trial_2 [240,29,63,64,1,24,238,432,51,17] => class should be [100] (from training set)

# [Feature: Array input & single match prediction]
# def convertStringArgsToArray(stringArg):
#   return list(map(int, stringArg.strip('[]').split(',')))
  
# #take 1 tuple of features from command line argument
# print('length of argv =',len(sys.argv))
# print('argv =', sys.argv)
# if len(sys.argv) == 2:
#   single_match = convertStringArgsToArray(sys.argv[1])
#   matches = []
#   matches.append(single_match)
#   single_prediction = svm.predict(matches)
#   class_probabilities = svm.predict_proba(matches)
  
#   print('prediction for single match =', single_prediction)
#   print('confidence for single match =', class_probabilities)
#   print('classes of probabilities =', svm.classes_)
  
#   # TODO 1: not sure if probability is accurate ()
#   # Trial_2 prediction = 100 (which is right), but ...
#   # probability score says .98 for [200]
#   # probability score says .017 for [100]
#   # shouldn't this be reversed?
#   print(pandas.DataFrame(svm.predict_proba(matches), columns=svm.classes_))
  
# else:
#   print('no args passed in...')
