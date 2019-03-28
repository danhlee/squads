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
from random import randint

import sys

TREE = 'TREE'
BAYES = 'BAYES'

# creates model using input parameter (string) and
def get_model(model_name):
  dataset = load_csv()
  describe_data(dataset)
  dataArray = dataset.to_numpy()
  features = dataArray[:,0:10]
  classes = dataArray[:,10]
  validation_size = 0.20
  seed = randint(1,10)
  print()
  print('using seed value:', seed)
  # split dataset into 80% training and 20% testing (also split by features and class for each)
  features_train, features_validation, classes_train, classes_validation = model_selection.train_test_split(features, classes, test_size=validation_size, random_state=seed)
  
  # create model dynamically (default is decision tree)
  if (model_name == 'TREE'):
    model = DecisionTreeClassifier()
  elif (model_name == 'BAYES'):
    model = GaussianNB()
  else:
    model = DecisionTreeClassifier()
    model_name = 'TREE'

  scoring = 'accuracy'
  kfold = model_selection.KFold(n_splits=10, random_state=seed)
  cv_results = model_selection.cross_val_score(model, features_train, classes_train, cv=kfold, scoring=scoring)
  msg = "%s: %f (%f)" % (model_name, cv_results.mean(), (cv_results.std()))
  print()
  print('--== CROSS-VALIDATION RESULTS ==--')
  print('model_name, mean, stdev =', msg)
  
  # evaluate model with validation set and print results
  evaluate_model(model, model_name, features_train, classes_train, features_validation, classes_validation)

  return model

# loads matches.csv
# returns DataFrame
def load_csv():
  csv = "matches.csv"
  names = ['b_top','b_jung','b_mid','b_bot','b_sup','r_top','r_jung','r_mid','r_bot','r_sup','class']
  dataset = pandas.read_csv(csv, names=names)
  return dataset

def describe_data(dataset):
  # shape shows (numberOfTuples, numberOfFeatures)
  print('--== (#TUPLES, #FEATURES) ==--')
  print(dataset.shape)
  # shows first 20 tuples
  print()
  print('--== FIRST 20 TUPLES ==--')
  print(dataset.head(20))
  print()
  print('--== DESCRIPTION of DATASET ==--')
  print(dataset.describe())
  print()

def train_models_and_compare(dataset):
  dataArray = dataset.to_numpy()
  # arraySlice[start_row:end_row_exclusive, start_col:end_col_exclusive]
  # features = tuples minus class
  # classes = just the classes of each tuple
  features = dataArray[:,0:10]
  classes = dataArray[:,10]
  #20 percent of tuples reserved for validation
  validation_size = 0.20
  seed = randint(1,10)
  print('using seed value:', seed)
  features_train, features_validation, classes_train, classes_validation = model_selection.train_test_split(features, classes, test_size=validation_size, random_state=seed)

  models = []
  # Logistic Regression classifier
  models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
  # Linear Discriminant Analysis
  models.append(('LDA', LinearDiscriminantAnalysis()))
  # K-nearest neighbors classifier
  models.append(('KNN', KNeighborsClassifier()))
  # Decision tree classifier
  models.append(('TREE', DecisionTreeClassifier()))
  # Gaussian Naive Bayes
  models.append(('BAYES', GaussianNB()))
  # Support Vector Classification
  models.append(('SVM', SVC(gamma='auto')))

  print('--== MODEL ACCURACY COMPARISONS ==--')
  results = []
  names = []
  scoring = 'accuracy'
  for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, features_train, classes_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print('model_name: mean (stdev)  =', msg)
    print('VALIDATION SET for', name)
    evaluate_model(model, name, features_train, classes_train, features_validation, classes_validation)
    
def evaluate_model(model, name, features_train, classes_train, features_validation, classes_validation):
  model.fit(features_train, classes_train)
  class_predictions = model.predict(features_validation)
  print()
  print('--== VALIDATION for', name, '==--')
  print('...class predictions for validation set =', class_predictions)
  print('...accuracy of predictions =', accuracy_score(classes_validation, class_predictions))
  print()
  print('--== CONFUSION MATRIX ==--')
  print(confusion_matrix(classes_validation, class_predictions))
  print()
  print('--== CLASSIFICATION REPORT ==--')
  print(classification_report(classes_validation, class_predictions))

# single_match_roster = JSON request
# returns prediction 100 or 200 as JSON response ie - {"winner": "100"}
def get_prediction(model, json_roster):
  array_roster = json_roster_to_array(json_roster)
  class_prediction = model.predict(array_roster)
  
  # probability / confidence?
  # class_probabilities = svm.predict_proba(matches)

  prediction_json = {
    "winner": str(class_prediction[0])
  }
  print()
  print('[ MAKING PREDICTION ] =', prediction_json)

  return prediction_json

# create array roster that only contains the 10 players' champions in row order
def json_roster_to_array(json_roster):
  array_roster = []
  array_roster.append(json_roster["b_top"])
  array_roster.append(json_roster["b_jung"])
  array_roster.append(json_roster["b_mid"])
  array_roster.append(json_roster["b_bot"])
  array_roster.append(json_roster["b_sup"])
  array_roster.append(json_roster["r_top"])
  array_roster.append(json_roster["r_jung"])
  array_roster.append(json_roster["r_mid"])
  array_roster.append(json_roster["r_bot"])
  array_roster.append(json_roster["r_sup"])
  matches = [array_roster]
  return matches