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
import pickle
import sys
import os
from data import generateCsv
TREE = 'TREE'
BAYES = 'BAYES'


# loads matches.csv
# returns DataFrame
def loadCsv():
  csv = "matches.csv"
  names = ['b_top','b_jung','b_mid','b_bot','b_sup','r_top','r_jung','r_mid','r_bot','r_sup','winner']
  dataset = pandas.read_csv(csv, names=names)
  return dataset

# creates a model specified by model_name parameter and saves as model_name.pkl file (overwrites .pkl file)
def trainModel(model_name):
  generateCsv()
  # create dataframe and convert to 2d array
  dataset = loadCsv()
  describeData(dataset)
  dataArray = dataset.to_numpy()

  # slice array into features set and classes set
  features = dataArray[:,0:10]
  classes = dataArray[:,10]
  validation_size = 0.20
  seed = randint(1,10)


  # split dataset into 80% training and 20% testing (also split by features and class for each)
  features_train, features_validation, classes_train, classes_validation = model_selection.train_test_split(features, classes, test_size=validation_size, random_state=seed)
  
  # create model dynamically (default is decision tree)
  if (model_name == 'TREE'):
    print()
    print('...creating decision tree model')
    model = DecisionTreeClassifier()
  elif (model_name == 'BAYES'):
    print()
    print('...creating naive bayesian model')
    model = GaussianNB()
  else:
    print()
    print('...bad model name given...creating decision tree model by default')
    model = DecisionTreeClassifier()
    model_name = 'TREE'
  
  print()
  print('...performing cross-validation using k-Folds with 10 splits')
  print()
  print('...using seed value:', seed)
  scoring = 'accuracy'
  kfold = model_selection.KFold(n_splits=10, random_state=seed)
  cv_results = model_selection.cross_val_score(model, features_train, classes_train, cv=kfold, scoring=scoring)
  msg = "%s: %f (%f)" % (model_name, cv_results.mean(), (cv_results.std()))
  print()
  print('--== CROSS-VALIDATION RESULTS ==--')
  print('model_name, mean, stdev =', msg)
  
  # evaluate model with validation set and print results
  print()
  print('...evaluating model using validation set')
  fitted_model = createAndEvaluateModel(model, model_name, features_train, classes_train, features_validation, classes_validation)

  if model_name == 'BAYES':
    filename = 'bayes.pkl'
  else:
    filename = 'tree.pkl'

  outputFile = open(filename, 'wb')
  pickle.dump(fitted_model, outputFile)
  outputFile.close()

def createAndEvaluateModel(model, name, features_train, classes_train, features_validation, classes_validation):
  fitted_model = model.fit(features_train, classes_train)
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
  return fitted_model

def describeData(dataset):
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


#returns model object from model.pkl file using model_name argument, creates new model.pkl if it doesn't exist
def getModel(model_name):
  if model_name == 'BAYES':
    fileName = 'bayes.pkl'
  else:
    fileName = 'tree.pkl'

  if not os.path.isfile(fileName):
    print('...model not found...training a new model', fileName)
    trainModel(model_name)
  
  print('...loading model', fileName)
  inputFile = open(fileName, 'rb')
  loaded_model = pickle.load(inputFile)
  inputFile.close()
  return loaded_model
  
# json_roster = JSON request
# model_name = 'BAYES' or 'TREE' (all default to 'TREE')
# returns prediction 100 or 200 as JSON response ie - {"winner": "100"}
def getPrediction(model_name, json_roster):
  array_roster = json_roster_to_array(json_roster)

  model = getModel(model_name)
  class_prediction = model.predict([array_roster])
  
  # probability / confidence?
  # class_probabilities = svm.predict_proba(matches)

  prediction_json = {
    "winner": str(class_prediction[0])
  }
  print()
  print('[ PREDICTION ] =', prediction_json)

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
  return array_roster



# test functions for comparing different models
def runModelComparison():
  dataset = loadCsv()
  trainAndCompareModels(dataset)

def trainAndCompareModels(dataset):
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
    createAndEvaluateModel(model, name, features_train, classes_train, features_validation, classes_validation)


# json_roster_test = {
#   "b_top": "240",
#   "b_jung": "64",
#   "b_mid": "1",
#   "b_bot": "29",
#   "b_sup": "63",
#   "r_top": "17",
#   "r_jung": "24",
#   "r_mid": "238",
#   "r_bot": "51",
#   "r_sup": "432"}

# print('start code!')
# # prediction = getPrediction('BAYES', json_roster_test)
# # print('prediction ==>', prediction)
# # print('array roster =>', json_roster_to_array(json_roster_test))
# array_roster = json_roster_to_array(json_roster_test)


# loaded_model = getModel('BAYES')
# print('loaded_model =>', loaded_model)

# manual_array = ['240', '64', '1', '29', '63', '17', '24', '238', '51', '432']
# manual_array = manual_array.astype(np.float64)
# print('manual_array =>', manual_array)
# class_prediction = loaded_model.predict([manual_array])
# prediction_json = {
#   "winner": str(class_prediction[0])
# }

# print('prediction =>', prediction_json)