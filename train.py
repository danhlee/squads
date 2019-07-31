# Load libraries
import numpy
import pandas
from pandas.plotting import scatter_matrix
import matplotlib
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from random import randint
import pickle
import sys
import os
from data import generateCsv
TREE = 'TREE'
RAND = 'RAND'

import scikitplot as skplt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score  

###################################################
#
#  getPrediction
#
###################################################
# json_roster = JSON request
# model_name = 'RAND' or 'LDA' (all default to 'RAND')
# returns prediction 100 or 200 as JSON response ie - {"winner": "100"}
def getPrediction(model_name, json_roster):
  array_roster = json_roster_to_array(json_roster)

  model = getModel(model_name)
  print('.')
  print('array_roster =', array_roster)

  # # [TODO] conversion possibly necessary for LDA and other numeric features
  # array_roster = numpy.array(array_roster, dtype=numpy.float64)
  class_prediction = model.predict([array_roster])
  
  # probability / confidence
  prediction_confidence = model.predict_proba([array_roster])
  print('raw prediction ------>', class_prediction)
  if class_prediction[0] == 100:
    confidence_index = 0
  else:
    confidence_index = 1

  print('.')
  print('prediction_confidence =', prediction_confidence)
  print('confidence_index = ', confidence_index)
  print('prediction_confidence[0][confidence_index] =', prediction_confidence[0][confidence_index])
  print('.')
  prediction_json = {
    "winner": str(class_prediction[0]),
    "confidence": prediction_confidence[0][confidence_index]
  }
  print('.')
  print('[ PREDICTION / CONFIDENCE ] =', prediction_json)
  
  return prediction_json



###################################################
#
#  getModel
#
###################################################
#returns model object from model.pkl file using model_name argument, creates new model.pkl if it doesn't exist
def getModel(model_name):
  
  ## if overwrite_model is False, model is only overwritten when /train is called explicitly
  overwrite_model = False
  print('...overwrite_model =', overwrite_model)

  if model_name == TREE:
    fileName = 'tree.pkl'
  else:
    fileName = 'rand.pkl'

  # [TODO] if overwrite_model == True, will always overwrite model.pkl file 
  # (intention is to always overwrite after gathering new data with /gather )
  if not os.path.isfile(fileName):
    print('...model not found...training new model', fileName)
    trainModel(model_name)
  elif overwrite_model is True:
    print('...overwriting model')
    trainModel(model_name)
  
  print('...loading model', fileName)
  inputFile = open(fileName, 'rb')
  loaded_model = pickle.load(inputFile)
  inputFile.close()
  return loaded_model



###################################################
#
#  trainModel
#
###################################################
# creates a model specified by model_name parameter and saves as model_name.pkl file (overwrites .pkl file)
# ENHANCEMENT: returns evaluation object {modelName, avg_precision, avg_recall, avg_accuracy, avg_f1_score, confusion_matrix(FP,FN,TP,TN) }
def trainModel(model_name):
  seed = 5
  # generate csv from all tuples in db
  generateCsv()
  
  # create dataframe and convert to 2d array
  dataset = loadCsv()
  describeData(dataset)
  dataArray = dataset.to_numpy()

  # slice array into features set and classes set
  features = dataArray[:,0:10]
  classes = dataArray[:,10]
  validation_size = 0.20

  print('seed =', seed)
  print('...splitting into training and validation sets using seed value:', seed)

  # split dataset into 80% training and 20% testing (also split by features and class for each)
  features_train, features_validation, classes_train, classes_validation = model_selection.train_test_split(features, classes, test_size=validation_size, random_state=seed)
  
  # set model statically at endpoint level
  # create model
  if (model_name == RAND):
    print()
    print('...creating Random Forest Classifier model')
    model = RandomForestClassifier(n_estimators=100)
  else:
    print()
    print('...creating TREE model')
    model = DecisionTreeClassifier()

  # [CROSS-VALIDATION] k-folds cross-validation
  print()
  print('...performing cross-validation using k-Folds with 10 splits')
  print('...using seed value:', seed)
  scoring = 'accuracy'
  kfold = model_selection.KFold(n_splits=10, random_state=seed)
  cv_results = model_selection.cross_val_score(model, features_train, classes_train, cv=kfold, scoring=scoring)
  
  print(' ')
  results_overview = "[ %f, %f ]" % (cv_results.mean(), cv_results.std())
  print(' ')
  print('--------------------------------------k-folds cross-validation---------------------------------------')
  print('[START] Cross-Validation Results for', model_name)
  print()
  print('             < 10-folds cross-validation accuracies >')
  print(cv_results)
  print()
  
  # mean accuracy and std of k-folds cross-validation
  print('[ mean, std ]')
  print(results_overview)
  print(' ')
  print('-----------------------------------------------------------------------------------------------------')

  ##[VALIDATION SET] evaluate model with validation set and print results
  fitted_model = getFittedModel(model, features_train, classes_train)
  fitted_model_evaluation = validateAndEvaluateModel(fitted_model, model_name, features_validation, classes_validation)

  
  if model_name == TREE:
    filename = 'tree.pkl'
  else:
    filename = 'rand.pkl'

  # [CREATE MODEL FILE] create .pkl file to store model
  print('...creating pickle file: ' + filename)
  # wb will overwrite binary files
  outputFile = open(filename, 'wb')
  pickle.dump(fitted_model, outputFile)
  outputFile.close()

  return fitted_model_evaluation


###################################################
#
#  getFittedModel
#
###################################################
def getFittedModel(model, features_train, classes_train):
  return model.fit(features_train, classes_train)

###################################################
#
#  validateAndEvaluateModel
#
###################################################
def validateAndEvaluateModel(fitted_model, model_name, features_validation, classes_validation):
  print('----------------------------------------validate model---------------------------------------------')
  print(' ')
  print('[START] Testing model with validation set for', model_name)
  print(' ')
  
  class_predictions = fitted_model.predict(features_validation)

  # print predictions for validation set
  print('                < class predictions for validation set >\n', class_predictions)
  print(' ')
  print(' ')

  # print confusion matrix
  print('--== CONFUSION MATRIX ==--')
  confusion_matrix_temp = confusion_matrix(classes_validation, class_predictions)
  confusion_matrix_array = [str(confusion_matrix_temp[0][0]), str(confusion_matrix_temp[0][1]), str(confusion_matrix_temp[1][0]), str(confusion_matrix_temp[1][1]) ]
  print('just confusion matrix:', confusion_matrix_temp)
  print(' ')
  print(pandas.DataFrame(confusion_matrix_temp))
  print(' ')
  print('             --== CLASSIFICATION REPORT ==--')
  vs_classificaiton_report = classification_report(classes_validation, class_predictions)
  print(vs_classificaiton_report)
  print(' ')

  # print avg accuracy
  print(' ')
  avg_accuracy = accuracy_score(classes_validation, class_predictions)
  accuracy_msg = '...avg prediction accuracy of ' + model_name + ' ='
  print(accuracy_msg, avg_accuracy)
  print(' ')
  print('-----------------------------------------------------------------------------------------------------')
  
  model_evaluation = {
    'modelName': model_name,
    'confusionMatrix': confusion_matrix_array,
    'avgAccuracy': avg_accuracy
  }

  return model_evaluation






###################################################
#
#  [HELPER] loadCsv
#
###################################################
# loads matches.csv
# returns DataFrame
def loadCsv():
  csv = "matches.csv"
  col_names = ['b_top','b_jung','b_mid','b_bot','b_sup','r_top','r_jung','r_mid','r_bot','r_sup','winner']
  dataset = pandas.read_csv(csv, names=col_names)
  return dataset


###################################################
#
#  [HELPER] json_roster_to_array
#
###################################################
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

###################################################
#
#  [LOG] describeData
#
###################################################
def describeData(dataset):
  # shape shows (numberOfTuples, numberOfFeatures)
  print('----------------------------------------describe dataset---------------------------------------------')
  print('.')
  print('--== (#TUPLES, #FEATURES) ==--')
  print(dataset.shape)
  # shows first 20 tuples
  print('.')
  print('--== FIRST 20 TUPLES ==--')
  print(dataset.head(20))
  print('.')
  print('--== DESCRIPTION of DATASET ==--')
  print(dataset.describe())
  print('.')
  dataset.info()
  print('-----------------------------------------------------------------------------------------------------')



  
###################################################
#
#  TEST functions for comparing different models
#
###################################################
def runModelComparison():
  generateCsv()
  dataset = loadCsv()
  describeData(dataset)
  trainAndCompareModels(dataset)

def runSingleModelTest():
  # generateCsv()
  dataset = loadCsv()
  
  ################ for model eval and testing ###################
  
  ## Histogram (don't need anymore...)
  # dataset.hist()

  ## Multivariate Scatter Plot
  # scatter_matrix(dataset)
  # plt.show()


  ## ARRAY VERSION
  # dataArray = dataset.to_numpy()
  # Y = dataArray[:,10]                                   # split classifications col
  # X = dataArray[:,0:10]                           # split features col


  ## DATAFRAME VERSION (CORRECT ONE!)
  # Y = dataset['winner']                                   # Split off classifications
  # X = dataset.loc[:,:'r_sup']                             # Split off features (all rows, for cols from START through r_sup - excludes winner column)

  # X_norm = (X - X.min())/(X.max() - X.min())
  # pca = PCA(n_components=2)                                 #2-dimensional PCA
  # transformed = pandas.DataFrame(pca.fit_transform(X_norm))

  # plt.scatter(transformed[Y==100][0], transformed[Y==100][1], label=100, c='blue')
  # plt.scatter(transformed[Y==200][0], transformed[Y==200][1], label=200, c='red')

  # plt.legend()
  # plt.show()
  

  dataArray = dataset.to_numpy()

  ## slice array into features set and classes set
  seed = 0
  features = dataArray[:,0:10]
  classes = dataArray[:,10]
  ## validation set is 20% training is 80%
  validation_size = 0.20

  print('seed =', seed)
  print('...splitting into training and validation sets using seed value:', seed)

  ## split dataset into 80% training and 20% testing (also split by features and class for each)
  features_train, features_validation, classes_train, classes_validation = model_selection.train_test_split(features, classes, test_size=validation_size, random_state=seed)
  
  ## select model to test
  model = RandomForestClassifier(n_estimators=100)
  # model = DecisionTreeClassifier()

  
  ## FIT model and get predict_proba array
  fitted_model = getFittedModel(model, features_train, classes_train)
  predicted_probabilities = fitted_model.predict_proba(features_validation)

  
  
  ########################################################################################################################################### TEST
  ########################################################################################################################################### TEST
  ########################################################################################################################################### TEST
  # multiClassPlotRocCurve(model, classes, features_train, classes_train, features_validation, classes_validation)
  # print('features_validation => ...',  features_validation)
  # print('classes_validation => ...',  classes_validation)
  # print('predicted_probabilities ==> ...  ', predicted_probabilities)
  


  ## SPLIT predict proba 2D array to get just 1 column
  predicted_probabilities = predicted_probabilities[:, 1]

  ## AUC calculation
  calculateAuc(classes_validation, predicted_probabilities)
  plotRocCurve(classes_validation, predicted_probabilities)

  ## Precision, Accuracy, & Recall calculation
  validateAndEvaluateModel(fitted_model, 'RANDOM_FOREST', features_validation, classes_validation)



from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
def multiClassPlotRocCurve(model, classes, features_train, classes_train, features_validation, classes_validation):
  y = label_binarize(classes, classes=[100, 200])
  
  n_classes = y.shape[1]
  classifier = OneVsRestClassifier(model)
  y_score = classifier.fit(features_train, classes_train).predict_proba(features_validation)
  print('y (before) =>', classes )
  print('y (after) =>', y )
  print('n_classes =>', n_classes)
  print('y_score =>', y_score)
  print('classes_validation =>', classes_validation)

  ## Compute ROC curve and ROC area for each class
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  for i in range(n_classes):
      fpr[i], tpr[i], _ = roc_curve(classes_validation[:, i], y_score[:, i])
      roc_auc[i] = auc(fpr[i], tpr[i])
  
  # fpr["micro"], tpr["micro"], _ = roc_curve(classes_validation.ravel(), y_score.ravel())
  # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

  # plt.figure()
  # lw = 2
  # plt.plot(fpr[2], tpr[2], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
  # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  # plt.xlim([0.0, 1.0])
  # plt.ylim([0.0, 1.05])
  # plt.xlabel('False Positive Rate')
  # plt.ylabel('True Positive Rate')
  # plt.title('Receiver operating characteristic example')
  # plt.legend(loc="lower right")
  # plt.show()

def calculateAuc(classes_validation, predicted_probabilities):
  auc = roc_auc_score(classes_validation, predicted_probabilities)
  print('AUC: %.2f' % auc)

def plotRocCurve(classes_validation, predicted_probabilities):
  ## ROC curve: ver_1
  # skplt.metrics.plot_roc(classes_validation, predicted_probabilities)
  # plt.show()

  ## ROC curve: ver_2
  fpr, tpr, thresholds = roc_curve(classes_validation, predicted_probabilities, pos_label=200)
  print('threshold:', thresholds)
  drawCurve(fpr, tpr)
  
  ## ROC curve: ver_3
  # y_score = fitted_model.decision_function(features_validation)
  # ## Compute ROC curve and ROC area for each class
  # fpr = dict()
  # tpr = dict()
  # roc_auc = dict()
  # for i in range(n_classes):
  #     fpr[i], tpr[i], _ = roc_curve(classes_validation[:, i], y_score[:, i])
  #     roc_auc[i] = auc(fpr[i], tpr[i])

def drawCurve(fpr, tpr):  
  plt.plot(fpr, tpr, color='orange', label='ROC')
  plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic (ROC) Curve')
  plt.legend()
  plt.show()



def trainAndCompareModels(dataset):
  dataArray = dataset.to_numpy()
  # arraySlice[start_row:end_row_exclusive, start_col:end_col_exclusive]
  # features = tuples minus class
  # classes = just the classes of each tuple
  features = dataArray[:,0:10]
  classes = dataArray[:,10]
  # 20 percent of tuples reserved for validation
  validation_size = 0.20
  seed = 0
  print('...seed value:', seed)
  print()
  features_train, features_validation, classes_train, classes_validation = model_selection.train_test_split(features, classes, test_size=validation_size, random_state=seed)

  models = []
  # Logistic Regression Classifier
  # models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
  # Linear Discriminant Analysis
  #models.append(('LDA', LinearDiscriminantAnalysis()))
  # K-nearest neighbors Classifier
  #models.append(('KNN', KNeighborsClassifier()))
  # Decision tree Classifier
  # models.append(('TREE', DecisionTreeClassifier()))
  # Gaussian Naive Bayes
  #models.append(('BAYES', GaussianNB()))
  # Support Vector Classification
  models.append(('SVM', SVC(kernel='linear', gamma='auto')))
  # Random Forest Classifier
  #models.append(('RAND', RandomForestClassifier()))

  print('----------------------------------- MODEL COMPARISONS -----------------------------------')
  print()

  scoring = 'accuracy'
  for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, features_train, classes_train, cv=kfold, scoring=scoring)

    results_overview = "[ %f, %f ]" % (cv_results.mean(), cv_results.std())
    print()
    print('[START] Cross-Validation Results for', name)
    print()
    print('             < 10-folds cross-validation accuracies >')
    print(cv_results)
    print()
    
    print('[ mean, std ]')
    print(results_overview)
    validateAndEvaluateAllModels(model, name, features_train, classes_train, features_validation, classes_validation)

def validateAndEvaluateAllModels(model, model_name, features_train, classes_train, features_validation, classes_validation):
  print('----------------------------------------validate all models---------------------------------------------')
  print(' ')
  print('[START] Testing model with validation set for', model_name)
  print(' ')
  fitted_model = model.fit(features_train, classes_train)
  class_predictions = fitted_model.predict(features_validation)

  # print predictions for validation set
  print('                < class predictions for validation set >\n', class_predictions)
  print(' ')
  print(' ')

  # print confusion matrix
  print('--== CONFUSION MATRIX ==--')
  confusion_matrix_temp = confusion_matrix(classes_validation, class_predictions)
  #confusion_matrix_array = [str(confusion_matrix_temp[0][0]), str(confusion_matrix_temp[0][1]), str(confusion_matrix_temp[1][0]), str(confusion_matrix_temp[1][1]) ]
  print('just confusion matrix:', confusion_matrix_temp)
  print('.')
  print(pandas.DataFrame(confusion_matrix_temp))
  print('.')
  print('             --== CLASSIFICATION REPORT ==--')
  vs_classificaiton_report = classification_report(classes_validation, class_predictions)
  print('.')
  print(vs_classificaiton_report)
  # print avg accuracy
  print('.')
  avg_accuracy = accuracy_score(classes_validation, class_predictions)
  accuracy_msg = '...avg prediction accuracy of ' + model_name + ' ='
  print(accuracy_msg, avg_accuracy)
  print('.')
  print('-----------------------------------------------------------------------------------------------------')


###################################################
#
#  direct comparison test (uncomment and run file!)
#
#  run heroku locally and train 
###################################################
# print('...running model and comparison')
# runModelComparison()

# print('...testing single model')
# runSingleModelTest()