from flask import Flask, jsonify, request, Response, json
from flask_pymongo import PyMongo
from flask_cors import CORS

from data import getMatchDataDirectory, insertMatches
from train import trainModel, getPrediction, getModel, TREE, RAND
from request_validation import valid_positions, valid_championIds



app = Flask(__name__)
CORS(app, supports_credentials=True)

#################### DEV ##########################
#
# app.config["MONGO_DBNAME"] = 'squads'
# app.config['MONGO_URI'] = 'mongodb://localhost:27017/squads'
#
#################### PROD #########################
#
app.config['MONGO_URI'] = 'mongodb://purple:pickle1@ds215019.mlab.com:15019/heroku_mxpzq74x'
#
###################################################

mongo = PyMongo(app)
matches = mongo.db.matches


###################################################
#
#  /
#
###################################################
# preprocesses all seed matches from JSON format and inserts into squads.matches
@app.route('/')
def index():
  msg = '[SqualorArchive] Welcome to Squads API'
  response = Response(response=msg, status=200, mimetype='text/plain')
  return response


###################################################
#
#  /seed
#
###################################################
# preprocesses all seed matches from JSON format and inserts into squads.matches
@app.route('/seed')
def seed():
  
  # prevent using seed matches twice
  count = matches.count_documents({})
  print('count =', count)
  msg = 'Database already contains seed matches.'
  if ( count == 0 ):
    insertMatches(getMatchDataDirectory('seed'))
    msg = 'Inserted seed matches into empty database.'

  newCount = matches.count_documents({})
  msg = msg + ' There are currently ' + newCount + 'matches in the database...'
  
  response = Response(response=msg, status=200, mimetype='text/plain')
  return response


###################################################
#
#  /gather
#
###################################################
# fetches pro_50 matches as JSON array and "saves in /gathered_data" OR "reads data, appends to matches.csv, then tosses"
# generates matches.csv using NEW json data
@app.route('/gather')
def gather():
  #fetchMatches()
  #saveAsJsonArray in /data
  insertMatches(getMatchDataDirectory('data'))
  count = matches.count_documents({})

  msg = 'Gathered new matches and inserted into database. Total document count is now: ' + str(count) + '...'

  response = Response(response=msg, status=200, mimetype='text/plain')
  return response


###################################################
#
#  /train
#  (model_name = TREE or RAND)
#
###################################################
@app.route('/train')
def train():
  # get model name from string param
  modelNameParam = request.args['modelName']

  # Enforce a default model as validation (TREE is default model)
  if modelNameParam == RAND:
    model_evaluation = trainModel(RAND)
    model_name = 'Random Forest Classifier'
  else:
    model_evaluation = trainModel(TREE)
    model_name = 'Decision Tree Classifier'
  
    
  msg = 'A new ' + model_name + ' model has been created...'

  evaluation_results = {
    'modelEvaluation': model_evaluation,
    'msg': msg
  }

  response = Response(response=json.dumps(evaluation_results), status=200, mimetype='application/json')
  return response



###################################################
#
#  /predict 
#  (model_name = TREE or RAND)
#
###################################################
@app.route('/predict', methods=['POST'])
def predict():
  
  # Enforce a default model as validation (TREE is default model)
  modelNameParam = request.args['modelName']
  if modelNameParam == RAND:
    model_name = RAND
  else:
    model_name = TREE
  

  # request null check AND see if roster object is present in request data
  if request.data and 'roster' in request.get_json():
    json_roster = request.json['roster']

    # validate roster positions (1 of each for each team)
    if valid_positions(json_roster):

      # validate championIds
      if valid_championIds(json_roster):

        # get predicted class using model and roster input from user
        json_prediction = getPrediction(model_name, json_roster)
        
        # create response object and return to user (202 Accepeted)
        response = Response(response=json.dumps(json_prediction), status=202, mimetype='application/json')
        return response

  # return 400 Bad Request
  return Response(response="Invalid Request! Make sure roster and role values are valid.", status=400, mimetype='text/plain')


if __name__ == '__main__':
  # app.run()
  app.run(debug=True)