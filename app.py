from flask import Flask, jsonify, request, Response, json
from flask_pymongo import PyMongo

from data import getMatchDataDirectory, insertMatches
from train import trainModel, getPrediction, getModel, TREE, RAND
from request_validation import valid_positions, valid_championIds



app = Flask(__name__)

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
  msg = 'database already contains seed matches...'
  if ( count == 0 ):
    insertMatches(getMatchDataDirectory('seed'))
    msg = 'inserted seed matches into database...'

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
  response = Response(response='gathered new matches and inserted into database...', status=200, mimetype='text/plain')
  return response


###################################################
#
#  /train
#  (model_name = TREE or RAND)
#
###################################################
@app.route('/train', methods=['POST'])
def train():
  # TODO: refactor to allow prediction with diff models via param input
  modelName = request.args['modelName']
  if modelName != TREE:
    modelName = RAND
  model_name = modelName

  trainModel(model_name)

  msg = 'model trained using ' + model_name
  response = Response(response=msg, status=200, mimetype='text/plain')
  return response



###################################################
#
#  /predict 
#  (model_name = TREE or RAND)
#
###################################################
@app.route('/predict', methods=['POST'])
def predict():
  # TODO: refactor to allow prediction with diff models via param input
  modelName = request.args['modelName']
  if modelName != TREE:
    modelName = RAND
  model_name = modelName

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
  return Response(response="INVALID REQUEST!", status=400, mimetype='text/plain')


if __name__ == '__main__':
  # app.run()
  app.run(debug=True)