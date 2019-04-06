from flask import Flask, jsonify, request, Response, json
from train import trainModel, getPrediction, LDA, RAND
from flask_pymongo import PyMongo
from data import getMatchDataDirectory, insertMatches
from request_validation import valid_positions, valid_championIds
from train import trainModel, getModel

app = Flask(__name__)
app.config["MONGO_DBNAME"] = 'squads'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/squads'
mongo = PyMongo(app)
matches = mongo.db.matches

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
# fetches pro_50 matches as JSON array and "saves in /data" OR "reads data, appends to matches.csv, then tosses"
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
#  /train (set model_name statically)
#
###################################################
@app.route('/train', methods=['POST'])
def train():
  # TODO: refactor to allow prediction with diff models via param input
  # modelName = request.args['modelName']
  model_name = RAND

  trainModel(model_name)

  msg = 'model trained using ' + model_name
  response = Response(response=msg, status=200, mimetype='text/plain')
  return response



###################################################
#
#  /predict (set model_name statically)
#
###################################################
@app.route('/predict', methods=['POST'])
def predict():
  # TODO: refactor to allow prediction with diff models via param input
  # modelName = request.args['modelName']
  model_name = RAND

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
  app.run(debug=True)