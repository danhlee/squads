from flask import Flask, jsonify, request, Response, json
from train import trainModel, getPrediction, TREE, BAYES
from flask_pymongo import PyMongo
from data import getMatchDataDirectory, insertMatches
from request_validation import valid_positions, valid_championIds
from train import trainModel, getModel

app = Flask(__name__)
app.config["MONGO_DBNAME"] = 'squads'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/squads'
mongo = PyMongo(app)

# generates matches.csv using seed json data
@app.route('/seed')
def seed():
  insertMatches(getMatchDataDirectory('seed'))
  response = Response(response='inserting seed matches into database...', status=200, mimetype='text/plain')
  return response

# fetches pro_50 matches as JSON array and "saves in /data" OR "reads data, appends to matches.csv, then tosses"
# generates matches.csv using NEW json data
@app.route('/gather')
def gather():
  #fetchMatches()
  #saveAsJsonArray in /data
  insertMatches(getMatchDataDirectory('data'))
  response = Response(response='inserting new matches into database...', status=200, mimetype='text/plain')
  return response

@app.route('/train', methods=['POST'])
def train():
  modelName = request.args['modelName']
  print('modelName =======>', modelName)
  if modelName == 'bayes':
    trainModel('BAYES')
  else:
    trainModel('TREE')

  response = Response(response='training model...', status=200, mimetype='text/plain')
  return response

@app.route('/predict', methods=['POST'])
def predict():
  modelName = request.args['modelName']
  # request null check AND see if roster object is present in request data
  if request.data and 'roster' in request.get_json():
    json_roster = request.json['roster']

    # validate roster positions (1 of each for each team)
    if valid_positions(json_roster):

      # validate championIds
      if valid_championIds(json_roster):
        print('modelName =', modelName)
        print(modelName == 'bayes')
        if modelName == 'bayes':
          model_name = 'BAYES'
        else:
          model_name = 'TREE'

        # get predicted class using model and roster input from user
        json_prediction = getPrediction(model_name, json_roster)
        
        # create response object and return to user (202 Accepeted)
        response = Response(response=json.dumps(json_prediction), status=202, mimetype='application/json')
        return response

  # return 400 Bad Request
  return Response(response="INVALID REQUEST!", status=400, mimetype='text/plain')


if __name__ == '__main__':
  app.run(debug=True)