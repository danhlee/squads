from flask import Flask, jsonify, request, Response, json
from train import getModel, getPrediction, TREE, BAYES
from flask_pymongo import PyMongo
from data import getMatchDataDirectory, generateCsv
from request_validation import valid_positions, valid_championIds
app = Flask(__name__)

app.config["MONGO_DBNAME"] = 'squads'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/squads'
mongo = PyMongo(app)

# generates matches.csv using seed json data
@app.route('/seed')
def seed():
  generateCsv(getMatchDataDirectory('seed'))
  response = Response(response='generating csv using seed data...', status=200, mimetype='text/plain')
  return response

# fetches pro_50 matches as JSON array and "saves in /data" OR "reads data, appends to matches.csv, then tosses"
# generates matches.csv using NEW json data
@app.route('/gather')
def gather():
  #fetchMatches()
  #saveAsJsonArray in /data
  generateCsv(getMatchDataDirectory('data'))
  response = Response(response='generating csv using new data...', status=200, mimetype='text/plain')
  return response

@app.route('/predict', methods=['POST'])
def predict():
  modelName = request.args['modelName']
  # request null check AND see if roster object is present in request data
  if request.method == 'POST' and request.data and 'roster' in request.get_json():
    json_roster = request.json['roster']

    # validate roster positions (1 of each for each team)
    if valid_positions(json_roster):

      # validate championIds
      if valid_championIds(json_roster):
        # check db for exact roster and return prediction from tuple if it exists
        # if it doesn't exist in db, create model for prediction and store results in db
        found_match = find_match(json_roster)
        if found_match is not None:
          print()
          print('[ EXACT ROSTER FOUND IN DB ]')
          response = Response(response=json.dumps(found_match['prediction']), status=202, mimetype='application/json')
          return response
        else:
          print()
          print('[ ROSTER NOT FOUND IN DB. GETTING SAVED MODEL (.pkl)... ]')

          # if modelName.pkl exists
          # filter based on query param containing ML model name
          if modelName == 'bayes':
            # check if bayes.pkl already exists
            # creates model according to query param
            model = getModel(BAYES)
          else:
            # check if tree.pkl already exists
            # creates model according to query param
            model = getModel(TREE)

          # get predicted class using model and roster input from user
          json_prediction = getPrediction(model, json_roster)
          
          # insert roster/prediction as a single match into collection (db)
          insert_match(json_roster, json_prediction)
          
          # create response object and return to user (202 Accepeted)
          response = Response(response=json.dumps(json_prediction), status=202, mimetype='application/json')
          return response

  # return 400 Bad Request
  return Response(response="INVALID REQUEST!", status=400, mimetype='text/plain')

# DB Access Functions
def find_match(json_roster):
  match = mongo.db.matches
  found_match = match.find_one({"roster": json_roster})
  return found_match

def insert_match(json_roster, json_prediction):
  match = mongo.db.matches
  single_match = {
    "roster": json_roster,
    "prediction": json_prediction
  }
  print()
  print('[ INSERTING SINGLE MATCH INTO DB ]')
  print(single_match)
  match.insert_one(single_match)

if __name__ == '__main__':
  app.run(debug=True)