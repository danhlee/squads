from flask import Flask, jsonify, request, Response, json
from classifier import get_model, get_prediction, TREE, BAYES
from flask_pymongo import PyMongo

app = Flask(__name__)

app.config["MONGO_DBNAME"] = 'matches'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/matches'
mongo = PyMongo(app)

@app.route('/')
def index():
  return 'Welcome to SQUADS!'

@app.route('/insert')
def insert():
  match = mongo.db.matches
  match.insert({
    "roster": {
        "b_top": "240",
        "b_jung": "64",
        "b_mid": "1",
        "b_bot": "29",
        "b_sup": "63",
        "r_top": "17",
        "r_jung": "24",
        "r_mid": "238",
        "r_bot": "51",
        "r_sup": "432"
    },
    "winner": "100"
  })
  return 'MATCH ADDED!'

@app.route('/predict', methods=['POST'])
def predict():
  json_roster = request.json['roster']
  
  # Verify incoming JSON roster has necessary values (each position)
  print('roster =', json_roster)
  if validateRoster(json_roster):
    # creates model according to string arg
    model = get_model(TREE)
    # get predicted class using model and roster input from user
    json_prediction = get_prediction(model, json_roster)

    # create response object and return to user
    response = Response(response=json.dumps(json_prediction), status=202, mimetype='application/json')
    return response
  else:
    return Response(response="Invalid Request!", status=400, mimetype='text/plain')

def validateRoster(roster):
  if ("b_bot" in roster and "b_jung" in roster and "b_mid" in roster and "b_sup" in roster and "b_top" in roster and"r_bot" in roster and"r_jung" in roster and "r_mid" in roster and "r_sup" in roster and "r_top" in roster):
    return True
  else:
    return False

if __name__ == '__main__':
  app.run(debug=True)