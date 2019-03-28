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

@app.route('/predict', methods=['POST'])
def predict():
  # request null check AND see if roster object is present in request data
  if request.method == 'POST' and request.data and 'roster' in request.get_json():
    json_roster = request.json['roster']

    # Verify incoming JSON roster has necessary values (all 5 positions for each team)
    print()
    print('[ REQUEST CONTAINS VALID ROSTER ]')
    print(json_roster)
    if validateRoster(json_roster):
      
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
        print('[ ROSTER NOT FOUND IN DB. GENERATING MODEL... ]')
        # creates model according to string arg
        model = get_model(TREE)

        # get predicted class using model and roster input from user
        json_prediction = get_prediction(model, json_roster)
        
        # insert roster/prediction as a single match into collection (db)
        insert_match(json_roster, json_prediction)
        
        # create response object and return to user (202 Accepeted)
        response = Response(response=json.dumps(json_prediction), status=202, mimetype='application/json')
        return response

  # return 400 Bad Request
  return Response(response="INVALID REQUEST!", status=400, mimetype='text/plain')

def validateRoster(roster):
  if ("b_bot" in roster and "b_jung" in roster and "b_mid" in roster and "b_sup" in roster and "b_top" in roster and"r_bot" in roster and"r_jung" in roster and "r_mid" in roster and "r_sup" in roster and "r_top" in roster):
    return True
  else:
    return False

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