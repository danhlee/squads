from flask import Flask, jsonify, request, Response, json
from flask_pymongo import PyMongo
from flask_cors import CORS
import requests

from data import getMatchDataDirectory, insertMatches, insertSingleMatch, isNewMatch, pro_usernames
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
  msg = msg + ' There are currently ' + str(newCount) + ' matches in the database...'
  
  print('msg = ', msg)

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
  ## check for empty api key
  ## if empty returns 401 unauthorized
  if request.args.get('api_key') is None or request.args['api_key'] == '':
    return Response(response='No API key provided!', status=401, mimetype='text/plain')
  if request.args.get('summoner') is None or request.args['summoner'] == '':
    return Response(response='No Summoner provided!', status=401, mimetype='text/plain')


  print('request.args[api_key] = ', request.args['api_key'])
  print('request.args[summoner] = ', request.args['summoner'])

  ## TODO: Toggle summoner_list from UI or pro_usernames list
  summoner_list = [str(request.args['summoner'])]
  # summoner_list = pro_usernames

  ## fetch matches from riot api
  apiKeyParam = '?api_key=' + request.args['api_key']
  countBefore = matches.count_documents({})

  ## iterate through all PROs in pro_usernames array
  for pro_username in summoner_list:
    ## get summonerId using user name
    url = 'https://na1.api.riotgames.com/lol/summoner/v4/summoners/by-name/' + pro_username + apiKeyParam 
    print('url =', url)
    summoner_json = requests.get( url ).json()
    print('summoner_json (response) =', summoner_json)


    ## Check for invalid api key. 
    ## status_code: 403 will be returned with message: forbidden
    if 'status' in summoner_json and summoner_json['status']['status_code'] == 403:
      return Response(response='Invalid API key!', status=403, mimetype='text/plain')
    if 'status' in summoner_json and summoner_json['status']['status_code'] == 404:
      msg_404 = str('Summoner', pro_username, 'not found!')
      return Response(response=msg_404, status=404, mimetype='text/plain')
    if 'status' in summoner_json and summoner_json['status']['status_code'] == 429:
      return Response(response='Riot API rate limit exceeded!', status=429, mimetype='text/plain')


    print('accountId in summoner_json ?', 'accountId' in summoner_json)
    if 'accountId' not in summoner_json:
      print('summoner_json (accountId check)= ', summoner_json)
      return Response(response='Riot API rate limit exceeded!', status=429, mimetype='text/plain')

    encrypted_summoner_id = summoner_json['accountId']
    print('encrypted_summoner_id =', encrypted_summoner_id)

    ## get 100 most recent matches by encrypted summonerId
    url = 'https://na1.api.riotgames.com/lol/match/v4/matchlists/by-account/' + encrypted_summoner_id + apiKeyParam
    matches_json = requests.get( url ).json()


    ## check if summoner match request gave a valid object, IF NOT skip
    if 'matches' not in matches_json:
      continue
    matches_array = matches_json['matches']
    print('.')
    print('[FETCH] Retrieved', len(matches_array), 'matches for:', pro_username)
    print('.')


    ## IF a summoner has MORE than 100 games played, iterate through first 100 (DRAFT/SOLOQ)
    ## LESS than 100 recent games played, iterate through ALL (DRAFT/SOLOQ)
    ## ELSE (if EQUALS 0) do nothing
    MATCH_COUNT_CAP = 70

    if len(matches_array) > MATCH_COUNT_CAP:
      i = 0
      # will make a fetch call for 20 latest games
      while i < MATCH_COUNT_CAP:
        # 400 = 5v5 Draft Pick
        # 420 = 5v5 Solo Queue Ranked
        if (matches_array[i]['queue'] == 400 or matches_array[i]['queue'] == 420):
          url = 'https://na1.api.riotgames.com/lol/match/v4/matches/' + str(matches_array[i]['gameId']) + apiKeyParam
          single_match_data = requests.get( url ).json()
          insertSingleMatch(single_match_data)
        i += 1
    elif len(matches_array) <= MATCH_COUNT_CAP and len(matches_array) > 0:
      for match in matches_array:
        if (match['queue'] == 400 or match['queue'] == 420):
          url = 'https://na1.api.riotgames.com/lol/match/v4/matches/' + str(match['gameId']) + apiKeyParam
          single_match_data = requests.get( url ).json()
          insertSingleMatch(single_match_data)
    
    print('[LOG] Finished processing matches for...', pro_username)

  countAfter = matches.count_documents({})
  numNewTuples = countAfter-countBefore
  msg = 'Gathered and inserted ' + str(numNewTuples) + ' new matches. Total document count is now: ' + str(countAfter) + '...'
  
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