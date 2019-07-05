import csv, json, sys
import os.path
from pymongo import MongoClient


#################### DEV ##########################
#
# connection = MongoClient('mongodb://localhost:27017/squads')
# db = connection['squads']
#
#################### PROD #########################
#
connection = MongoClient('mongodb://purple:pickle1@ds215019.mlab.com:15019/heroku_mxpzq74x')
db = connection['heroku_mxpzq74x']
#
###################################################



###################################################
#
#  insertMatches
#
###################################################
# 1. takes relative directory and loops through all JSON match files for cleaning/preprocessing
# 2. iterates through each of the 100 matches in each JSON file
# 3. calls insertSingleMatch
###################################################
def insertMatches(directory):
  dirtyCount = 0

  for jsonFile in os.listdir(directory):
    inputFile = open( directory + jsonFile, encoding = "ISO-8859-1" )
    matchesObject = json.load(inputFile)
    matchesArray = matchesObject['matches']

    # iterate through each match and extract roster and winner
    for match in matchesArray:
      isDirty = insertSingleMatch(match)

      # counts the number of omitted matches
      if isDirty is True:
        dirtyCount = dirtyCount + 1
    inputFile.close()

  print(dirtyCount, 'seed matches were omitted...')


###################################################
#
#  insertSingleMatch
#
###################################################
# 1. helper method that inserts each match object into the MongoDB collection
# 2. cleans data by omitting matches with duplicate roles & ambiguous DUO role
###################################################
def insertSingleMatch(match):
  cursor = db.matches.find({ 'gameId': str(match['gameId']) })

  ## if match with identical gameId doesn't already exist in db...
  if cursor.count() == 0: 
    participants = match['participants']
    teams = match['teams']
    winner = ''
    if teams[0]['win'] == 'Win':
      winner = '100'
    else:
      winner = '200'

    matchTuple = [None]*12
    for participant in participants:

      lane = participant['timeline']['lane']
      role = participant['timeline']['role']
      team = participant['teamId']
      
      if role == "DUO":
        # break
        return True

      if lane == "TOP" and team == 100:
        matchTuple[0] = participant['championId']
      if lane == "JUNGLE" and team == 100:
        matchTuple[1] = participant['championId']
      if lane == "MIDDLE" and team == 100:
        matchTuple[2] = participant['championId']
      if role == "DUO_CARRY" and team == 100:
        matchTuple[3] = participant['championId']
      if role == "DUO_SUPPORT" and team == 100:
        matchTuple[4] = participant['championId']
      if lane == "TOP" and team == 200:
        matchTuple[5] = participant['championId']
      if lane == "JUNGLE" and team == 200:
        matchTuple[6] = participant['championId']
      if lane == "MIDDLE" and team == 200:
        matchTuple[7] = participant['championId']
      if role == "DUO_CARRY" and team == 200:
        matchTuple[8] = participant['championId']
      if role == "DUO_SUPPORT" and team == 200:
        matchTuple[9] = participant['championId']
      
    matchTuple[10] = winner
    matchTuple[11] = str(match['gameId'])

    ## TODO test to count how many dirtyTuples exist in seed dataset
    # duplicate roles will cause 1 array position to contain None so it will not be inserted
    if None not in matchTuple:
      insertTupleIntoMatches(matchTuple)
      return False
    else:
      return True

def getMatchDataDirectory(dataSource):
  print('dataSource =', dataSource)
  if dataSource != 'seed':
    print('-----------> USING data dir')
    return './gathered_data/'
  else:
    print('-----------> USING seed dir')
    return './seed/'

def insertTupleIntoMatches(matchTuple):
  single_match = {
    'b_top': matchTuple[0],
    'b_jung': matchTuple[1],
    'b_mid': matchTuple[2],
    'b_bot': matchTuple[3],
    'b_sup': matchTuple[4],
    'r_top': matchTuple[5],
    'r_jung': matchTuple[6],
    'r_mid': matchTuple[7],
    'r_bot': matchTuple[8],
    'r_sup': matchTuple[9],
    'winner': matchTuple[10],
    'gameId': matchTuple[11]
  }
  db.matches.insert_one(single_match)

# Overwrites matches.csv with all tuples in squads.matches
def generateCsv():
  outputFile = open('matches.csv','w', newline='')
  csvWriter = csv.writer(outputFile)
  for match in db.matches.find():
    matchTuple = []
    matchTuple.append(match['b_top'])
    matchTuple.append(match['b_jung'])
    matchTuple.append(match['b_mid'])
    matchTuple.append(match['b_bot'])
    matchTuple.append(match['b_sup'])
    matchTuple.append(match['r_top'])
    matchTuple.append(match['r_jung'])
    matchTuple.append(match['r_mid'])
    matchTuple.append(match['r_bot'])
    matchTuple.append(match['r_sup'])
    matchTuple.append(match['winner'])
    csvWriter.writerow(matchTuple)

  outputFile.close()


################################################
#
# list of pro player usernames
#
################################################
## - 100 matches will be gathered for each player
## - mind rate limits
################################################
pro_usernames = [
  'Doublelift'
]


################################################
#
# for running data.py in command prompt arguments
#
################################################

# print(sys.argv)
# if len(sys.argv) == 2:
#   print('arg[1]', sys.argv[1])
#   dataDirectory = getMatchDataDirectory(sys.argv[1])
# else:
#   dataDirectory = getMatchDataDirectory('data')