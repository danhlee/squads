import csv, json, sys
import os.path
from pymongo import MongoClient

client = MongoClient('mongodb://purple:pickle1@ds215019.mlab.com:15019/heroku_mxpzq74x/squads')
db = client.squads


# takes relative directory and converts all JSON matches within into csv
def insertMatches(directory):
  for jsonFile in os.listdir(directory):
    inputFile = open( directory + jsonFile )
    matchesObject = json.load(inputFile)
    matchesArray = matchesObject['matches']

    # iterate through each match and extract roster and winner
    for match in matchesArray:
      participants = match['participants']
      teams = match['teams']
      winner = ''
      if teams[0]['win'] == 'Win':
        winner = '100'
      else:
        winner = '200'

      matchTuple = [None]*11
      for participant in participants:
        
        lane = participant['timeline']['lane']
        role = participant['timeline']['role']
        team = participant['teamId']
        
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
        if (role == "DUO_CARRY" or role == "DUO") and (matchTuple[8] == None) and team == 200:
          matchTuple[8] = participant['championId']
        if (role == "DUO_SUPPORT" or role == "DUO") and team == 200:
          matchTuple[9] = participant['championId']
        
      matchTuple[10] = winner
      
      if None not in matchTuple:
        insertTupleIntoMatches(matchTuple)

    inputFile.close()

def getMatchDataDirectory(dataSource):
  print('dataSource =', dataSource)
  if dataSource != 'seed':
    print('-----------> USING data dir')
    return './data/'
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
    'winner': matchTuple[10]
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








# print(sys.argv)
# if len(sys.argv) == 2:
#   print('arg[1]', sys.argv[1])
#   dataDirectory = getMatchDataDirectory(sys.argv[1])
# else:
#   dataDirectory = getMatchDataDirectory('data')