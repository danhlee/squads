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
# Returns: True = match is omitted
#          False = match is NOT ommitted
###################################################
def insertSingleMatch(match):

  if 'gameId' not in match:
    print('[ERROR] Error while inserting single match: match does not contain gameId property...')
    print('match variable contains: ', match)
    return True
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

    if ('gameId' in match):
      matchTuple[11] = str(match['gameId'])
    else:
      print('[ERROR] gameId property not found in match object. Match object printed below...')
      print(match)
      return False

    ## TODO test to count how many dirtyTuples exist in seed dataset
    # duplicate roles will cause 1 array position to contain None so it will not be inserted
    if None not in matchTuple:
      insertTupleIntoMatches(matchTuple)
      return False
    else:
      return True

def isNewMatch(match):
  if 'gameId' in match:
    cursor = db.matches.find({ 'gameId': str(match['gameId']) })
    ## if match with identical gameId doesn't already exist in db...
    return cursor.count() == 0
  return False

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
# TOP: 'Licorice', 'Liquid Impact','FLY V1per', 'TSM BB', 'Htzr', 'Ssumdayday', 'fox solo', 'StrongHuni', 'Dhokla', 'Darshan',
# JNG: 'dominans', 'Xmithie', 'Lizardsking', 'Chimpion', 'TSM Akaadian', 'dominans', 'Meteos', 'Wiggily', 'AnDa', 'clutch lira',
# MID: 'TL Jensen', 'TSM Bjergsen', 'Pobelter', 'goldenglue', 'WlN ONLY', 'Malevolent mid', 'Trash you', 'Damonte', 'Anivia Kid', 'Huhi',
# BOT: 'Doublelift', 'C9 Sneaky', 'TSM Zven', 'WildTurtle', 'Apollo Price', '100T Bang', 'Cody Pog', 'Stixxay', 'Deftly', 'OpTic Arrow',
# SUP: 'From Iron', '5tunt', 'Hakuho', 'TSM Smoothie', 'Sun Prince', 'FLY JayJ', 'Vulcan 01', 'Bíg T', 'Aphromoo', 'Humble Diligent'
################################################
pro_usernames = [
  
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