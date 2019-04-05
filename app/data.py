import csv, json, sys
import os.path
#if you are not using utf-8 files, remove the next line

# takes relative directory and converts all JSON matches within into csv
def generateCsv(directory):
  for jsonFile in os.listdir(directory):
    inputFile = open( directory + jsonFile )

    if directory == './seed/':
      outputFile = open('seed.csv','w', newline='')
    else:
      outputFile = open('data.csv','w', newline='')

    csvWriter = csv.writer(outputFile)
    matchesObject = json.load(inputFile)
    matchesArray = matchesObject['matches']

    for match in matchesArray:
      participants = match['participants']
      teams = match['teams']
      winner = ''
      if teams[0]['win'] == 'Win':
        winner = '100'
      else:
        winner = '200'

      championsTuple = [None]*11
      for participant in participants:
        
        lane = participant['timeline']['lane']
        role = participant['timeline']['role']
        team = participant['teamId']
        
        if lane == "TOP" and team == 100:
          championsTuple[0] = participant['championId']
        if lane == "JUNGLE" and team == 100:
          championsTuple[1] = participant['championId']
        if lane == "MIDDLE" and team == 100:
          championsTuple[2] = participant['championId']
        if role == "DUO_CARRY" and team == 100:
          championsTuple[3] = participant['championId']
        if role == "DUO_SUPPORT" and team == 100:
          championsTuple[4] = participant['championId']
        if lane == "TOP" and team == 200:
          championsTuple[5] = participant['championId']
        if lane == "JUNGLE" and team == 200:
          championsTuple[6] = participant['championId']
        if lane == "MIDDLE" and team == 200:
          championsTuple[7] = participant['championId']
        if (role == "DUO_CARRY" or role == "DUO") and (championsTuple[8] == None) and team == 200:
          championsTuple[8] = participant['championId']
        if (role == "DUO_SUPPORT" or role == "DUO") and team == 200:
          championsTuple[9] = participant['championId']
        
      championsTuple[10] = winner
      
      if None not in championsTuple:
        csvWriter.writerow(championsTuple)

    inputFile.close()
    outputFile.close()

def getMatchDataDirectory(dataSource):
  print('dataSource =', dataSource)
  if dataSource != 'seed':
    print('-----------> USING data dir')
    return './data/'
  else:
    print('-----------> USING seed dir')
    return './seed/'

def insertCsvIntoMatchesDb(dataSource):
  

# print(sys.argv)
# if len(sys.argv) == 2:
#   print('arg[1]', sys.argv[1])
#   dataDirectory = getMatchDataDirectory(sys.argv[1])
# else:
#   dataDirectory = getMatchDataDirectory('data')

# generateCsv(dataDirectory)