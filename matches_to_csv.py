import csv, json, sys
import os.path
#if you are not using utf-8 files, remove the next line

#check if you pass the input file and output file

directory = './matches/'
for jsonFile in os.listdir(directory):
  

  inputFile = open( directory + jsonFile )
  if os.path.isfile('matches.csv'):
    outputFile = open('matches.csv','a', newline='')
  else:
    outputFile = open('matches.csv','w', newline='')

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

    #match_tuple = ''
    championsTuple = [None]*11
    for participant in participants:
      #match_tuple += ( str(participant['championId']) + ',')
      
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