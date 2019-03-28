from champions import championId_list
from classifier import json_roster_to_array

def valid_positions(json_roster):
  roles = ["b_top","b_jung","b_mid","b_bot","b_sup","r_top","r_jung","r_mid","r_bot","r_sup"]
  if all(role in json_roster for role in roles):
    print()
    print('[ REQUEST CONTAINS VALID ROSTER POSITIONS ]')
    print(json_roster)
    return True
  else:
    print()
    print('[ REQUEST CONTAINS INVALID ROSTER POSITIONS ]')
    print(json_roster)
    return False

def valid_championIds(json_roster):
  request_array_roster = json_roster_to_array(json_roster)
  if all(championId in championId_list for championId in request_array_roster):
    print()
    print('[ REQUEST CONTAINS VALID CHAMPION IDs ]')
    print(request_array_roster)
    return True
  else:
    print()
    print('[ REQUEST CONTAINS INVALID CHAMPION IDs ]')
    print(request_array_roster)
    return False


# # test
# json_roster = {
#   "b_top": "5",
#   "b_jung": "64",
#   "b_mid": "1",
#   "b_bot": "29",
#   "b_sup": "63",
#   "r_top": "17",
#   "r_jung": "24",
#   "r_mid": "238",
#   "r_bot": "51",
#   "r_sup": "432"
# }
# valid_championIds(json_roster)