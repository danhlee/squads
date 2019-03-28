-1st 5 objects in match.participants are team 100 (blue team)
-2nd 5 objects in match.participants are team 200 (red team)
-order for csv:
    b_top,b_jung,b_mid,b_adc,b_sup, r_top,r_jung,r_mid,r_adc,r_sup,winning_team

    winning_team = 100 or 200

match.participants[0] through match.participants[4]
match.participants[5] through match.participants[9]




for participant in match.participants:
    if (participant.teamId = 100):
        if (participant.timeline.lane == "TOP")

    
    participant.timeline.lane == "JUNGLE"
    participant.timeline.lane == "MIDDLE"
    participant.timeline.lane == "BOTTOM" && participant.timeline.role == "DUO_CARRY"
    participant.timeline.lane == "BOTTOM" && participant.timeline.lane == "SUPPORT"

for participant in match.participants:
    participant.timeline.lane == "TOP"
    participant.timeline.lane == "JUNGLE"
    participant.timeline.lane == "MIDDLE"
    participant.timeline.lane == "BOTTOM" && participant.timeline.role == "DUO_CARRY"
    participant.timeline.lane == "BOTTOM" && participant.timeline.lane == "SUPPORT"


i = 0
while i < 5
    // do stuff
    i += 1
