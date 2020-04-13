import random
LENGTH = 256
pos = 0
for x in range(LENGTH):
    if(pos >=random.randrange(6,15) or pos >= 10):
        slope = random.randrange(0,10)/-10.0
        pos = 10
    elif(pos <=0):
        slope = random.randrange(0,10)/10.0
        pos = 0
    pos = pos + slope
    print(str(pos) + ",")