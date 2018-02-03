

file = open('./wnids.txt')
netIDs = []
for id in file.readlines():
    netIDs.append(id.split('\n')[0])
print(netIDs)