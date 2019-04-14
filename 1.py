# python2.7
import re
lineList=[] #/home/admin/logs/
with open('webx.log','rw') as f:
    for line in f:
        if 'Login' in line:
                lineList.append(line)
lineList.sort()

loginSet = set(lineList)
# set gets rid of the repeated elements
dictList=dict()
# using dict to hold key-value pairs
for item in loginSet:
    for i in range(len(lineList)):
        if lineList[i]==item:
            if item not in dictList:
                dictList[item]=1
            else dictList[item]=dictList[item]+1
# there seems to be key-value pairs in set() too, but I am not very familiar with that
valuesKeys = []
for key,value in dictList:
    valuesKeys.append((value,key))
valuesKeys=values.sort()
valuesKeys=valuesKeys.reverse() 
for item in valuesKeys:
	print item
    
# done!