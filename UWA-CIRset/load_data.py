import os
path = os.getcwd()
f = open(path+'/data.txt', 'r')
len = 0
input = []
label = []
for ln in f:
    lns = ln.split()
    input.append(lns[0])
    label.append(lns[1])
    len += 1

f.close()