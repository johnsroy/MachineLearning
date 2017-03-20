#/bin/python

import os
import matplotlib.pyplot as plt
import numpy as np
path='/Users/roysourish/Desktop/SENG 607'
dirname='project'

x=[]
y=[]
d={}

path = path+"/"+dirname
dirs = os.listdir(path)  #list all the dir in path
#print (dirs)

folders = ([dirname for dirname in os.listdir(path)
            if os.path.isdir(os.path.join(path, dirname))])
#print folders


# get all directories
for folder in folders:
    contents = os.listdir(os.path.join(path,folder))# get list of contents
    y.append(len(contents)-1)
    x.append(folder)

d= {x:y.count(x) for x in y}

frequency = sorted(d.items(), key = lambda i: i[0])

print frequency

a=[]
b=[]
i=0
for element in frequency:
    a.append(element[0])
    b.append(element[1])

fig=plt.figure()
ax = fig.add_subplot(111)

plt.plot(a,b)
plt.xlabel("Release number")
plt.ylabel("Frequency")
ax.set_title('Frequency distribution of apps with 20 releases')
#k = np.arange(len(a))

#opacity = 0.4
#error_config = {'ecolor': '0.3'}
#ax.bar(k, b, width = 0.5, color = 'r')
plt.xlim([0,20])
#plt.xticks(k, a, rotation='vertical')
#plt.rcParams.update({'font.yticks': 2})

#rcParams['fig.figsize'] = 5, 10

#plt.plot(x,y)
#ax.axis([0,180,0,140000])
plt.subplots_adjust(bottom = 0.3)
plt.show()
