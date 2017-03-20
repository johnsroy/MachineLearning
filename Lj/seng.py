#/bin/python

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

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
print len(folders)
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

#plt.plot(a,b)
plt.xlabel("Number of Releases")
plt.ylabel("Frequency (log scale)")
ax.set_title('Frequency distribution of apps based on the number of releases')
k = np.arange(len(a))

#opacity = 0.4
#error_config = {'ecolor': '0.3'}
#plt.hist(k, bins=50, normed=True)
ax.bar(k, b, width = 1.0, color = 'y')
#plt.xlim([-1, k.size])
#plt.xlim([0,20])
ax.set_yscale('log')
#plt.ylim([0,100]
#plt.xticks(k, a, rotation='vertical')
#plt.rcParams.update({'font.yticks': 2})

#rcParams['fig.figsize'] = 5, 10

#plt.plot(x,y)
#formatter = FuncFormatter(k)
#plt.gca().yaxis.set_major_formatter(formatter)
#plt.subplots_adjust(bottom = 0.3)


#formatter = FuncFormatter(lambda y, pos: "%d%%" % (y))
#ax.yaxis.set_major_formatter(formatter)
plt.show()
