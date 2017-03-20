#/bin/python

import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.use('agg')
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
y.sort()
print y
'''
d= {x:y.count(x) for x in y}

frequency = sorted(d.items(), key = lambda i: i[0])

print frequency

a=[]
b=[]
i=0
for element in frequency:
    a.append(element[0])
    b.append(element[1])

print a, b
'''
fig=plt.figure()
ax = fig.add_subplot(111)

#plt.plot(a,b)
plt.xlabel("Box & Whisker Plot for 40 releases")
plt.ylabel("Number of releases")
ax.set_title(' Box and Whisker Plot for apps with 40 Releases')
#k = np.arange(len(a))
#bp = ax.boxplot(data_to_plot)
box = plt.boxplot(y)
plt.setp(box['boxes'][0], color='green')
plt.setp(box['caps'][0], color='green')
plt.setp(box['whiskers'][0], color='green')
#ax.boxplot(a)
#df = pd.DataFrame(z, index=['Age of pregnant women', 'Age of pregnant men'])
#plt.ylim([0,580])
plt.grid(True, axis='y')
#df.T.boxplot(vert=False)
plt.subplots_adjust(left=0.25)
plt.ylim(0,40)
plt.show()
