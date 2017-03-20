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
a = np.array(frequency)
p = np.percentile(a, 25) # return 25th percentile
q= np.percentile(a,50) # returns 50th percentile (Eg Median)
r= np.percentile(a,75)
s=np.percentile(a,100)
t=np.percentile(a,0)

print t,p,q,r,s

from matplotlib import mlab

d = [1.0,2.0,17.0,53.25,566]

# Percentile values
p = np.array([0.0, 25.0, 50.0, 75.0, 100.0])

perc = mlab.prctile(d, p=p)

plt.plot(d)
# Place red dots on the percentiles
plt.plot((len(d)-1) * p/100., perc, 'ro')

# Set tick locations and labels
plt.xticks((len(d)-1) * p/100., map(str, p))
x_labels = [1.0,2.0,17.0,53.25,566]
#plt.set_xticklabels(x_labels)
plt.title('Percentile distribution based on the number of releases')
plt.ylabel("Number of Releases")
plt.xlabel("Percentiles and 1st,2nd,3rd and 4th Quartiles")
plt.show()


'''
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
ax.set_title('Percentile distribution of the number of releases')
#k = np.arange(len(a))

#opacity = 0.4
#error_config = {'ecolor': '0.3'}
#ax.bar(k, b, width = 0.5, color = 'r')
#plt.xlim([0,20])
#plt.xticks(k, a, rotation='vertical')


#rcParams['fig.figsize'] = 5, 10

#plt.plot(x,y)
#ax.axis([0,180,0,140000])
plt.subplots_adjust(bottom = 0.3)
plt.show()
'''
