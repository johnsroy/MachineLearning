import glob
import os
import pandas as pd

path =r'/Users/roysourish/Desktop/SENG 607/project/batphone/' # use your path
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    #if (file_ == 'release_dates.csv'):
    #pass
    #else:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)
frame = pd.concat(list_)

print list_