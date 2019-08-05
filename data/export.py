import pandas as pd
import numpy

class Export():
    def __init__(self):
        print('init data')
    def read(self,list,path):
        #read the file and output a list
        data_file = open(path,'rt', errors='replace')
        for line in data_file:
            #erase all backspaces and save to the list
            #line=line.replace(ord(\),"")
            list.append(line.rstrip())

        #check and output the lsit
        return list
    def print(self,list,n_elements):
        print(list[0:n_elements])
        print('\n')
    def create_df(self,list):
        df = pd.DataFrame(list)
        return df
    def exec(self):
        print('exec data processing...')
