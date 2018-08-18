import pandas as pd
from pandas import *

def reset_columns(df,columns=None):
    '''similar to reset_index()
'''
    df.columns = range(len(df.columns)) if columns is None else columns
    return df


def printIndex(df= None,index = None):
    '''How about we just print the index ONLY?
'''
    if index is None:
        index = df.index
    s = index.to_series().to_frame().to_csv(index=0)
    print s
    


pd.DataFrame.reset_columns = reset_columns
pd.DataFrame.printIndex = printIndex