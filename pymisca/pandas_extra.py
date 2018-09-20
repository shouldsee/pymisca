from __future__ import absolute_import

import pandas as pd
from pandas import *
__all__ = ['reset_columns',
          'printIndex',
          'melt']

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
#     print s
    return s
    
    
def melt(self,):
    ''' specified melting
'''
    df = pd.melt(
        self.reset_index(),
        id_vars=self.index.name,
#         **kwargs
    )       
    df = df.rename(columns = {self.index.name:'ind',
                             self.columns.name:'col'})
    return df


pd.DataFrame.reset_columns = reset_columns
pd.DataFrame.printIndex = printIndex


del absolute_import