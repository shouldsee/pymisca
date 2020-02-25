from __future__ import absolute_import

import pandas as pd
from pandas import *
__all__ = ['reset_columns',
          'printIndex',
          'melt',
           'df__iterTupleList',
           'df__fromTupleList',
          ]
from pymisca.header import setAttr,setItem



import itertools
import pandas as pd
#@setItem(__all__,-1)
import numpy
import itertools
@setAttr(pd.DataFrame, "iterTupleList")
def df__iterTupleList(df, columns = 1,index=0, name = None):
    assert index==0,"not implemented"
    assert name is None,"not impl"
    
    df = df.replace({pd.np.nan: None})
    it = df.itertuples(index=index, name=name)
    
    if columns:
        _dt = list([(unicode(x),y.str) for x,y in df.dtypes.items()])
        it = itertools.chain([_dt],it)
    return it

def df__fromTupleList(arr,header=1):
    if header:
        _dtype,arr = arr[0],arr[1:]
    else:
        _dtype = list([('f%d'%(i),'O') for i in range(len(arr[0]))])    
        
    if _dtype is not None:
        if not isinstance(_dtype,numpy.dtype):
            _dtype = numpy.dtype((numpy.record,  list(_dtype) ))
        
    res = pd.np.rec.array(arr,
                  _dtype,
                  )
    res = pd.DataFrame(res)
    return res

if __name__ == '__main__':
    df = pandas.DataFrame([[0,'1'],[2,3]])
    it = df__iterTupleList(df)
    it = list(it)
    print (it)
    df__fromTupleList(it)
    print (df__fromTupleList(it[1:],columns=0)['f0'])
    
# def setAttr(other, name=None):
#     class temp():
#         _name = name
#     def dec(func):
#         if temp._name is None:
#             temp._name = func.__name__
# #         assert 0,(other,temp._name,func)
#         setattr(other,temp._name,func)
#         return func
#     return dec
####### set default for df.query()

# pd.DataFrame._query = pd.DataFrame.query
# def query(self,expr,engine='python',**kw):
#     return self._query(expr,engine=engine,**kw)
# pd.DataFrame.query = query

if not hasattr(pd.DataFrame,'_eval'):
#     pd.Data
    pd.DataFrame._eval = pd.DataFrame.eval    
    @setAttr(pd.DataFrame)
    def eval(self,expr,engine='python', level=0, **kw):
        return self._eval(expr,engine=engine, level=level+1, **kw)

# pd.DataFrame.eval = evaluate

########

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
# try:
import re

@setAttr(pd.DataFrame)
def toJSON(dfc,**kw):
    return dfc.to_json(**kw)

@setAttr(pd.DataFrame,'to_label')
def df__toLabel(_dfc):
    res = pyext.df__paste0( _dfc,_dfc.columns,sep=', ')
    res = ['(%s)'%x for x in res ]
    return res

@setAttr(pd.DataFrame,'to_md')
def df__toMarkDown(df,out=None,width=1,**kw):
    '''Return a markdown table from 
    '''
    del kw
    df = df.reset_index()
    index=0
    sep='|'
    
    
    def _wrap(x):
        x = '|'.join([ _x.strip() or 'NA' for _x in x.split('|')])
        x = '|%s|'%x
#         x = re.sub('(?=\|\s*\|)','NA',x)
        x = re.sub('\|(?=\s*\|)','|NA|',x)
#         if not x.strip():
#             x = 'NA'
        return x 

    shead = df.loc[[],:].to_csv(sep=sep,index=index).splitlines()
    assert len(shead)==1,('cant take multi-index columns',df.columns)
    shead = shead[0]
    lst = []
    lst += [(shead)]
    lst += ['|'.join([ '-'*int(max(1,len(x)*width)) for x in shead.split('|')])]
    lst += df.to_csv(header=0,index=index,sep=sep).splitlines()
    lst = map(_wrap,lst)

    if isinstance(out,basestring):
        out = open(out,'w')
    if hasattr(out,'write'):
        for line in lst:
            out.write(line+'\n')
        return
    else:
        res = '\n'.join(lst)
        return res
        
#     return res
# except Exception as e:
#     print(e)
#     pass
@setAttr(pd.Series,'to_index')
def series2index(clu):
    res = clu.index[clu.values==1]
    return res

del absolute_import