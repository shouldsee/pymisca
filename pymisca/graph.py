import itertools
import collections
import pandas as pd
def graph_build(lst, 
                as_dict=1, 
                as_df=1,
                sep=None,
                debug=0,
               ):
    lst = sorted(lst,key=len)
    it = itertools.groupby(lst,key=len)
    it = [(x,list(y)) for x,y in it   ]
    it = collections.OrderedDict(it)

    out = collections.OrderedDict()
    def level_build((level, nodes)):
        lastLevel = it.get(level - 1, [])
        def getParent(node,):
#             print node
            parents = [x for x in lastLevel if node[:-1] == x]
            if debug:
                print (node,parents)
            if len(parents) ==0:
                return [], node
            else:
                assert len(parents) == 1,str(parents)
                return parents[0],node

        out[level] = res =map(getParent,nodes)
        return res

    map(level_build, it.iteritems())    

#     if as_dict or as_df:
    if 1:
        out = sum( (out).values(),[])
#         out[0] = 
        out = map(lambda x:map(tuple,x),out)
        out = map(tuple, tuple(map(tuple,out)))
#         out = collections.OrderedDict(out)

        if as_df:
#             out = pyext.mapDict( sep.join, out, mapkey=1)
            
            df = pd.DataFrame(out,columns=['FROM','TO'])
            
#             df['from'] = out.keys()
#             df['to']   = out.values()
            df['DIFF'] = df['TO'].str.get(-1)
            df['FROM'] = df['FROM'].map(sep.join)
            df['TO'] = df['TO'].map(sep.join)
#             df.columns = ['FROM','TO','DIFF']
            out = df
#             out = pd.DataFrame( out, columns=['from','to'])

    return out