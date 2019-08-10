import dill
import itertools
import pymisca.header
##### Multiprocessing map
import multiprocessing as mp
dill.settings['recurse']=True

def run_dill_encoded(payload):
    fun, args = dill.loads(payload)
    res = fun(*args)
    res = dill.dumps(res)
#     if isinstance(args,tuple):
#         pass
#     else:
#     args = (args,)
    return res

@pymisca.header.setAttr(mp.Pool,'dill_map_async')
def dill_map_async(pool, fun, args_list,
                   as_tuple=True,
                   **kw):
    if as_tuple:
        args_list = ((x,) for x in args_list)
        
    it = itertools.izip(
        itertools.cycle([fun]),
        args_list)
    it = itertools.imap(dill.dumps, it)
    return pool.map_async(run_dill_encoded, it, **kw)

def mp_map(f,lst,n_cpu=1, chunksize= None, callback = None, 
           NCORE=None,use_dill=1,
#            kwds = {}, 
           **kwargs):
    if NCORE is not None:
        n_cpu = NCORE
    if n_cpu > 1:
        p = mp.Pool(n_cpu,**kwargs)
        if use_dill:
            OUTPUT= dill_map_async(p, f,lst, 
                                   chunksize=chunksize, 
    #         OUTPUT=p.map_async(f,lst, chunksize=chunksize, 
    #                            kwds = kwds,
                               callback = callback)
            OUTPUT = OUTPUT.get(999999999999999999)
            OUTPUT = map(dill.loads, OUTPUT)
        else:
            OUTPUT= p.map_async(f,lst, chunksize=chunksize, 
                               kwds = kwds,
                               callback = callback)            
            OUTPUT = OUTPUT.get(999999999999999999) ## appx. 7.6E11 years
#         OUTPUT = p.map(f,lst)
        p.close()
        p.join()
    else:
        OUTPUT = map(f,lst)
    return OUTPUT