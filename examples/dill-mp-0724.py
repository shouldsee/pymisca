import dill
import itertools
def run_dill_encoded(payload):
    fun, args = dill.loads(payload)
    res = fun(*args)
    res = dill.dumps(res)
    return res

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

if __name__ == '__main__':
    import multiprocessing as mp
    import sys,os
    p = mp.Pool(4)
    res = dill_map_async(p, lambda x:[sys.stdout.write('%s\n'%os.getpid()),x][-1],
                  [lambda x:x+1]*10,)
    res = res.get(timeout=100)
    res = map(dill.loads,res)
    print(res)
    