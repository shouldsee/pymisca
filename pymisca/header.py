import os,sys
import warnings
import itertools
import pymisca.shell as pysh

import inspect

import sys,inspect
def module__getClasses(mod):
    '''https://stackoverflow.com/a/1796247/8083313
    '''
    if isinstance(mod,basestring):
        mod = sys.modules[mod]
    clsmembers = dict(inspect.getmembers(mod, inspect.isclass))
    return clsmembers


def get__frameDict(frame=None,level=0):
    '''
    if level==0, get the calling frame
    if level > 0, walk back <level> levels from the calling frame
    '''
    if frame is None:
        frame = inspect.currentframe().f_back

    for i in range(level):
        frame = frame.f_back
    context = frame.f_locals
    del frame
    return context

def runtime__dict():
    import __main__
    res = vars(__main__)
    return res

def runtime__file(silent=1):
    dct = runtime__dict()
    res = dct['__file__']
    if not silent:
        sys.stdout.write (res+'\n')
    return res


def set__numpy__thread(NCORE = None):
    if NCORE is None:
#     if 'NCORE' not in locals():
        warnings.warn("[WARN] NUMPY is not limited cuz NCORE is not set")
    else:
        #     print (')
        keys = '''
        OMP_NUM_THREADS: openmp,
        OPENBLAS_NUM_THREADS: openblas,
        MKL_NUM_THREADS: mkl,
        VECLIB_MAXIMUM_THREADS: accelerate,
        NUMEXPR_NUM_THREADS: numexpr
        '''.strip().splitlines()
        keys = [ x.split(':')[0] for x in keys]


        try:
            ipy = get_ipython()
        except:
            ipy = None

        for key in keys:
            val = str(NCORE)
            if ipy:
                ipy.magic('env {key}={val}'.format(**locals()))
            os.environ[key] = val

def mpl__setBackend(bkd='agg',
                   whitelist = ['module://ipykernel.pylab.backend_inline']):
#     exception = '
#     print ('debug',"matplotlib" in sys.modules)
#     if "matplotlib" not in sys.modules:
#     if 1:
    if bkd is None:
        bkd = 'agg'
    import matplotlib
    bkdCurr = matplotlib.get_backend()
    if (bkdCurr != bkd) and (bkdCurr not in whitelist):
        matplotlib.use(bkd)

def base__check(BASE='BASE',strict=0,silent=0,
#                 default=None
               ):
#     if default is None:
#         default = ''
    default = ''
    res = os.environ.get(BASE, default)
    
    if res == '':
        if strict:
            raise Exception('variable ${BASE} not set'.format(**locals()))
        else:
#             PWD =  os.getcwd()
            PWD = pysh.shellexec("pwd -L").strip()
#             if not silent:
            warnings.warn('[WARN] variable ${BASE} not set,defaulting to PWD:{PWD}'.format(**locals()))
            os.environ[BASE] = PWD
    if not silent:
        print('[%s]=%s'%(BASE,os.environ[BASE]))
    return os.environ[BASE]
#     print('[BASE]=%s'%os.environ[BASE])
    
def base__file(fname='', 
               BASE=None, HOST='BASE', 
               baseFile = 1,
               force = False,silent= 1, asDIR=0):
    
    '''find a file according under the directory of environment variable: $BASE 
    '''
    if fname is None:
        fname = ''
        
    if baseFile == 0:
        return fname
    elif isinstance(baseFile,basestring):
        BASE=baseFile
    
    if not isinstance(BASE, basestring):
        BASE = base__check(strict = 1,silent=silent)
#        BASE  = os.environ.get( HOST,None)
#        assert BASE is not None
    BASE = BASE.rstrip('/')
    fname = fname.strip()
    res = os.path.join(BASE,fname)
    if BASE.startswith('/'):
        existence = os.path.exists(res)
        if not force:
            assert existence,'BASE={BASE},res={res}'.format(**locals())
        else:
            if not existence:
                if asDIR:
                    pysh.shellexec('mkdir -p {res}'.format(**locals()), silent=silent)
                else:
                    pysh.shellexec('touch {res}'.format(**locals()), silent=silent)
                
        with open('%s/TOUCHED.list' % BASE, 'a') as f:
            f.write(fname +'\n')
    return res        

def execBaseFile(fname,):
    fname = base__file(fname)
    g= vars(sys.modules['__main__'])
#     g = __main__.globals()
    res = execfile(fname, g, g)
#     exec(open(fname).read(), g)
    return
    
def list__nTuple(lst,n,silent=1):
    """ntuple([0,3,4,10,2,3], 2) => [(0,3), (4,10), (2,3)]
    
    Group a list into consecutive n-tuples. Incomplete tuples are
    discarded e.g.
    
    >>> group(range(10), 3)
    [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    """
    if not silent:
        L = len(lst)
        if L % n != 0:
            print '[WARN] nTuple(): list length %d not of multiples of %d, discarding extra elements'%(L,n)
    return zip(*[lst[i::n] for i in range(n)])    
nTuple = list__nTuple

def it__window(seq, n=2,step=1,fill=None,keep=0):
    '''Returns a sliding window (of width n) over data from the iterable
   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...  
   Adapted from: https://stackoverflow.com/a/6822773/8083313
'''   
    it = iter(seq)
    result = tuple(itertools.islice(it, n))    
    if len(result) < n:
        result = result + (fill,) * (n-len(result))
        if keep:
            yield result
        else:
            pass
        return
    else:
        yield result
#     else:

    while True:        
        elem = tuple( next(it, fill) for _ in range(step))
        result = result[step:] + elem        
        if elem[-1] is fill:
            if keep:
                yield result
            break
        yield result
    pass    
window = it__window

def it__len(it):
    it,_it = itertools.tee(it)
    i = -1
    for i, _ in enumerate(_it):
        pass
    return it, i + 1

def self__install(
    lst=['/data/repos/pymisca'
                  ]):
    if isinstance(lst,basestring):
        lst = [lst]
    CMDS =[ 'cd {DIR} && python2 setup.py install --user &>LOG && echo DONE'.format(DIR=x) for x in lst]
    res = map(pysh.shellexec,CMDS)
    print (res)
    
bedHeader = [
   'chrom',
 'start',
 'end',
 'acc',
 'score',
 'strand',
 'FC',
 'neglogPval',
 'neglogQval',
 'summit',
 'img']
    
bettyHeader = ['OLD_DATA_ACC','SPEC_PATHOGEN','SPEC_HOST','TREATMENT','LIB_STRATEGY']