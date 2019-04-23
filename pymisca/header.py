import os,sys
import pymisca.shell as pysh
import warnings



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

def base__check(BASE='BASE',strict=0,silent=0):
    res = os.environ.get(BASE,'')
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
    
def nTuple(lst,n,silent=1):
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
    