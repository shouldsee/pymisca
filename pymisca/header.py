import os,sys
import pymisca.shell as pysh
def set__numpy__thread(NCORE = None):
    if NCORE is None:
#     if 'NCORE' not in locals():
        print ("[WARN] NUMPY is not limited cuz NCORE is not set")
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

def base__check(BASE='BASE',strict=0):
    res = os.environ.get(BASE,'')
    if res == '':
        if strict:
            raise Exception('variable ${BASE} not set'.format(**locals()))
        else:
            PWD =  os.getcwd()
            print ('[WARN] variable ${BASE} not set,defaulting to PWD:{PWD}'.format(**locals()))
            os.environ[BASE] = PWD
    print('[%s]=%s'%(BASE,os.environ[BASE]))
    return os.environ[BASE]
#     print('[BASE]=%s'%os.environ[BASE])
    
def base__file(fname, BASE=None, HOST='BASE', force = False,silent= 1):
    '''find a file according under the directory of environment variable: $BASE 
    '''
    if not isinstance(BASE, basestring):
        BASE  = os.environ.get( HOST,None)
        assert BASE is not None
    BASE = BASE.rstrip('/')
    res = os.path.join(BASE,fname)
    if BASE.startswith('/'):
        existence = os.path.exists(res)
        if not force:
            assert existence,'BASE={BASE},res={res}'.format(**locals())
        else:
            if not existence:
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
        