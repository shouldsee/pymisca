import os,sys
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

def mpl__setBackend(bkd='agg'):
#     print ('debug',"matplotlib" in sys.modules)
#     if "matplotlib" not in sys.modules:
#     if 1:
    import matplotlib
    if not matplotlib.get_backend() == bkd:
        matplotlib.use(bkd)
