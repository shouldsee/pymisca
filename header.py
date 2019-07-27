import os,sys,re
import warnings
import itertools,functools
import pymisca.shell as pysh
import inspect
import sys,inspect
import operator
import collections
import sys
import types
_this_mod = sys.modules[__name__]


def sanitise__argName(s):
# def funcArgName_sanitised(s,):
    return re.sub("[^a-zA-Z0-9_]",'_',s)

def PlainFunction(f):
    f._plain =True
    return f
def is__plainFunction(f):
    return getattr(f,'_plain',False)

def func__setAs(other, setter, name=None):
    class temp():
        _name = name
    def dec(func):
#         functools.wraps(f)
        if temp._name is None:
            temp._name = func.__name__

        setter(other, temp._name, func)
        return func
    return dec    
def func__setAsAttr(other, *a):
    return  func__setAs(other, setattr,*a)
setAttr = func__setAsAttr

def func__setAsItem(other, *a):
    return  func__setAs(other, operator.setitem,*a)
setItem = func__setAsItem



@setAttr(_this_mod, "renameVars")
def func__renameVars(varnames=['xxx','y']):
    def dec(f,varnames=varnames):
        # code = copy.copy(f.__code__)
        code = f.__code__
        if isinstance(varnames, list):
            _varnames = tuple(map(sanitise__argName,varnames))
        if isinstance(varnames, collections.MutableMapping):
            _varnames = tuple( sanitise__argName( varnames.get(x,x) ) 
                              for x in code.co_varnames[:])

        assert len(_varnames) == len(code.co_varnames),(_varnames, code.co_varnames)

        _code = types.CodeType(
            code.co_argcount,
            code.co_nlocals,
            code.co_stacksize,
            code.co_flags,
            code.co_code,
            code.co_consts,
            code.co_names,
            _varnames,  #     code.co_varnames,
            code.co_filename,
            code.co_name,
            code.co_firstlineno,
            code.co_lnotab,
            code.co_freevars,
            code.co_cellvars,
        )
        g = types.FunctionType( _code, 
                               f.__globals__, 
                               f.__name__, 
                               f.__defaults__, 
                               f.__closure__)
        return g
    return dec

#     return 
# def func__setAsAttr(other, name=None,setter=setattr):
#     class temp():
#         _name = name
#     def dec(func):
# #         functools.wraps(f)
#         if temp._name is None:
#             temp._name = func.__name__

#         setter(other, temp._name, func)
#         return func
#     return dec
# setAttr = func__setAsAttr

def func__setAsItem(other,name = None):
    
    pass
# _devnull
class Suppresser:
    devnull = open(os.devnull, "w")
    def __init__(self, suppress_stdout=False, suppress_stderr=False):
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
        self.original_stdout = None
        self.original_stderr = None
#         self.devnull = 

    def _switch(self, toSuppress=None):
        if toSuppress is None:
            toSuppress = self.original_stdout is None
        if self.suppress_stdout:
            if toSuppress:
                if sys.stdout is not self.devnull:
                    self.original_stdout, sys.stdout = sys.stdout, self.devnull
            else:
                if sys.stdout is self.devnull:
                    self.original_stdout, sys.stdout = None, self.original_stdout

        if self.suppress_stderr:
            if toSuppress:
                if sys.stdout is not self.devnull:
                    self.original_stderr, sys.stderr = sys.stderr, self.devnull
            else:
                if sys.stdout is self.devnull:
                    self.original_stderr, sys.stderr = None, self.original_stderr
        return 
    
    def close(self):
        self._switch(0)
        return self
    
    def suppress(self):
        self._switch(1)
        return self
#     def suppress(self):
#         self._switch(1)
        
    def __enter__(self):
        import sys, os
        pass
#         print '[pas'
#         self._switch()
    
    def __exit__(self, *args, **kwargs):
        import sys
        self._switch()
        
#     def __enter__(self):
#         import sys, os        
#         if self.suppress_stdout:
#             self.original_stdout, sys.stdout = sys.stdout, self.devnull
# #             sys.stdout = self.devnull

#         if self.suppress_stderr:
#             self.original_stderr, sys.stderr = sys.stderr, self.devnull
# #             self.original_stderr = sys.stderr
# #             sys.stderr = self.devnull

#     def __exit__(self, *args, **kwargs):
#         import sys
#         # Restore streams
#         if self.suppress_stdout:
#             sys.stdout = self.original_stdout

#         if self.suppress_stderr:
#             sys.stderr = self.original_stderr

class Suppress:
    def __init__(self, suppress_stdout=1, suppress_stderr=1):
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
        self.original_stdout = None
        self.original_stderr = None

    def __enter__(self):
        import sys, os
        devnull = open(os.devnull, "w")

        # Suppress streams
        if self.suppress_stdout:
            self.original_stdout = sys.stdout
            sys.stdout = devnull

        if self.suppress_stderr:
            self.original_stderr = sys.stderr
            sys.stderr = devnull

    def __exit__(self, *args, **kwargs):
        import sys
        # Restore streams
        if self.suppress_stdout:
            sys.stdout = self.original_stdout

        if self.suppress_stderr:
            sys.stderr = self.original_stderr


def module__toModule(mod):
    res = collections.OrderedDict()
    res['NAME']=mod.__name__
    res['INPUT_FILE'] = getattr(mod,"__file__",None)
    res['MODULE']=mod
    return res

def get__defaultModuleDict():
#     import sys
    d = {}
    for k,v in sys.modules.items():
        if v is not None:
            d[k] = module__toModule(v)
        else:
            pass
#             sys.stderr.write(k + '\n')
    return d
#     return {k:module__toModule(v) for k,v in sys.modules.items() if v is not None
           
#            }

def module__getClasses(mod):
    '''https://stackoverflow.com/a/1796247/8083313
    '''
    if isinstance(mod,basestring):
        mod = sys.modules[mod]
    clsmembers = dict(inspect.getmembers(mod, inspect.isclass))
    return clsmembers


def get__frameDict(frame=None,level=0, getter="dict"):
    return get__frame(frame,level=level+1,getter=getter)

def get__frameName(frame=None,level=0, getter="func_name"):
    return get__frame(frame,level=level+1,getter=getter)
    
def get__frame(frame=None,level=0, getter="dict"):
    '''
    if level==0, get the calling frame
    if level > 0, walk back <level> levels from the calling frame
    '''
    if frame is None:
        frame = inspect.currentframe().f_back

    for i in range(level):
        frame = frame.f_back
    _getter  = {
        "dict":lambda x:x.f_locals,
        "func_name":lambda x:x.f_code.co_name
    }[getter]
    context = _getter(frame)
#     context = frame.f_locals
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

def execBaseFile(fname,**kw):
    fname = base__file(fname,**kw)
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
    
    
class columns(object):
    gtf = ['SEQID','SOURCE','TYPE','START','END','SCORE','STRAND','PHASE','ATTRIBUTES']
    

columns.bed = bedHeader = [
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
    
columns.betty = bettyHeader = ['OLD_DATA_ACC','SPEC_PATHOGEN','SPEC_HOST','TREATMENT','LIB_STRATEGY']


buf = '''
table genePredExt
"A gene prediction with some additional info."
    (
    string name;        	"Name of gene (usually transcript_id from GTF)"
    string chrom;       	"Chromosome name"
    char[1] strand;     	"+ or - for strand"
    uint txStart;       	"Transcription start position"
    uint txEnd;         	"Transcription end position"
    uint cdsStart;      	"Coding region start"
    uint cdsEnd;        	"Coding region end"
    uint exonCount;     	"Number of exons"
    uint[exonCount] exonStarts; "Exon start positions"
    uint[exonCount] exonEnds;   "Exon end positions"
    int score;            	"Score"
    string name2;       	"Alternate name (e.g. gene_id from GTF)"
    string cdsStartStat; 	"Status of CDS start annotation (none, unknown, incomplete, or complete)"
    string cdsEndStat;   	"Status of CDS end annotation (none, unknown, incomplete, or complete)"
    lstring exonFrames; 	"Exon frame offsets {0,1,2}"
    )
'''
columns.genepred = COLUMNS_GENEPREDEXT =  re.findall('.*\s+([a-zA-Z0-9]+);.*',buf)
