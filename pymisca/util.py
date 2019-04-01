from __future__ import absolute_import

import os,sys,subprocess
import json
import functools, itertools, copy,re
import urllib2,urllib,io    
import operator


if 'matplotlib' not in sys.modules:
    import matplotlib; mpl = matplotlib; mpl.use('Agg')
else:
    matplotlib = mpl = sys.modules['matplotlib']
plt = getattr(matplotlib,'pyplot',None)
if plt is None:
    import matplotlib.pyplot as plt
#     try:
#         plt = matplotlib.pyplot
#     except Exception as e:
#         import matplotlib.pyplot as plt
    
# try:
#     mpl.use('Agg')
#     import matplotlib.pyplot as plt
# except Exception as e:
#     sys.stderr.write('[WANR]%s\n'%e)
    
cluMap = mpl.colors.ListedColormap(['r', 'g', 'b', 'y', 'w', 'k', 'm'])

import StringIO

from pymisca.jbrowse import *

import numpy as np
from pymisca.oop import *
from pymisca.fop import *
from pymisca.canonic import *
from pymisca.shell import *
from pymisca.ext import *
# from pymisca.jobs import *
import pymisca.jinja2_util as pyjin
import pymisca.ext as pyext

try: 
	import scipy
	from scipy.optimize import curve_fit
	import scipy.stats as spstat#
	import scipy.spatial.distance as spd
	import scipy.spatial.distance as spdist
except Exception as e:
	sys.stderr.write('scipy not installed \n')
try:
	from sklearn.metrics import mutual_info_score
except Exception as e:
	sys.stderr.write('package not installed:%s \n'%'sklearn' )

import warnings
def silentWarning():
    ###### Disable some warnings
    warnings.filterwarnings("ignore", 
                            message="Pandas doesn't allow columns to be created via a new attribute name")
    warnings.filterwarnings("ignore", 
                            module='matplotlib',
                           message = 'Unable to find pixel distance along axis for interval padding of ticks;')
    return 
silentWarning()

pyutil = sys.modules[__name__]

def job__saveFig(figs,exts=['png'],dpi=150,DIR='.') :
    dfig = pyutil.saveFigDict(figs,
                              DIR = DIR,
                              exts=exts,
                             dpi = dpi)
    dfig['fignames'] = [x for x in dfig['fignames'] if x.endswith('.png')]
    buf=[pyutil.ppJson(dfig)]
    ofname = 'figures.json'
    pyutil.printlines(buf,ofname)
    return dfig

def render__images(figs,
                   templateFile=None,
                   DIR='.',
                   exts=['png'],
                   dpi = 150,
                   ofname = None,
                   baseFile=1,
                  ):
    if templateFile is None:
        templateFile = 'Templates/listImages.html'
    templateFile = pyext.base__file(templateFile,baseFile = baseFile)
    if ofname is None:
        ofname = 'figure.html'
    if ofname.startswith('/'):
        pass
    else:
        ofname = pyutil.os.path.join(DIR,ofname)    
    context = pyutil.job__saveFig(figs,
                                   exts=exts,
                                  DIR=DIR,
                                   dpi=dpi)
    ofname = pyjin.quickRender(templateFile=templateFile,
                               context=context,
                               ofname=ofname,
                              )
    return ofname


def tsv__getColumns(fname,ext='tsv',silent=1,
#                     reader = pyutil.readData
                   ):
#     pyutil.readData()
    res = file__header(fname,silent=silent,head=1)
#     res = res.read().split()
#     return res
    df = pyutil.readData(res,ext=ext,guess_index=0)
    return df.columns.tolist()

def df__addCol(df,name = 'testCol', expr='index==index' ):
    df[name] = df.eval(expr)
    return df




    

import datetime
def datenow():
    res = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return res
def lineCount(fname):
    return int(shellexec('wc -l %s'%fname, silent=1).split()[0])

def grepFileByKeys(fname,keys,ofname = None, silent=1):
    ''' grep a given file using a list of keys
'''
    if ofname is None:
        ofname = 'greped__' + pyutil.os.path.basename(fname)
#     expr = u'\|'.join(keys)
    tempFile = pyutil.tempfile.mktemp()
    pyutil.printlines(keys,fname = tempFile)

    cmd = 'grep {fname} -f {tempFile} >{ofname}'.format(**locals())
    res = pyutil.shellexec(cmd,silent=silent)
    return ofname

#### Regex
def arg2dict(s, as_string=0):
#     s = '''a=1,b=2,c=3
#     '''
    lst = re.findall('([^ =]+)[ ]?=[ ]?([^ ,]+)[, ]',s)
    print lst
    dct = {k: (v if as_string else eval(v) ) for k,v in lst}
    return dct
retype = type(re.compile('hello, world'))

ptn = util_obj()
s = '\{([^\{\}]+)\}'
ptn.templateKW =  re.compile(s)

def revSub(ptn, dict):
    '''Reverse filling a regex matcher.
    Adapted from: https://stackoverflow.com/a/13268043/8083313
'''
    if isinstance(ptn, retype):
        ptn = ptn.pattern
    ptn = ptn.replace(r'\.','.')
    replacer_regex = re.compile(r'''
        \(\?P         # Match the opening
        \<(.+?)\>
        (.*?)
        \)     # Match the rest
        '''
        , re.VERBOSE)
    res = replacer_regex.sub( lambda m : dict[m.group(1)], ptn)
    return res

def qc_matrix(C):
    ''' General QC to check matrix is not too big.
'''
    d = collections.OrderedDict()
    d['Mean'],d['Std'],d['Shape'] = C.mean(),C.std(),C.shape
    s = '[qc_matrix]%s'% pyutil.packFlat([d.items()],seps=['\t','='])[0]
    return s



    
# def mapRNA(data,dbg=0):
#     p = subprocess.Popen(['RNAfold','--noPS'],
#                          bufsize=1,
#                          stdout=subprocess.PIPE,
#                          stdin=subprocess.PIPE)
#     s = '\n' + data['data']
    

#     if s is None:
#         res = None
#     else:
#         res = p.communicate(s)[0]

#     data['data'] = res
#     p.stdin.close()
#     return data    
    
    


import itertools
def window(seq, n=2,step=1,fill=None,keep=0):
    '''Returns a sliding window (of width n) over data from the iterable
   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...  
   Adapted from: https://stackoverflow.com/a/6822773/8083313
'''   
    it = iter(seq)
    result = tuple(itertools.islice(it, n))    
    if len(result) <= n:
        yield result
    while True:        
        elem = tuple( next(it, fill) for _ in range(step))
        result = result[step:] + elem        
        if elem[-1] is fill:
            if keep:
                yield result
            break
        yield result
    pass    

def symmetric_hamm(x,y):
    hamm  = np.mean(x == y)
    return 2 * min(hamm,1-hamm)
random_binvec = lambda size: binarise(np.random.randint(2,size= size))
random_binvec_densi = lambda size,densi: binarise([np.random.permutation(size[1]) - densi*float(size[1])  for _ in range(size[0])])
random_bernoulli = lambda size,p: np.random.choice([-1,1],size= size,p = (1-p,p))
def default_false(arr):
    if isinstance(arr,np.ndarray):
        out = np.zeros_like(arr,dtype='bool')
    else:
        out = np.zeros(dtype='bool',shape=arr)
    return out

def binarise(i):
    o = (np.array(i) > 0) * 2 - 1
    return o 
def residual(h,ovect,vect):
    res = np.mean(vect == ovect,axis = 1);
    return np.minimum( res, 1 -res)
def canonlise(vect):
#     d = binarise(np.sum(vect, axis = 1, keepdims = 1) )
    d = np.sign(np.sum(vect, axis = 1, keepdims = 1) )
    isZero = d.ravel()==0
    d[isZero] = np.sign(vect[isZero,0:1])
    vect = d * vect 
    return vect
def make_label(ret,tar,res = None):
    if res is None:
        res = ['']*len(ret)
#     L = len(im)
    
    lab = ['Retrival %d:%s'%(i,r) for (i,r) in zip(range(len(ret)+1)[1:],res)] + ['*Memory %d'%i for i in range(len(tar)+1)[1:]]
    return (range(len(ret)+len(tar)),lab)

    
##### Extra functions for numpy
def stderr(arr, axis = None):
    LEN = arr.shape[axis] if axis is not None else arr.size
    SERR = np.std(arr,axis = axis) / LEN**0.5
    return SERR
np.stderr = stderr  ### pretend this is from NumPy XD


def sigmoid(arr):
    return 1 / (1 + np.exp(-arr))
np.sigmoid = sigmoid



def renorm(x,axis=None,silent=0):
    ''' Renormalising a vector along an axis so that sum up to 1.
    '''
    SUM = np.sum(x,axis=axis,keepdims=1)
    isZero = SUM==0
    if np.any(isZero) and not silent:
        print '[WARN]: Zero sum when normalising vector'
    SUM[isZero]=1
    return x/SUM
##### Extra functions for numpy----End



def enlarge(IN,rd=0):
    if isinstance(IN,np.ndarray):
        x = IN.tolist()
    x = IN[:]
    x[0] = x[0] - rd
    x[1] = x[1] + rd
    return x

def mapper_2d(wFunc, xs,ys, as_list =0):
    out = map(lambda y: 
              map(lambda x: wFunc(x,y),
            xs),
          ys)
    if as_list is 0:
        return np.array(out)
    return out

def worker__mapper2d((x,y),wFunc=None):
    res = wFunc(x,y)
    return res


def mapper_2d(wFunc, xs,ys, NCORE = 1, as_list =0):
    it = itertools.product(xs,ys)
    worker = functools.partial(worker__mapper2d, wFunc=wFunc)
    res = pyutil.mp_map(worker,it,NCORE=NCORE)
    res = pysh.nTuple(res,len(ys))
    if not as_list:
        res = np.array(res)
    return res

def shapeit(A):
    if ~isinstance(A,np.ndarray):
        A = np.array(A)
    if A.ndim is 1:
        A = A[None,:]
    return A
# %timeit a= mapper_2d(lambda x,y:x+y,nTarlst.tolist()*2,sizelst.tolist()*10);



##### A slower mapper
# def mapper_2d(wFunc,xs,ys):
#     col = []
#     for x in xs:
#         row = []
#         for y in ys:
#             row += [wFunc(x,y)]
#         col += [row]
#     return col            

# %timeit b= mapper_2d(lambda x,y:x+y,nTarlst.tolist()*2,sizelst.tolist()*10);

#zip(np.meshgrid(nTarlst,sizelst))
#np.apply_over_axes



##### MI spdist
# from util import *

def bincount(x):
    L = x.size
    x1 = (x==1).sum() 
    x0 = L - x1    
    return (x0,x1)
# %timeit np.histogram(x,bins=[-2,0,2])[0]
# %timeit bincount(x)
# A = h.target
# # x = A[1];y = A[5]
# x = A[1];y = A[3]
# # print x.shape
import collections
import operator
def MI_bin(x,y):
    ##### Assume x,y \in (-1,1)
    L = len(x)
    c = collections.Counter((x + 1) + y*2 + 2 )
    c_xy = [c.get(n,0) for n in [0,2,4,6]] ### (0,0),(1,0),(0,1),(1,1) 
#     if np.any( [x in [0,L] for x in [sum(c_xy[:2]),c_xy[0]+c_xy[2]] ] ):
    if reduce(operator.or_,[x in [0,L] for x in [sum(c_xy[:2]),c_xy[0]+c_xy[2]] ] ):
        return 0
    else:
        c_xy = [c_xy[:2],c_xy[2:]]
        g, p, dof, expected = spstat.chi2_contingency( c_xy,correction=0,
#                                                       lambda_ = 0,
                                                      lambda_="log-likelihood"
                                                     )
        mi = 0.5 * g / sum(map(sum,c_xy))    
    return mi

def MI_sk(x, y, bins=[-2,0,2]):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi
if __name__=='__main__':
    A = [[1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, 1], [1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, -1, -1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1], [1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1], [1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1]]
    A = np.array(A)
    metric = MI_bin
    d = spd.pdist(A.T,metric = metric )
#     %timeit d = spd.pdist(A.T,metric = metric )
    dmat = spd.squareform(d,checks = 0)
    np.fill_diagonal(dmat, map(lambda x: metric (x,x), A))
    plt.imshow(dmat)
    plt.colorbar()
    plt.show()

    metric = sk_MI
    d = spd.pdist(A.T,metric = metric )
#     %timeit d = spd.pdist(A.T,metric = metric )
    dmat = spd.squareform(d,checks = 0)
    np.fill_diagonal(dmat, map(lambda x: metric (x,x), A))
    plt.imshow(dmat)
    plt.colorbar()
    plt.show()

    
    
    
def merge_data(d1,d2):
#     d = d1['data']
    d1['data'] = d1['data'] + d2['data']
#     np.concatenate( (d,d2['data']), axis = 1)
    print d1['marginal']['ys'],d2['marginal']['ys']
    d1['marginal']['ys'] = list(d1['marginal']['ys'])
    d1['marginal']['ys'][1] = d1['marginal']['ys'][1].tolist() + d2['marginal']['ys'][1].tolist()
#     d = d.extend(d2['data'].tolist())
    return d1

def findx(m,xs,thres,axis = 1):
    xpt = np.apply_along_axis( lambda vs:np.interp(thres,vs,xs), axis , m)
    return xpt

def changept(x):
    D = x
    idx = np.where(D[1:]*D[:-1]<0)[0]
    if len(idx)!=0:
        idx = idx[0] 
    else:
        idx = -1
    return idx

##### Curve fitting utilities
import numpy as np

def sigmoid(x, x0, k, mean,span):
     y = span * np.tanh(k*(x-x0)) + mean
#      y = span / (1 + np.exp(-k*(x-x0)))+mean
     return y
def sigmoid_fixedlow(x, x0, k, span):
     y = span * ( np.tanh(k*(x-x0)) + 1 )
#      y = span / (1 + np.exp(-k*(x-x0)))+mean
     return y

def guess_sigmoid(xdata,ydata):
    x0 = np.mean(xdata)
    k = (ydata[-1]-ydata[0]) / float(xdata[-1]-xdata[0])
    mean = np.mean(ydata)
    span = max(ydata)-min(ydata)
    return x0,k,span,mean

def relu(x,x0,k):
    y = np.maximum(k*(x-x0),0)
    return y
def guess_relu(xdata,ydata):
    x0 = np.mean(xdata) 
    k = (ydata[-1]-ydata[0]) / float(xdata[-1]-xdata[0])    
    return x0,k
def softplus(x):
    return np.log(1+np.exp(x))
def softrelu(x,x0,kx,ky):
    y = ky*softplus(kx*(x-x0))
    return y
def guess_softrelu(xdata,ydata):
    x0 = np.mean(xdata) 
    ky = (ydata[-1]-ydata[0]) / float(xdata[-1]-xdata[0])    
    kx = 1
    return x0,kx,ky

def linear_model(x,x0,k):
    y = k*(x-x0)
    return y

guess_dict ={sigmoid:guess_sigmoid,
            relu:guess_relu,
            softrelu: guess_softrelu,}

model2eqn = {sigmoid_fixedlow: lambda x0,k,span: r'{2:.2f}(1+\tanh [ {1:.2f} (x-{0:.2f} )])'.format(x0,k,span)}

def curve_fit_wrapper(model,xdata = None,ydata = None,plot = 1):#
    if xdata is None:
        xdata = np.array([0.0,   1.0,  3.0, 4.3, 7.0,   8.0,   8.5, 10.0, 12.0])
        ydata = np.array([0.01, 0.02, 0.04, 0.11, 0.43,  0.7, 0.89, 0.95, 0.99])
    # xdata = coord
    # ydata = tofit
    guessor = guess_dict.get(model,lambda x,y: [1]*(model.func_code.co_argcount - 1) )
    popt, pcov = curve_fit(model, xdata, ydata, p0 = guessor(xdata,ydata) )
    fitted = model(xdata,*popt)
    if plot:
        pass
    ##### Avoid importing matplotlib here
#         x = np.linspace(min(xdata), max(xdata), 50)
#         y = model(x, *popt)
#         plt.plot(xdata, ydata, 'o', label='data')
#         plt.plot(x,y, label='fit')
#         # pylab.ylim(0, 1.05)
#         plt.legend(loc='best')
#         plt.show()
    return popt,pcov,fitted



###### Formatting a polynomial for printing ####
def eqn_lm(fit):
    return 'y=%.3fx+%.3f'%(tuple(fit.tolist()))


def term(c, n):
    fmt = [
#         [ "", "", "" ],
#         [ "{c:+g}", "{sign:s}x", "{sign:s}x^{n:g}" ],
        [ "{c:+g}", "{c:+g}x", "{c:+g}x^{n:g}" ],
        [ "{c:+g}", "{sign:s}x", "{sign:s}x^{n:g}" ],
        [ "{c:+g}", "{c:+g}x", "{c:+g}x^{n:g}" ],
    ]
    return fmt[cmp( abs(c), 1)+1][cmp(n,1)+1].format(sign="- +"[cmp(c,0)+1], c=c, n=n)

def poly( xxs,suppsig = 1,decimal =None):
    if decimal:
        xxs = np.round(xxs, decimal)
    s = "".join(term(xxs[i],len(xxs)-i-1) for i in xrange(len(xxs)))
    if suppsig:
        s = suppsign(s)
    return s

def suppsign(s):
    if s[0] == '-':
        return s
    return s[1:]


######## Utilities for signal/dynamical systems
def isPeriodic(sol,radius = 10):
    '''
    Use nearest neighbor to determine whether trajectory matrix of shape (T,N) is periodic
    radius: topological continuous cutoff. 
    Method: Periodicity is claimed if anything outside radius is the nearest to the final point
    return: 
        -1 if not periodic
        i if periodic where i is the index of nearest neighbor
    '''
    md  = np.min(spdist.cdist(sol[-1:,:],sol[-radius:-1:]))
    Dout = spdist.cdist(sol[-1:,:],sol[:-radius,:])
    omdidx = np.argmin(Dout)
    omd = Dout.flat[omdidx]
    if omd<=md:
        return omdidx
    else: 
        return -1
    
##### Utilities to detecting periodic behavior
def positive_avgCrossing(x,):
    '''
    Find out positions where signal crosses its average level
    '''
    der1 = np.diff(x,axis = 0)
    xMavg = x - np.mean(x,axis = 0)
    avgX= np.where(xMavg[1:] * xMavg[:-1] <0 )[0]
    return avgX[ der1[avgX] > 0 ] 

def detect_period(x,t,trunc = 3,debug=0,raw = 0):
    '''
    Detect periodicity of a given signal "x"
    Return mean period identified as defined by where 
        1. x(t) - E(x(t)) changes sign
        2. x'(t)>=0
        (Implemented in positive_avgCrossing)
    '''
    peridx  = positive_avgCrossing(x)
    perdur = np.diff(t[peridx])
    N = len(perdur)
    if N > trunc:
        perdur = perdur[trunc:]
    else:
        if debug:
            print "N is too small:%d, setting N=0"%N
        return np.inf,0
    if raw:
        return perdur
    STD = perdur.std()
    MEAN= perdur.mean()
    stderr = STD/MEAN
    if debug:
        print N,MEAN,STD
    if stderr > 0.01:
        if debug:
            print "realative error is too big:%.3f%% "% (stderr*100)
#     print MEAN
    return MEAN,N


def optimiser_gd(W,gradF, lr = 0.1, alpha = 0.0, 
                tol = 0.001,
                maxStep = 1000,
                callback = None,
                meta = 0):
    W = W.copy()
    lst = []
    for i in range(maxStep):
        grad = gradF(W)
        if callback is not None:
            lst.append(callback(W,grad))
        delta = lr*(grad - alpha * W)
        if abs(delta).mean() < tol:
            break        
        W += delta
#         W = W + lr*grad
    if i+1==maxStep:
        print "[WARN]: Gradient descent not converged"
    if meta:
        lst = {'callback':lst,'step':i}
    return (W,lst)


#### Functional Programming Utilities

#from util import *

mse  = lambda t,y: np.mean(np.square(t-y))
mae  = lambda t,y: np.mean(np.abs(t-y))

def f_2d(f,transpose = 0):
    if transpose:
        return lambda x,y:f(np.vstack((x,y)).T)
    else:       
        return lambda x,y:f((x,y))



# def compositeF(lst):
#     return reduce(lambda f,g: lambda x:g(f(x)),lst,lambda x:x)
def transform_linear(x,W = None,b = None):
    if W is not None:
        x = np.dot(x,W)
    if b is not None:
        x = np.add(x,b)
    return x
def combine_args(f):
    return lambda x: f(*x)

def map_ph(x):
    return x





import itertools
def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return itertools.izip_longest(*args, fillvalue=fillvalue)


def vectorize_lazy(f,**kwargs):
    if isinstance(f,np.lib.function_base.vectorize):
        f = f
    else:
        f = np.vectorize(f,**kwargs)
    return f
np.vectorize_lazy = vectorize_lazy





if __name__ == '__main__':
    def f(IN):
        res = 'test'
        return res
    os.system('rm -rf test/*')
    os.system('mkdir -p test')
    p = mp.Pool(4)
    res,p = MapWithCache(f,[None]*10,ALI='test/test')
    # print res.get(1000000)
    p.close()

def flatten(gd):
    return np.concatenate([x.flat for x in gd])

import pymisca.ext
# .hasIPD
hasIPD = pymisca.ext.hasIPD
ipd = getattr(pymisca.ext,'ipd',None)
sys.stderr.write('is in ipython: %s \n'%hasIPD)

    

def ppJson(d):
    '''
    Pretty print a dictionary
    '''
    s = json.dumps(d,indent=4, sort_keys=True)
    return s

def invert_interp(F):
    '''
    Invert a functools.partial-made interpolator
    '''
    Fi = functools.partial(np.interp,
                      xp=F.keywords['fp'],
                      fp=F.keywords['xp'],                     
                     )
    return Fi

import scipy.interpolate as spinter

def interp_bytime(rnac,ts = None, n=100, **kwargs):
    '''Average the data over time assuming it is a time-series.
    Columns should be numerics in hours
'''
#     rnac = rnaseq_wk2sd.relabel('ZTime_int')
#     interpF = make_ZTime_interpolater(rnac,ts = ts, n=n, **kwargs)
    if ts is None:
        ts = rnac.columns
    tss = np.linspace(0,24,n)
    fs = rnac.values
    ts = np.hstack([ts-24., ts ,ts + 24.])
    fs = np.hstack([fs,fs,fs])

    interpF = spinter.interp1d(ts, fs)
    vals= interpF(tss)
    if hasattr(rnac,'setDF'):
        res=  rnac.setDF(vals)
    else:
        res = pd.DataFrame(vals, index=rnac.index, columns = tss )
    return res

def mat2str(mat,decimal=None,sep='\t'):
    if not isinstance(mat,np.ndarray):
        mat = np.array(mat)
    if not decimal is None:
        mat = np.round(mat,decimal)
    mat = mat.astype('str')
    lst = mat.tolist()
    
    if mat.ndim == 1:
        lst = [lst]
    elif mat.ndim == 2:
        pass
    s = '\n'.join(sep.join(x) for x in lst )
    return s

def cov2cor(COV):
    D = np.diag(COV)
    COR = COV /  np.sqrt(D[:,None]*D[None,:])
    return COR




###### Curve fitting
def model_exp(x, x0, bt, c):
    return c*np.exp(x/(-bt)) + x0
#     return c*np.exp(-bt*x) + x0

# ys = arr[0]
def fit_exp(ys,xs = None):
    if xs is None:
        xs=  np.arange(ys.size)
    popt,pcov,yfit = curve_fit_wrapper(model_exp, xs, ys,plot=0)
    loss = np.mean((yfit-ys)**2)
    return popt,yfit,loss

def escalIndex(N,i):
    '''
Pull "i" to the front of a list of N.
'''
    l = list(range(N))
    l.insert(0,l.pop(i))
    return l

def flat2dict(lst):
    '''
    Group duplicate objects 
    '''
    d = {}
    for i,val in enumerate(lst):
        d[val] = d.get(val,[]) + [i]
    return d
def DictListInvert(d):
    '''
DictList = {
a:[1,3],
b:[2,0],
}
out = {
0:a,
1:a,
2:b,
3:a,    
}
'''
    res = {}
    for k,v in d.items():
        dd = {x:k for x in v}
        res.update(dd)
    return res


def appendIndex(lst):
    '''
    Add index to duplicated objects
    '''
    d  = flat2dict(lst)
    out = len(lst)* [None]
    for k,pos in d.items():
        vals = ['%s%d'%(k,i) for i in range(len(pos))]
        for i,v in zip(pos,vals):
            out[i]=v
    return out
if __name__=='__main__':
    IN = ['a','b','a','a','b']
    print IN
    print addIndex(IN)


def expand(rg,rd=None):
    '''
    Expand an interval
    '''
    if rd is None:
        rd = np.diff(rg)/5.
    return (rg[0]-rd,rg[-1]+rd)

def dirname(infile,absolute=0):
    ''' get dirname from pathname
    [TBC] don't distinguish file from directory
'''
    if '/' in infile:
        DIR = infile.rsplit('/',1)[0]
    else:
        DIR = '.'    
    if absolute:
        assert 0,'not implemented'
    return DIR


def fname2mdpic(fname):
    '''Generate a markdown picture referr using a filename
    '''
    s = '![\label{fig:%s}](%s)'%(pyutil.getBname(fname),fname)
    return s 

def showsavefig(fname='test.png',fig=None,show=1,**kwargs):
#     print fname2mdpic(fname)
    fig = plt.gcf() if fig is None else fig
    fig.savefig(fname,**kwargs)    
    if show:
        plt.show(fig)
    else:
        plt.close(fig)
    md = fname2mdpic(fname); print(md)
    if hasIPD:
        ipd.display(ipd.Markdown(md[1:]))
    return md

def query2flat(q):
    qnorm = re.sub('[\" ]','',q).replace('==','=').replace('&','_')
    return qnorm

# def savePrint(fname,fig = None):
#     s = showsavefig(fname=fname,fig = fig); print s
#     ipd.display(pyutil.ipd.Markdown(s[1:]))
#     return s


    
def mat2str(mat,decimal=None,sep='\t',linesep='\n'):
    ''' Converting a numpy array to formatted string
    '''
    if not isinstance(mat,np.ndarray):
        mat = np.array(mat)
    if not decimal is None:
        mat = np.round(mat,decimal)
    mat = mat.astype('str')
    lst = mat.tolist()
    
    if mat.ndim == 1:
        lst = [lst]
    elif mat.ndim == 2:
        pass
    s = linesep.join(sep.join(x) for x in lst )
    return s 
def mat2latex(mat):
    '''
    Converting a numpy array to latex array
    '''
    s = mat2str(mat,sep='&',linesep='\\\\')
    s = wrap_env(s,env='pmatrix')
    return s.replace('\n','') 

def wrap_env(s,env=None):
    if env is None:
        return s
    if len(env)==0:
        return s
    return '\\begin{{{env}}} \n {s} \n \\end{{{env}}}'.format(s=s,env=env)

def wrap_math(s):
    return '$%s$'%s

def wrap_table(tab,caption = '',pos = 'h'):
    fmt='''\\begin{{table}}[{pos}]
    {tab}
    \\caption{{ {cap} }}
    \\end{{table}}
    '''
    s = fmt.format(pos=pos,cap = caption,tab = tab)
    return s

def df2latex(df,fname=None,caption=None,label=None,hsep=r'\\\hline',
             longtable=0,**kwargs):
    '''Convert a Data Frame to a latex string
'''
    if longtable:
        raise Exception('Not implemented')
    if label is None:
        if fname is not None:
            label = pyutil.basename(fname)
    if df.index.name is None:
        kwargs['index'] = False
    res = df.to_latex(buf=None,longtable=longtable,
                      **kwargs)
    caption = '' if caption is None else caption
    fmt = r'''
\begin{table}[h]
\begin{center}
\caption{\label{tab:%s}
%s } 
%s
\end{center}
\end{table}
'''.strip()
    res = fmt % (label,caption,res,)
    res = res.replace(r'\\',hsep)
    if fname is None:
        return res    
    else:
        with open(fname,'w') as f:
            f.write(res)
        s = r'\input{%s}'%fname
        return s


class RedirectStdStreams(object):
    '''Source:https://stackoverflow.com/a/6796752/8083313
Example:
    with RedirectStdStreams(stdout=devnull, stderr=devnull):
        print("You'll never see me")
    '''
    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr
    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr
    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        


def reindexZero(idx):
    ''' Mapped a index to an consecutive integer sequence starting from 0
    '''
    if isinstance(idx,pd.Series):
        idx = idx.values        
    uniq = np.unique(idx)
    mapper = { oi:ni for ni,oi in enumerate(uniq)}
    newidx = np.vectorize(mapper.get)(idx)
    return newidx, uniq        

#### R-like utils
import pandas as pd
#### Slower implementation
# def paste0(ss,sep=None,na_rep=None,):
#     '''Analogy to R paste0
#     '''
#     ss = [pd.Series(s) for s in ss]
#     ss = [s.astype(str) for s in ss]
#     s = ss[0]
#     res = s.str.cat(ss[1:],sep=sep,na_rep=na_rep)
#     return res
# pasteA=paste0

def get_cluCount(clu):
    ''' Count occurences of elements
'''
    cluCount = clu.groupby('clu').apply(len).reset_index()
    cluCount.columns = ['clu','count']
    return cluCount

def getConfusionMat(y_pred,y_true,):
#     pred_targ = mdl.predict(test_data)
    dfc = pd.DataFrame(dict(pred=y_pred, ground=y_true))
    dfc['num'] = 1
    confusion = dfc.pivot_table(index='ground',columns='pred',values='num',aggfunc='sum').fillna(0.)
    return confusion

def init_DF(C,rowName= None,colName=None):
    '''Conveniently initialise a pandas.DataFrame from a matrix "C"
'''
    df = pd.DataFrame(C)
    if rowName is not None:
        df.set_index(rowName,inplace=True)
    if colName is not None:
        df.columns = colName
    return df

def colGroupMean(dfc,axis=1,level=None):
    '''Group by level 0 index and take average over rows
'''
    gp = dfc.groupby(axis=axis,level=0,sort=False)
    dfc = gp.apply(lambda x:x.mean(axis=axis))
    return dfc

def colGroupStd(dfc,axis=1,level=None):
    '''Group by level 0 index and take average over rows
'''
    gp = dfc.T.groupby(level=0,sort=False)
    dfc = gp.apply(lambda x:x.std(axis=axis,level=level)).reset_index(level=0,drop=1)
    return dfc



# def paste0(ss,sep=None,na_rep=None,castF=unicode):
#     '''Analogy to R paste0
#     '''
#     if sep is None:
#         sep=u''
    
#     L = max([len(e) for e in ss])
#     it = itertools.izip(*[itertools.cycle(e) for e in ss])
#     res = [castF(sep).join(castF(s.decode('utf8')) for s in next(it) ) for i in range(L)]
#     res = pd.Series(res)
#     return res
# pasteB = paste0



def df__pad(df,prefix='val',suffix='',sep='_'):
    df = df.copy()
    lst= [] 
    if prefix != '':
        lst.append([prefix])
    lst.append("index")
    if suffix != '':
        lst.append([suffix])
        
    cols = pyext.df__paste0(df.T, lst,sep=sep)
    df.columns = cols
    return df


import operator

def df__makeContrast(dfc,
                     contrast=None,
                     treatment=None,
                     control=None,
                     func = operator.sub ):
    '''Use the supplied control/treatment to make contrasted values
'''
    if treatment is None or control is None:
        assert contrast is not None,"Must specify contrast dataframe\
        when any other variable is missing"
        treatment = contrast.treatment
        control = contrast.control
    vtreat = dfc.reindex(columns = treatment)
    vcontrol = dfc.reindex(columns = control)    
    vdiff = func(vtreat.values,vcontrol.values) 
    columns = pyext.paste0([treatment, control],'__').tolist()
#     columns = pyutil.columns__makeContrast(treatment=treatment,
#                                            control=control)
    dfdiff = pd.DataFrame(vdiff,
                          index=dfc.index,
                          columns=columns)
    return dfdiff

def meta__makeContrast(dfc,func = None,sep= '__',**kwargs):
    if func is None:
        func=np.vectorize(lambda t,c: sep.join([t,c]) if t!=c else t )
    res = df__makeContrast(dfc,func=func,**kwargs)
    return res

def df__makeContrastWithMeta( 
    self, 
   buf = None,
   contrast = None,
   control = None,
   treatment = None,
   colMeta = None,
   name = None,
   **kwargs):
    if buf is not None:
        contrast = pyext.read__buffer(buf.strip(),ext='csv',header=None).T
    elif contrast is None:
        assert control is not None
        contrast = pd.DataFrame(dict(control = list(control),
                                     treatment=list(treatment),
                                    ))
    else:
        pass
#     contrast = pyutil.read__buffer(buf,ext='csv',header=None).T
    dfc = pyutil.df__makeContrast( self, 
                                  contrast=contrast,
                                 )
    if colMeta is None:
        colMeta = self.colMeta
    dfcm = pyutil.meta__makeContrast( colMeta.T,
                                      contrast = contrast,
                                       sep='/').T

    _class = self.__class__
#     print (_class, _class.__name__=='countMatrix')
    dbg = _class.__name__.endswith('countMatrix')
#     print ('[dbg]%s' %dbg)
#     if _class.__name__.endswith('countMatrix'):
    if _class.__name__==('countMatrix'):
        #### assume scount.countMatrix
        dfc = _class(dfc,
                     colMeta=dfcm,
                     name=name,
                     **kwargs
                    )
#         print (type(dfc))
        return dfc
    else:
        return dfc,colMeta

def template__checkKeys(template,keys,force=0):
    keys = set(keys)
    kwSet = set(pyutil.ptn.templateKW.findall(template))
    if force:
        return kwSet.issubset(keys)
    else:
        assert kwSet.issubset(keys),\
        'keys not found in dataframe: %s' %kwSet.difference(keys)
    
        return 1
    

def df__fillTemplate(dfc,template,force = 0):
    '''Fill a templated string with context dictionary contained in a DataFrame
'''
    kwSet = set(pyutil.ptn.templateKW.findall(template))
    if force:
        #### set Default
        for k in kwSet.difference(dfc.columns):
            dfc[k] = '{%s}'%k
        # dfcc[list(kwSet.difference(dfc.columns)]
    assert kwSet.issubset(dfc.columns),'keys not found in dataframe: %s' %kwSet.difference(dfc.columns)

    res = []
    for row in dfc.itertuples():
        s = template.format(**row.__dict__)
        res += [s]
        
    return pd.DataFrame({'softText':res},index=dfc.index)

def df2multiIndex(df):
    '''Source:https://stackoverflow.com/a/33143219/8083313
    '''
    multi=df.set_index(df.columns.tolist(), inplace=False).index
#     print multi
    return multi

def gQuery(gQuery,gRef,id_col='ID'):
    ''' Query a DataFrame with a set of ID's
    '''
    if not  isinstance(gQuery, pd.DataFrame):
        gQuery = pd.DataFrame({'ID':gQuery})
    else:
        gQuery = gQuery.rename(columns = {gQuery.keys()[0]:'ID'},)
    if 'index' in gRef:
        gRef = gRef.drop('index',1)
    gRef = gRef.rename(columns ={id_col:'ID'},)
    gRes = gRef.reset_index().merge(gQuery).set_index('index')        
    return gRes

def pd2md(df):
    ''' Source https://stackoverflow.com/a/33869154/8083313
    '''
    fmt = ['---' for i in range(len(df.columns))]
    df_fmt = pd.DataFrame([fmt], columns=df.columns)
    df_formatted = pd.concat([df_fmt, df])
    return df_formatted.to_csv(sep="|", index=False)

def columnFilter(df,exclude='_'):
    df = df.loc[:,df.columns.str.find(exclude) == -1]
    return df

def explode(df,tosplit, tokeep, sep = ','):
    '''split var1 by separator and keep var2
    Source: https://stackoverflow.com/a/28182629/8083313
    [Add to ]: pd.DataFrame
'''
    var1 = tosplit
    var2 = tokeep
    a = df
    a = a.dropna()
    b = pd.DataFrame(a[var1].str.split(sep).tolist(), index=a[var2])
    b = b.stack()
    b = b.reset_index()[[0, var2]] # var1 variable is currently labeled 0
    b.columns = [var1, var2] # renaming var1
    return b

# def df2flat(meta,exclude='.+?[_/]'):
#     ''' Convert a meta dataFrame into a serialised format
#     '''
#     mc = meta.columns
#     m0 = meta.iloc[0]
#     ex = ['{}{}'.format(*x) for x in zip(mc,m0)]
#     ptn = re.compile(exclude)
#     idx = [ (ptn.match(x) is None) for x in ex]
#     mcurr = meta.iloc[:,idx]
#     L = len(mcurr)
#     for c in mcurr.columns:
#         mcurr.loc[:,c] = paste0([[c]*L, mcurr[c]],sep='=')
#     res = (paste0(mcurr.values.T,sep='_'))
#     return res

def meta2name(meta,exclude='.+?[_/:]',
#               keys=['gtype','light','Age','ZTime'], 
             ):    
    ex = ['{}{}'.format(*x) for x in zip(meta.columns, meta.iloc[0] )]
    ptn = re.compile(exclude); idx = [ptn.match(x) is None for x in ex ] 
    mcurr = meta.loc[:,idx] 
    res = df2flat(mcurr)
    return res

def df2flat(df,keys=None):
    if keys is None:
        keys = df.columns
    lst = [ 
        pyext.paste0( 
        [ [ k ],df[k] ],
        '=') 
           for k in keys]
    res = pyext.paste0( lst,'_')        
    return res

def unpackFlat(s,seps=['_']):
    '''My first recursive function that unpack a string according to a
    sequence of separators
    '''
    if len(seps) == 0:
        return s,seps
    else:
        sep,seps = seps[0],seps[1:]
#             print sep,s,s.split(sep)
        s = [upk(x, seps)[0] for x in s.split(sep)]
        return s,seps
upk = unpackFlat
def packFlat(s,seps=['_']):
    if len(seps) == 0:
        return s,seps
    else:
        sep,seps = seps[0],seps[1:]
        s = sep.join([ str( pck(x,seps)[0] ) for x in s]) 
        return s,seps
pck = packFlat
        
def flat2meta(ss,seps= ['_','=']):
    ''' Unpack a string produced by meta2flat() or df2flat()
    '''
    def f(x):
        return unpackFlat(x,seps)[0]
    res = map(f,ss)
    return res
def meta2flat(ss,seps= ['_','=']):
    ''' Unpack a string produced by meta2flat() or df2flat()
    '''
    def f(x):
        return packFlat(x,seps)[0]
    res = map(f,ss)
    return res


def metaContrast(mRef,mObs):
    ''' Merge two flattened meta descriptor 
    '''
    mSeq = zip(*map(flat2meta,[mRef,mObs]))
    def mMerge( (ref,obs) ):
        def f((rr,oo)):
            assert rr[0]==oo[0]
            if rr[1]==oo[1]:
                return rr
            else:
                return [rr[0],'%s-%s'%(oo[1],rr[1])]
        seq = zip(ref,obs)
        oseq = map(f,seq)
        return oseq
    mSeq = map(mMerge,mSeq)
    mFlat = meta2flat(mSeq)
    return mFlat


    
    
def file_ncol(fname,silent=1,sep=None):
    '''Get number of columns of a file
'''    
    if sep is None:
        sep = pyext.guess_sep(fname)
    f = open(fname)
    line = next(f);f.close()
    ncol = len(line.split(sep)) 
    return ncol

def read__remote( url,reader=None,):
    f = urllib.urlopen(url)
    return f
#     f = io.BytesIO(urllib.urlopen(url).read()) as f:
#         if reader is None:
#             res = f
#         else:
#             res = reader(f)
#     return res      


def read_json(fname):
    with open(fname,'r') as f:
        res = json.load(f)
    return res






# def jsonLoad(f,object_pairs_hook=collections.OrderedDict,**kwargs):
#     return json.load(f,object_pairs_hook=object_pairs_hook,**kwargs)
# def jsonLoads(f,object_pairs_hook=collections.OrderedDict,**kwargs):
#     return json.loads(f,object_pairs_hook=object_pairs_hook,**kwargs)


    
def readData_multiple(fnames, axis=0, NCORE=1, 
                      addFname = 1, guess_index=0, **kwargs):
    '''
    Convenient function to bulky apply readData()
'''
    worker = functools.partial(pyext.readData,
                               addFname=addFname, guess_index=guess_index,
                               **kwargs)
    dfs = mp_map(worker,fnames,n_cpu = NCORE)
    if axis is not None:
        dfs = pd.concat(dfs,axis=axis)
    return dfs

def sanitise_query(query):
    query = query.replace('>','-GT-')
    query = query.replace('<','-LT-')
    query = query.replace('==','-EQ-')
    query = query.replace('.','dot')
    query = query.replace(' ','')
    query = re.sub('[@\(\)\[\]]','-',query)
    return query

def queryCopy(infile,query, reader=None,inplace=False,**kwargs):
    '''query a dataset and save as a new copy
'''
    if reader is None:
        reader = pyext.readData
    querySans = sanitise_query(query)
    DIR = dirname(infile)
    
    base = pyext.getBname(infile)
    base += '_query=%s'%querySans
    ofile = base + '.tsv'
    
    if inplace:
        ofile = os.path.join(DIR,ofile)

    df = reader(infile)
    df = pd.DataFrame(df)

    df = df.query(query)
    df.to_csv(ofile, sep='\t',index=False,header = None)
    return ofile




def propose_mergeDB(mcurr,dataFile,force=0):
    '''
    propose merging DataFrame "mcurr" to an existing database "dataFile"
'''
    tempFile = dataFile+'.tmp'
    meta = pyext.readData(dataFile,)
#     meta.index.name='DataAcc'
    DUP = mcurr.index.isin(meta.index)
    print DUP
#     meta.drop(index=mcurr.index[DUP],inplace=True)
    assert mcurr.index.is_unique
    if not force:
        assert not (DUP.any())
    mc = mcurr.loc[~DUP]
    meta = pd.concat([meta,mc],sort=False)
    meta.index.name='DataAcc'
    meta.ZTime_int = meta.ZTime_int.fillna(-100).astype(int)
#     assert meta.index.is_unique
    
    meta.to_csv(tempFile, sep='\t',index=True)
    return (dataFile,tempFile)

def routine_combineData(fnames,ext=None,addFname = 0):
    dfs = map(lambda x: pyext.readData(x,ext=ext,addFname = addFname), fnames)
    idx = reduce(pyutil.functools.partial( 
        pyext.mergeByIndex, 
        as_index=1,how= 'outer'),
                 dfs)
    dfs = [df.reindex(idx.index) for df in dfs]
    return dfs
#### Bash-like utils
def head(lst, n ):
    if not isinstance(n,int):
        n = int(len(lst)*n)
    if n >= 0:
        res = lst[:n]
    if n < 0:
        res = lst[:n]
    return res
def tail(lst, n ):
    if not isinstance(n,int):
        n = int(len(lst)*n)
    if n > 0:
        res = lst[-n:]
    if n < 0:
        res = lst[-n:]
    elif n==0:
        res = lst[:0]
        
    return res

def keep(lst, n ):
    '''
    Keep until specified elements in the list and drop the rest 
    '''
    if n >= 0:
        res = lst[:n]
    if n < 0:
        res = lst[n:]
    return res

def drop(lst, n):
    '''
    Drop until specified elements in the list and keep the rest 
    '''
    if n >= 0:
        res = lst[n:]
    if n < 0:
        res = lst[:n]
    return res
l = list(range(10))

def test(f):
    print f.func_name
    print f(l,1)
    print f(l,-1)
    print f(l,0)

if __name__=='__main__':
    test(keep)
    test(drop)
    test(head)
    test(tail)






from pymisca.linalg import *
# from numpy_extra import np
from pymisca.numpy_extra import np,logsumexp,span
pyutil.span = np.span
from pymisca.mc_integral import *


from pymisca.pandas_extra import *
from pymisca.models import *
# pd,reset_columns,melt

def columns__makeContrast(contrast = None,treatment=None,control=None):
    if treatment is None or control is None:
        assert contrast is not None,"Must specify contrast dataframe\
        when any other variable is missing"
        treatment = contrast.treatment
        control = contrast.control
    cols = pyext.paste0([['TREAT_'],treatment,['__CONTROL_'],control])
    return cols


import textwrap
def formatName(name,maxLine=8):
    '''replace '_' by '\n'
'''
    name = str(name); lst = name.split('_')
    lst = textwrap.wrap(lst[0],20) if len(lst) <= 1 else lst
#     maxLine = 8; 
    trackName = '\n'.join(lst[:maxLine])        
    return trackName


def d_kldiv(X,Y):
    '''Estimate KL-divergence between discrete distribs
'''
    eps = 1E-8
    lX = np.log(X+eps)
    lY = np.log(Y+eps)
    C = X * ( lX - lY)
    return np.sum(C)

def d_jsdiv(X, Y, log = 0, base = 2):
    '''Estimate JS-divergence between discrete distribs
'''
    M = (X+Y)/2.
    d = (d_kldiv(X,M) + d_kldiv(Y,M))/2. 
    d = d/ np.log(base)
#     dsq = 1 - np.sum( np.sqrt (X*Y))
#     d = np.sqrt(dsq)
    return d

def d_hellinger(X, Y, log = 0):
    '''Estimate hellinger distance between discrete distribs
'''
    dsq = 1 - np.sum( np.sqrt (X*Y))
    d = np.sqrt(dsq)
    return dsq

def pdist_worker((i,j),metric=None,X=None):
    res = metric(i,j)
#     res = metric(X[i], X[j])
    return res

def pdist(X,metric,squareform=1,NCORE=1):
    '''Paralellised variant of scipy.spatial.distance
'''
#     X = np.asarray(X,order='c',)
#     m, n = np.shape(X)
    m = len(X)
    print m,
    ndm = (m * (m - 1))//2  ### length of flat distance vec
    k = 0
    it = [None]*ndm
    for i in xrange(0, m - 1):
        for j in xrange(i + 1, m):
#             it.append((k,(i,j)))    
#             it[k] = (i,j)
            it[k] = (X[i],X[j])
            k += 1
    worker = pyutil.functools.partial(pdist_worker,metric=metric,X=X)
    dm = pyutil.mp_map(worker,it,n_cpu=NCORE,
#                        kwds={'X':X,'metric':metric}
                      )
    it = [(x,x) for x in X]
    dg = pyutil.mp_map(worker,it,n_cpu=NCORE,
#                        kwds={'X':X,'metric':metric}
                      )    
    if squareform:
        D = spdist.squareform(dm)
        np.fill_diagonal(D,dg)
        res = D
    else:
        res = (dm,dg)
    return res

def msq(x,y=None,axis=None,keepdims=0):
    if y is None:
        y = 0.
    res = np.mean(np.square(x-y),axis =axis,keepdims = keepdims)
    return res
                  
# pyutil.pdist_worker = pdist_worker
# pyutil.pdist = pdist
# reload(pyutil)
def distPseudoInfo(pA,pB,
    logIN = 0,
    xlab=None,ylab=None,maxLine=4,vlim = [-2,2],
    silent=1,short=1,
):

    if not logIN :
        lpA = np.log(pA)
        lpB = np.log(pB)
    else:
        lpA = pA
        lpB = pB

    logC = pyutil.proba_crosstab(lpA,lpB) #### estimate joint distribution of labels
    margs =pyutil.get_marginal(logC) #### calculate marginal
    entC = pyutil.wipeMarg(logC,margs =margs)      #### wipe marginals from jointDist

#     MI = pyutil.entExpect(logC)
    # MI = np.sum(np.exp(logC)*entC)
    H1 = -pyutil.entExpect(margs[0])
    H2 = -pyutil.entExpect(margs[1])
    Hj = -pyutil.entExpect(logC.ravel())
    MI = H1+ H2 - Hj
#     Hj = H1 + H2 - MI

    if not silent:
        print 'MI=',MI
        print 'H1=',H1
        print 'H2=',H2
        fig,axs= plt.subplots(1,2,figsize=[14,4]);axs=axs.ravel()
        if resA is not None:
            xlab = resA.formatName(maxLine=maxLine) if xlab is None else xlab
        if resB is not None:
            ylab = resB.formatName(maxLine=maxLine) if ylab is None else ylab

        im = entC
        if CUTOFF is not None:
            xidx = np.where((np.exp(margs[0].ravel())*N)>CUTOFF)[0]
            yidx = np.where((np.exp(margs[1].ravel())*N)>CUTOFF)[0]
            im = im[xidx][:,yidx]

        pyvis.heatmap(logC,transpose=1,cname='log proba', ax=axs[0])
        pyvis.heatmap(im.T,
                      vlim=vlim,
                      cname='log likelihood ratio',
                      ax=axs[1],
                      xlab = xlab,
                      ylab = ylab,
                      ytick=yidx,
                      xtick=xidx)
    if short:
        if short =='MI':
            return MI
        
        return Hj
    else:
        return [H1,H2,Hj,MI], [entC,logC,margs]
    

def distNormJointH(it,NCORE=1,norm=1,avg='mean'):
    '''Calculate normalised joint entropy based distance
    d = 2*H(A,B) / (H(A,A) + H(B,B)) - 1
'''
    dm,dg = pyutil.pdist(it,metric=pyutil.distPseudoInfo,NCORE=NCORE,
                         squareform=0)
    D = spdist.squareform(dm)
    np.fill_diagonal(D,dg)
    
    Dx = np.array(dg)[:,None]
    D,Dx =  np.broadcast_arrays(D,Dx)
    Dy = Dx.T
    
    if avg == 'mean':
        Dn = (Dx+Dy)/2.
    elif avg =='harm':
        Dn = 2./(1/Dx + 1/Dy)
    Z = D - Dn
    if norm:
        Z = Z/Dn
    return Z

def getDim(model):
    if hasattr(model,'means_'):
        D = model.means_.shape[-1]
    if hasattr(model,'cluster_centers_'):
        D = model.cluster_centers_.shape[-1]
    return D
    
def predict_proba_safe(model,C, hard=0, W = None):
    Dc = np.shape(C)[-1]
    Dm = getDim(model)
    if Dc > Dm:
        if Dc == Dm +1 :
            Wn = meanNormBasis(Dc,orthonormal=1)
            C = np.dot(C,Wn.T)
    if hasattr(model,'predict_proba'):        
        clu = model.predict_proba(C)
        if hard:
            clu = (clu == np.max(clu,axis = 1,keepdims=1) ).astype('float')
            clu = clu/np.sum(clu,axis=1,keepdims=1)
    elif hasattr(model,'predict'):
        clu = model.predict(C)
        clu = pyutil.oneHot(clu)
    if W is not None:
        clu = clu*W
    return clu


# pyutil.distPseudoInfo = distPseudoInfo
def attrShape(self):
    '''map attr to their shapes
'''
    res = {k:np.shape(x) for k,x in self.__dict__.items()}
    return res

def proba2Onehot(A,log=1):
    '''Convert proba to one-hot vectors
'''
    mA = A.max(axis=1,keepdims=1)
    pA = (A == mA)
    if log:
        pA = np.log(pA)
    return pA



# def GitemGetter(val):
#     ''' Custom item getter similar to operator.itemgetter()
# '''
#     def itemGetter(data,val=val):
#         if isinstance(val,int):
#             res = data[val]
#         elif isinstance(val,slice):
#             ### only applicable for list-like data
#             res = data[val]
#         elif isinstance(val,pyutil.collections.Iterable):
#             val = list(val)
#             if isinstance(data,np.ndarray):
#                 res = data[val]
#             else:
#                 res = [data[i] for i in val]
# #                 res =np.array(
#         return res
#     return itemGetter

# ##### Numpy patches        
# from numpy import *
# def as_2d(*arys):
#     """
#     View inputs as arrays with exactly two dimensions.

#     Parameters
#     ----------
#     arys1, arys2, ... : array_like
#         One or more array-like sequences.  Non-array inputs are converted
#         to arrays.  Arrays that already have two or more dimensions are
#         preserved.

#     Returns
#     -------
#     res, res2, ... : ndarray
#         An array, or list of arrays, each with ``a.ndim >= 2``.
#         Copies are avoided where possible, and views with two or more
#         dimensions are returned.

#     See Also
#     --------
#     atleast_1d, atleast_3d

#     Examples
#     --------
#     >>> np.atleast_2d(3.0)
#     array([[ 3.]])

#     >>> x = np.arange(3.0)
#     >>> np.atleast_2d(x)
#     array([[ 0.,  1.,  2.]])
#     >>> np.atleast_2d(x).base is x
#     True

#     >>> np.atleast_2d(1, [1, 2], [[1, 2]])
#     [array([[1]]), array([[1, 2]]), array([[1, 2]])]

#     """
#     res = []
#     for ary in arys:
#         ary = asanyarray(ary)
#         if ary.ndim == 0:
#             result = ary.reshape(1, 1)
#         elif ary.ndim == 1:
#             result = ary[newaxis,:]
#         elif ary.ndim > 2:
#             result = ary.reshape(ary.shape[:1] + (-1,))            
#         else:
#             result = ary
#         res.append(result)
#     if len(res) == 1:
#         return res[0]
#     else:
#         return res
# np.as_2d = as_2d# import pymisca.util as pyutil
# flat = rnaseq.columns


# def flatSubset(flat,keep= None, as_list=0):
#     ''' Take a list of flat identifier and filter in only the
#     ones specified in the 'keep' list
# '''
#     nestList = flat2meta(flat)
#     if keep is None:
#         keep  =[ x[0] for x in nestList[0]]
# #     keep = ['ZTime']

#     for i,x in enumerate(nestList):
#         y = []
#         for k in x:
#             if k[0] in keep:
#                 y.append(k)
#         nestList[i] = y
#     res = nestList
#     if as_list:
#         pass
#     else:
#         res = meta2flat(res)
#     return res


# flatSubset(flat,keep = 'ZTime')# import pymisca.util as pyutil
# flat = rnaseq.columns
def flatSubset(flat,keep= None, as_list=0, negate=0):
    ''' Take a list of flat identifier and filter in only the
    ones specified in the 'keep' list
'''
    nestList = flat2meta(flat)
    if keep is None:
        keep  =[ x[0] for x in nestList[0]]
    if keep is str:
        keep = [keep]
#     keep = ['ZTime']

    for i,x in enumerate(nestList):
        y = []
        for k in x:
            if negate ^ (k[0] in keep):
                y.append(k)
        nestList[i] = y
    res = nestList
    if as_list:
        pass
    else:
        res = meta2flat(res)
    return res
# flatSubset(flat,keep = 'ZTime')
def TableToMat(fnames,ext='tsv',idCol ='Gene ID',valCol = 'TPM', match = 'Brad',callback = None):
    addFname=1
    df = pd.concat(pyutil.routine_combineData(fnames,ext='tsv',addFname = addFname))
    df = df.reset_index().pivot_table(values = valCol,index = idCol, columns='fname')
    C1 = scount.countMatrix.from_DataFrame(C1)
    if match is not None:
        C1 = C1.filterMatch(match)
    if callback is not None:
        C1 = callback(C1)
    return C1
# def TableToMat(fnames,ext='tsv',idCol ='Gene ID',valCol = 'TPM', match = 'Brad',callback = None):
#     addFname=1
#     df = pd.concat(pyutil.routine_combineData(fnames,ext='tsv',addFname = addFname))
#     df = df.reset_index().pivot_table(values = valCol,index = idCol, columns='fname')
#     C1 = scount.countMatrix.from_DataFrame(C1)
#     if match is not None:
#         C1 = C1.filterMatch(match)
#     if callback is not None:
#         C1 = callback(C1)
#     return C1

def col2meta(df=None,columns = None):
    if columns is None:
        lst = df.columns
    else:
        lst = columns
    meta = map(dict,pyutil.flat2meta(lst))
    meta = pd.DataFrame(meta)
    k = 'Unnamed: 0'
    if k in meta:
        meta.drop(k,1,inplace=True,)
    cols = pyutil.meta2name(meta)
    meta['header_']=cols
    meta['ZTime_int'] = meta['ZTime'].str.strip('ZT').astype(int)
    return meta

def argsortByRef(targ,ref):
    '''Calculating the sorting index to reorder into a reference sequence
'''
    assert all(x in ref for x in targ)
    skey = {k:i for i,k in enumerate(ref)}
    targ = list(targ)
    res = sorted(enumerate(targ),
                 key=lambda x: skey.get(x[1]))
    od  = [x[0] for x in res] 
#     res = [targ[x] for x in od] 
    return od

def TableToMat(fnames,ext='tsv',idCol ='Gene ID',valCol = 'TPM', match = 'Brad',callback = None):
    addFname=1
    df = pd.concat([ pyext.readData(fname=x,ext=ext,addFname=addFname) for x in fnames ],)
    df = df.pivot_table(values = valCol,index = df.index, columns='fname')
    od = argsortByRef(df.columns, fnames)
    df = df.iloc[:,od]
#     df = df[df.columns[od]]
    if match is not None:
        df = df.loc[df.index.str.match(match)]
    if callback is not None:
        df = callback(df)
    return df

### [DRPRECATED]
# def df2mapper(meta,key='fname_',val='header_'):
#     ''' Convert a dataFrame into a dictionary
# '''
#     if meta.index.name is not None:
#         meta = meta.reset_index()
#     for f in [key,val]:
#         assert f in meta
# #     if 'fname_' in meta:
#     mc = meta.set_index(key,drop=0)
# #     else:
# #         mc = meta
#     mapper = dict(zip(mc[key],mc[val]))
#     return mapper

def df__asMapper(dfc,key1,key2):
    res = dfc.eval('({key1},{key2})'.format(**locals()))
    res = dict(zip(*res))
    return res
df2mapper=  df__asMapper

def test_TableToMat():
    mcurr = meta.query('Age=="Wk2" & gtype== "Bd21"')
    mcurr = mcurr.query('light=="SD"')

    dfc = dfc.loc[dfc.index.str.match('Bradi')]
    df1 = dfc.reindex(columns=mcurr.header_)
    df2 = pyutil.TableToMat(mcurr.fname_,match='Bradi')

    # index = df2.index

    # df1 = df1.reindex(index)
    C1 = df1.values
    C2 = df2.values
    C = C1 - C2
    assert np.all(C==0)
    plt.scatter(C1[:1000].ravel(),C2[:1000].ravel(),3)


Table2Mat = TableToMat


def filterMatch(df,key,negate=0):

    MATCH = df.index.str.match(key)
#     print '[TYPE]',type(MATCH)
    df = df.loc[bool(negate) ^ MATCH]
    return df    




def mapAttr(lst,attr):
    return [getattr(x,attr) for x in lst]

def get_logP(df=None, mdl=None,X=None,axis=None):
    '''estimate categorial log-proba but allow arbitrary normalisation 
    by specifying axis.
    Assuming "mdl" is equipped with _estimate_weighted_log_prob() [BAD idea]
'''
    if df is not None:
        mdl, X = df.model, df.values
    M,C = mdl.means_, mdl.covariances_
    logP = mdl._estimate_weighted_log_prob(X)
    cluPart = pyutil.logsumexp(logP, axis = axis)
    logP_cluNorm = logP - cluPart
    return logP_cluNorm

def proba_crosstab( A, B, as_entrop=0, selfNorm=1,logIN = 1):
    '''A in shape (n_sample,n_category_a)
    B in shape (n_sample,n_category_b)
    Assuming both are column-normalised log probability
'''
    if not logIN:
        A = np.log(A);
        B = np.log(B)

    if selfNorm:
        zA = logsumexp(A,axis=1,keepdims=1,log=0)
        zB = logsumexp(B,axis=1,keepdims=1,log=0)
        zzA = zA.sum(); zzB = zB.sum()
        wA,wB = zA/zzA, zB/zzB  ### calculate weights
        w  = (wA + wB)/2.       ### combine weights
        A = A - np.log(zA); B = B - np.log(zB) ### convert into conditionals
        A[np.isnan(A)] = -np.inf
        B[np.isnan(B)] = -np.inf
#         print w.mean(),A.mean(),B.mean()
    else:
        w = [ [ 1./len(A)] ]*len(A) 
    lW = np.log( w) [:,:,None]
        
    logC = C0 =  ( 
        A[:,:,None] 
       + B[:,None] 
        +  lW
    )
    C = logsumexp(logC,axis=0,keepdims=0)
    
#     if not selfNorm:
#         C= C - np.log(len(A))
    
    
#     C = logsumexp(logC,axis=0,keepdims=0) - np.log(len(A))
    if as_entrop:
        C = wipeMarg(C,logIN=1)
    return C
def wipeMarg(C,logIN=1,margs=None):
    '''Subtract log-marginals
'''
    if not logIN:
        C = np.log(C)
    margs = get_marginal(C,logIN=1) if margs is None else margs
#     margs = 
    CC = reduce(np.subtract,[C]+margs)
    return CC
def get_marginal(C,logIN=1):
    '''calculate marginals
'''
    if not logIN:
        C = np.log(C)
    axes = range(C.ndim)
    lst = []
    for axre in range(C.ndim):
        axis = axes[:]; axis.pop(axre)
        marg = logsumexp(C,axis= tuple(axis),keepdims=1)
        lst +=[ marg]
    res = lst
    return res
def entExpect(C,logIN=1):
    '''E(np.exp(C)*C)
    convert C into log[p/(p-assumed-independency)] if C is multi-axis
'''
    if np.squeeze(C).ndim ==1:
        entC = C
    else:
        entC = wipeMarg(C)
    entC = entC.copy()
    entC[np.isneginf(entC)] = 0
    H = np.sum(np.exp(C)*entC)
    return H



# def logsumexp(X,axis=None,keepdims=1,log=1):
#     '''
#     log( 
#         sum(
#             exp(X)
#             )
#         )
# '''
#     xmax = np.max(X)
#     y = np.exp(X-xmax) 
#     S = y.sum(axis=axis,keepdims=keepdims)
#     if log:
#         S = np.log(S)  + xmax
#     else:
#         S = S*np.exp(xmax)
#     return S

# def dist2ppf(vals):
#     '''Estimate percentile locations from a sample from distribution
# '''
#     ### for some reason you need to argsort twice.....
#     idx = np.argsort(np.ravel(vals)).argsort()
# #     od = idx.argsort()
# #     idx = od
# #     idx = np.arange(len(vals))[od]
#     per = (idx+0.5)/(max(idx)+1)
#     if isinstance(vals,pd.Series):
#         per = pd.Series(per,index=vals.index, name=vals.name)
#     return per

def areaRatio(xs):
    '''
    Extract the tail information by observing the ratio bewtween the
 cumulative integral and the minimum.
    '''
    od = np.argsort(xs)[::-1]
    ood = np.argsort(od)
#     per_x = pyutil.dist2ppf(xs)
    MAX = max(xs)
    xmax = MAX -xs
    nseq = np.arange(len(xs))
    xavg=(xmax[od].cumsum()\
           / ( 1 + nseq))
#     xmed = xmax[od] [ nseq //2][ood]
#     res = xmed/xavg

    res = xavg / xmax[od]
    res[xmax[od]==0]=0.
    res = np.maximum.accumulate(res)
    res = res[ood]
    return res

# def oneHot(values,MAX=None):
#     values = np.ravel(values)
#     if MAX is None:
#         MAX  = np.amax(values)
#     n_values = MAX + 1
#     res = np.eye(n_values)[values]
#     return res



def random_covmat(D = 2):
    matrixSize = D
    A = np.random.rand(matrixSize,matrixSize) - 0.5
    B = np.dot(A,A.transpose())
    C = B
    return C

random_angle = lambda x: np.random.uniform(0,np.pi*2,size=x) - np.pi

def random_unitary(size=(1,3)):
    X = np.random.normal(0.,1.,size=size)
    X = X/ np.linalg.norm(X,axis=1,keepdims=1)
    return X


def detNorm(K):
    '''Linearly rescale so that det(K)=1
'''
    K = random_covmat()
    det = np.linalg.det(K)
    K = K / (det ** (1. / len(K)))
    return K
def detNorm(K):
    '''Linearly rescale so that det(K)=1
'''
#     K = random_covmat()
    det = np.linalg.det(K)
    K = K / (det ** (1. / len(K)))
    return K
def detNorm(K):
    '''Linearly rescale so that det(K)=1
'''
#     K = random_covmat()
    det = np.linalg.det(K)
    K = K / (det ** (1. / len(K)))
    return K
def name2dict(self,names):
    res = { name:getattr(self,name) for name in names}
    return res



def wrapTFmethod(tfunc,sess = None):
    '''Safely call tensorflow method as np method
'''
    def gfunc(x):
        if isinstance(x,np.ndarray):
            x = x.astype(np.float32)
        y = tfunc(x).eval(session = sess)
        return y
    return gfunc

# def arrayFunc2mgridFunc(arrayFunc):
#     def mgridFunc(*x):
#         ''' x = [xgrid,ygrid]
# '''
#     #     print (map(np.shape,x))
#         shape = x[0].shape 
#         X = np.vstack(map(np.ravel,x)).T ### compatible with TF
#         val = arrayFunc(X,)
#         val = np.reshape(val,shape,)
#         return val
#     mgridFunc.arrayFunc = arrayFunc
#     return mgridFunc

def firstByKey(fname=None,df=None,keys = ['feature_acc','FC'],guess_index=0,save=1,
              header = 0,**kwargs):
    if df is None:
        assert fname is not None
        df = pyext.readData(fname,guess_index=guess_index,header=header,**kwargs)
    dfc = df.sort_values(keys,ascending=False)
    dfcc = dfc.groupby(keys[0],sort=True).first()
#     dfcc.hist('FC')
    ofname = pyext.getBname(fname) + '_type=firstByKey' +'.tsv'
    if save:
        dfcc.to_csv(ofname, sep='\t')
        return ofname
    else:
        return dfcc

import sys
import urllib2
def string_goenrichment( buf =None,gids= None, species=None, ofname = None,
                       nMax=800):
    if gids is None:
        assert buf is not None
        gids = buf.strip().splitlines()
    string_api_url = "https://string-db.org/api"
    output_format = "tsv"
    method = "enrichment"
    my_app  = "www.newflaw.com"


    spec2id = {'ATHA':3702,
               'BDIS':15368,
               'DMEL':7227,
               'HSAP':9606,
               None:None,}

    specID = spec2id.get(species.upper(),None)
    assert specID is not None


    ## Construct the request

    request_url = string_api_url + "/" + output_format + "/" + method + "?"
    request_url += "identifiers=" + "%0d".join(gids[:nMax])
    request_url += "&" + "species=" + str(specID)
    request_url += "&" + "caller_identity=" + my_app

    ## Call STRING

    try:
        response = urllib2.urlopen(request_url)
    except urllib2.HTTPError as err:
        error_message = err.read()
        print(error_message)
        return None
#         sys.exit()

    ## Read and parse the results

    result = response.readline()

    if result:
        header = result.strip().split('\t')
    df = pyext.readData(response,ext='tsv',header=None,index_col = None,guess_index=0,columns = header)
#     df= pd.DataFrame.from_csv(response,sep='\t',header=None,index_col=None)
#     df.columns = header
    df.sort_values(['category','fdr'],inplace=True)
    
    if ofname is not None:
        df.to_csv(ofname)
        return ofname
    else:
        return df

import tempfile 

def make__tempDIR(DIR,silent=1, **kwargs):
    pyutil.shellexec('mkdir -p %s'%DIR,silent=silent)
    DIR= tempfile.mkdtemp(dir=DIR)
    return DIR

def saveFigDict(figs,DIR=None,exts=['svg'],silent=1,
               **kwargs):
    if DIR is None:
        DIR=pyutil.os.environ.get('HOME',None)
        assert DIR is not None, 'cannot get environment variable:$HOME'
        DIR = '%s/cache/plots' % (DIR)
        DIR = make__tempDIR(DIR)
    pyutil.shellexec('mkdir -p %s'%DIR,silent=silent)
    DIR = DIR.rstrip('/')
    ofnames = []
    for bname,fig in figs.items():
        noEXT = '%s/%s' % (DIR,bname)
        DDIR = pyutil.dirname(noEXT)
        if not pyutil.os.path.exists(DDIR):
            pyutil.os.makedirs(DDIR,)
#         pyutil.shellexec('mkdir -p %s'%DIR,silent=silent)
        
        for ext in exts:
            ofname = '%s.%s'%(noEXT,ext)           
            fig.savefig(ofname,
                        bbox_inches='tight',
                        **kwargs
                       )
            ofnames += [ofname]
    fignames = ofnames
    l = locals()
    return {x: l.get(x) for x in ['DIR','fignames']}

try:
    from jinja2_util import *
except Exception as e:
     sys.stderr.write('[WARN]%s\n'%e)
# from pymisca.vis_util import qc_index    
    