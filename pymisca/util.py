import os,sys
import functools, copy,re
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from fop import *
from canonic import *

import re
import datetime
def datenow():
    res = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return res

def arg2dict(s):
#     s = '''a=1,b=2,c=3
#     '''
    lst = re.findall('([^ =]+)[ ]?=[ ]?([^ ,]+)[, ]',s)
    print lst
    dct = {k:eval(v) for k,v in lst}
    return dct

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


##### OOP utility

class util_obj(object):
    def __init__(self):
        pass
    
    def reset(h,method):
        mthd = getattr(h,method)
        if isinstance(mthd, functools.partial):
            setattr(h,method,mthd.func)
        else:
            print "[WARN]:Trying to reset a native method"
        pass

    def partial(h,attrN,**param):
        attr = getattr(h,attrN)
        newattr = functools.partial(attr,**param)
        setattr(h,attrN,newattr)
        pass
    def set_attr(h,**param):
        for k,v in param.items():
            setattr(h, k, v)
        return h

    
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
from util import *

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
import scipy.stats as spstat#
import scipy.spatial.distance as spd
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

from sklearn.metrics import mutual_info_score
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
from scipy.optimize import curve_fit

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
import scipy.spatial.distance as spdist
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


def repeatF(f,n):
    of = composeF([f]*n)
    return of
def composeF(lst):
    return reduce(lambda f,g: lambda x:g(f(x)),lst,lambda x:x)
compositeF = composeF
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



##### Multiprocessing map
import multiprocessing as mp
def mp_map(f,lst,n_cpu=1,**kwargs):
    if n_cpu > 1:
        p = mp.Pool(n_cpu)
        OUTPUT=p.map_async(f,lst).get(999999999999999999) ## appx. 7.6E11 years
#         OUTPUT = p.map(f,lst)
        p.close()
    else:
        OUTPUT = map(f,lst)
    return OUTPUT

def MapWithCache(f,it,ALI='Test',nCPU=1,force=0):
    print '[MSG] Mapping function:   Under cahce alias: %s'%(ALI)
    fmt = '%s_%%05d'%(ALI)
    def cb(resLst):
        for i,res in enumerate(resLst):
            fname = pyutil.canonic_npy([fmt%i])[0]
            np.save(fname,res)    
            print 'Saving to %s'%fname
        return resLst
    fname = pyutil.canonic_npy([fmt%0])[0]
    print fname
    if os.path.exists(fname) and not force:
        res = np.load(fname).tolist()
        p = None
    else:        
        p = mp.Pool(nCPU)
        
        res = p.map_async(f,it,callback=cb)
    return res,p

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

def is_ipython():
    try:
        get_ipython
        return 1
    except:
        return 0
print 'is in ipython:',is_ipython()


def printlines(lst):
    print '\n'.join(lst)
    
    
def invert_interp(F):
    '''
    Invert a functools.partial-made interpolator
    '''
    Fi = functools.partial(np.interp,
                      xp=F.keywords['fp'],
                      fp=F.keywords['xp'],                     
                     )
    return Fi

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


def flat2dict(lst):
    '''
    Group duplicate objects 
    '''
    d = {}
    for i,val in enumerate(lst):
        d[val] = d.get(val,[]) + [i]
    return d
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
def basename(fname):
    '''
    Extract "bar" from "/foo/bar.ext"
    '''
    bname = fname.split('/')[-1].rsplit('.',1)[0]
    return bname



def fname2mdpic(fname):
    '''Generate a markdown picture referr using a filename
    '''
    s = '![\label{fig:%s}](%s)'%(basename(fname),fname)
    return s 

def showsavefig(fname='test.png',fig=None,show=1,**kwargs):
#     print fname2mdpic(fname)
    fig = plt.gcf()
    fig.savefig(fname,**kwargs)    
    if show:
        plt.show(fig)
    return fname2mdpic(fname)
    
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

def paste0(ss,sep=None,na_rep=None):
    '''Analogy to R paste0
    '''
    if sep is None:
        sep=''
    res = [sep.join(str(s) for s in x) for x in zip(*ss)]
    res = pd.Series(res)
    return res
pasteB = paste0

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

def df2flat(meta,exclude='.+?[_/]'):
    ''' Convert a meta dataFrame into a serialised format
    '''
    mc = meta.columns
    m0 = meta.iloc[0]
    ex = ['{}{}'.format(*x) for x in zip(mc,m0)]
    ptn = re.compile(exclude)
    idx = [ (ptn.match(x) is None) for x in ex]
    mcurr = meta.iloc[:,idx]
    L = len(mcurr)
    for c in mcurr.columns:
        mcurr.loc[:,c] = paste0([[c]*L, mcurr[c]],sep='=')
    res = (paste0(mcurr.values.T,sep='_'))
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
        s = sep.join([pck(x,seps)[0] for x in s]) 
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


from numpy_extra import np

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
# np.as_2d = as_2d