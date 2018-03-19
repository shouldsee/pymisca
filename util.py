import functools, copy
import matplotlib.pyplot as plt
import numpy as np

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

model = sigmoid
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

from util import *

mse  = lambda t,y: np.mean(np.square(t-y))
mae  = lambda t,y: np.mean(np.abs(t-y))

def f_2d(f,transpose = 0):
    if transpose:
        return lambda x,y:f(np.vstack((x,y)).T)
    else:       
        return lambda x,y:f((x,y))


def repeatF(f,n):
    of = compositeF([f]*n)
    return of
def compositeF(lst):
    return reduce(lambda f,g: lambda x:g(f(x)),lst,lambda x:x)
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
        OUTPUT=p.map(f,lst)
        p.close()
    else:
        OUTPUT = map(f,lst)
    return OUTPUT

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