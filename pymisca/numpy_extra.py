##### Numpy patches        
import numpy as np
from numpy import *
import sys
pynp = sys.modules[__name__]

def as_2d(*arys):
    """
    View inputs as arrays with exactly two dimensions.

    Parameters
    ----------
    arys1, arys2, ... : array_like
        One or more array-like sequences.  Non-array inputs are converted
        to arrays.  Arrays that already have two or more dimensions are
        preserved.

    Returns
    -------
    res, res2, ... : ndarray
        An array, or list of arrays, each with ``a.ndim >= 2``.
        Copies are avoided where possible, and views with two or more
        dimensions are returned.

    See Also
    --------
    atleast_1d, atleast_3d

    Examples
    --------
    >>> np.atleast_2d(3.0)
    array([[ 3.]])

    >>> x = np.arange(3.0)
    >>> np.atleast_2d(x)
    array([[ 0.,  1.,  2.]])
    >>> np.atleast_2d(x).base is x
    True

    >>> np.atleast_2d(1, [1, 2], [[1, 2]])
    [array([[1]]), array([[1, 2]]), array([[1, 2]])]

    """
    res = []
    for ary in arys:
        ary = asanyarray(ary)
        if ary.ndim == 0:
            result = ary.reshape(1, 1)
        elif ary.ndim == 1:
            result = ary[newaxis,:]
        elif ary.ndim > 2:
            result = ary.reshape(ary.shape[:1] + (-1,))            
        else:
            result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res
np.as_2d = as_2d

def span(xs,p=100.):
    MIN = percentile(xs,100.-p)
    MAX = percentile(xs,p)
    return MIN,MAX
np.span = span

def logsumexp(X,axis=None,keepdims=1,log=1):
    '''
    log( 
        sum(
            exp(X)
            )
        )
'''
    xmax = np.max(X,axis=axis, keepdims=1)
    y = np.exp(X-xmax) 
    S = y.sum(axis=axis,keepdims=keepdims)
    if log:
        S = np.log(S)  + xmax
    else:
        S = S*np.exp(xmax)
    return S
np.logsumexp = logsumexp

def arr__l2norm(X,axis=None,keepdims=1):
    return np.sqrt((X**2).mean(axis=axis,keepdims=keepdims))
def arr__sumNorm(X,axis=None,keepdims=1):
    SUM = (X.sum(axis=axis,keepdims=keepdims)) 
    SUM[SUM==0.]= 1.
    X = X /SUM
    return X
def arr__rowNorm(X,axis=1):
    X  = arr__sumNorm(X,axis=axis)
    return X
def arr__colNorm(X,axis=0):
    X  = arr__sumNorm(X,axis=axis)
    return X

def distance__hellinger(resp1,resp2,check = True):
    if check:
        resp1 =  arr__rowNorm(resp1,axis=1)
        resp2 =  arr__rowNorm(resp2,axis=1)
    return 0.5 * np.mean((np.sqrt(resp1) - np.sqrt(resp2))**2)
def randMat(d,N=1):
    res = np.random.random(size=(N,d))
    res = arr__rowNorm(res)
    return res

def expect(X,func=None, weight=None,axis=None,keepdims=True):
    if func is not None:
        X = func(X)
    if weight is None:
        weight = np.ones(X.shape)
    while np.ndim(X)<np.ndim(weight):
#         if np.ndim(X)<np.ndim(weight):
        X= np.expand_dims(X,-1)
    X = X*weight
    res = np.sum(X,axis=axis,keepdims=keepdims) \
    / np.sum(weight,axis=axis,keepdims=keepdims)
    return res

def weightedVariance(X,weight=None,axis=1,keepdims=True):
    Ex2 = pynp.expect(X,func=np.square, weight=weight,axis=axis,keepdims=keepdims)
    Ex = pynp.expect(X,func=None, weight=weight,axis=axis,keepdims=keepdims)
    res = Ex2 - Ex**2
    return res
