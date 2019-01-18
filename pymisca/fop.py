import numpy as np
import collections

def repeatF(f,n):
    of = composeF([f]*n)
    return of
def composeF(*lst):
    return reduce(lambda f,g: lambda *x:g(*f(*x)),lst, lambda *x:x)
compositeF = composeF

def l1_normF(f):
    g= lambda *args:np.sum(np.abs(f(*args)),axis=-1)
    return g
def l2_normF(f):
    g= lambda *args:np.sum(np.square(f(*args)),axis=-1)**0.5
    return g
def negF(f):
    g = lambda *args:-f(*args)
    return g
def addF(f,g):
    def h(*args):
        return f(*args) + g(*args)
    return h

def sumF(Fs):
    outF=  lambda *x: sum( (F(*x) for F in Fs) )
    return outF
def identity(*x):
    return x
def none(*x,**kwargs):
    return None

def make_gradF(f,eps=1E-4):
    def gradF(*x):
        f0 = f(*x)
        grad = [0]*len(x)
        for i,xi in enumerate(x):
            xcurr = list(x)[:]
            xcurr[i]= xi +eps
            df = f(*xcurr) - f0
            grad[i]= df/eps
        return grad
    return gradF

def alignF(f,g):
    '''
    Chain functions together but only change the first of inputed arguments
    '''
    h = lambda *args:g(*(f(*args),)+args[1:])
    return h


def linearF(IN,):
    '''
    return a Linear function:
        1. if IN is float, return a constant function
        2. if IN is list, return a linear combination using
        elements as coef.
    '''
    if isinstance(IN,float):
        F = lambda *x:IN
    if isinstance(IN,np.ndarray):
        IN = IN.tolist()
    if isinstance(IN,list):
        L = len(IN)
        F = lambda *x:np.dot(IN,np.ravel(x))
    return F

def deltaF(delta):
    '''Apply a finite deviation (delta) to a state vetor (sv)
    '''
    F = lambda sv: np.add(sv,delta)
    F.D = len(delta)
    return F

def safeMap(f,it):
    '''Perform single-evaluation for non-iterable
    '''
    try:
        it = iter(it)
        res = map(f,it)
#         if len(res)==1:
#             res = res[0]
    except TypeError:
        res = f(it)
    return res




def GitemGetter(val):
    ''' Custom item getter similar to operator.itemgetter()
'''
    def itemGetter(data,val=val):
        if isinstance(val,int):
            res = data[val]
        elif isinstance(val,slice):
            ### only applicable for list-like data
            res = data[val]
        elif isinstance(val, collections.Iterable):
            val = list(val)
            if isinstance(data,np.ndarray):
                res = data[val]
            else:
                res = [data[i] for i in val]
#                 res =np.array(
        return res
    return itemGetter


def arrayFunc2mgridFunc(arrayFunc):
    def mgridFunc(*x):
        ''' x = [xgrid,ygrid]
'''
    #     print (map(np.shape,x))
        shape = x[0].shape 
        X = np.vstack(map(np.ravel,x)).T ### compatible with TF
        val = arrayFunc(X,)
        val = np.reshape(val,shape,)
        return val
    mgridFunc.arrayFunc = arrayFunc
    return mgridFunc