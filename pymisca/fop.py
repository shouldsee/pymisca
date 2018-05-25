import numpy as np
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