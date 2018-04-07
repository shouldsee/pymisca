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
