# import pymisca.ext as pyext


import autograd.numpy as np
import autograd

from pymisca.iterative.base import getTraj
import pymisca.numpy_extra as pynp

# import numpy as np
# import pymisca.oop as pyop

# from pymisca.ext import entropise
# pynp.entropise = entropise


def dirichlet__minimise__grad(
    Y,
#     loss,
    X0=None,
    beta=None,
#     1.0,
    stepSize=1.0,
    eps=1E-10,
    **kwargs
):
    '''
    Author: Feng Geng
    X: 
    Y: 
    S0: 
    
    Minimise the hellinger distance between Y and X \cdot S
    minimise  E = 
    by iteratively setting 
    in a normalised fashion
    '''
#     dx,dy = S.shape
#     N,dy = Y.shape
#     size = (N,dx)
    Y = np.array(Y)
    Y = pynp.arr__rowNorm(Y)
    if beta is None:
        beta = Y.shape[1]
    alpha = 1./beta
    
    size = Y.shape
    
    #### re-alias
#     X0 
#     S_ = S
    def proj(X):
        X = np.exp(X)
        X = X / np.sum(X,axis=1,keepdims=1)
        return X
    def lossFunc(X):
        X = proj(X)
        res = (alpha) * np.log(X + eps)
        res = res * Y
        res = np.sum(res)
        res = -res
        return res
    
    gradFunc = autograd.grad(lossFunc)
#     stepSize = beta
    
    def step(X):
        '''
        Core iterative update
        '''
        grad = gradFunc(X)
        X = X - stepSize * grad
        return X
    # dx,dy=size
    if X0 is None:
        X0 = np.random.random(size=( size[0],size[1]))
#         S0 = randMat(N=dx,d=dy)
#     X0 = pynp.arr__rowNorm(X0)
        
    res = getTraj(step,
                  X0,
                  lossFunc=lossFunc,
                  **kwargs)
    
    res.last = proj(res.last)
    return res





def MCE__grad(
    Y,
#     loss,
    X0=None,
    beta=0.0,
#     1.0,
    stepSize=1.0,
    eps=1E-10,
    **kwargs
):
    '''
    Author: Feng Geng
    X: 
    Y: 
    S0: 
    
    minimise cross-entropy \sum_{ik} q_{ik} \log ( r_{ik} ) wrt q_{ik}
    
    Minimise the hellinger distance between Y and X \cdot S0
    minimise  E = 
    by gradient methods in a log-transformed parameter space

    in a normalised fashion
    '''
#     dx,dy = S.shape
#     N,dy = Y.shape
#     size = (N,dx)
    Y = np.array(Y)
    Y = pynp.arr__rowNorm(Y)
#     if beta is None:
#         beta = Y.shape[1]
#     alpha = 1./beta
    
    size = Y.shape
    
    #### re-alias
#     X0 
#     S_ = S
    def proj(X):
        X = np.exp(X)
        X = X / np.sum(X,axis=1,keepdims=1)
        return X
    
    def lossFunc(X):
        X = proj(X)
        #### beta = 0. reduce to standard EM
        #### beta = 1. corresponding to reduce_entropy = 1.0
        res = (X) * ( np.log(Y + eps) + (beta - 1. ) * np.log(X + eps) )
        res = np.sum(res)
        
        #### we are maximising the sum, hence minimising its negative
        res = -res 
        return res
    
    gradFunc = autograd.grad(lossFunc)
#     stepSize = beta
    
    def step(X):
        '''
        Core iterative update
        '''
        grad = gradFunc(X)
        X = X - stepSize * grad
        return X
    
    # dx,dy=size
    if X0 is None:
        X0 = np.random.random(size=( size[0],size[1]))
        
#         S0 = randMat(N=dx,d=dy)
#     X0 = pynp.arr__rowNorm(X0)
        
    res = getTraj(step,
                  X0,
                  lossFunc=lossFunc,
                  **kwargs)
    
    res.last = proj(res.last)
    
    return res 
