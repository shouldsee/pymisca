# import pymisca.ext as pyext

from pymisca.iterative.base import getTraj
import pymisca.numpy_extra as pynp

import numpy as np
import pymisca.oop as pyop

from pymisca.ext import entropise
pynp.entropise = entropise


def entropise_weight(
    Y,X0=None,
    beta=1.0,
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

    size = Y.shape
    
    #### re-alias
#     X0 
#     S_ = S
    
    
    def step(X):
        '''
        Core iterative update
        '''
#         R = np.sqrt(S)
#         print X.shape
        gamma = np.sqrt(X)
        A = gamma * ( 2 * np.log(gamma+eps) + 1.) * beta  + 1./gamma
        grad = Y * A
        X = pynp.arr__rowNorm(grad**2)
        return X
    
    def lossFunc(S):
#         S = S**2
        x = pynp.entropise(S,axis=1)
        H = x.sum(axis=1,keepdims=1)
        ll =  -H * beta + np.log(S+eps)
        ll = ll * Y
        ll = ll.mean()
#         print ll
        return ll
    
#     def lossFunc(S):
#         ll = pynp.distance__hellinger(Y, S.dot(S_))
#         return  ll
    # dx,dy=size
    if X0 is None:
        X0 = np.random.random(size=size)
#         S0 = randMat(N=dx,d=dy)
    X0 = pynp.arr__rowNorm(X0)
        
    res = getTraj(step,
                  X0,
                  lossFunc=lossFunc,
                  **kwargs)
    return res
main = entropise_weight



from pymisca.iterative.base import getTraj
import pymisca.numpy_extra as pynp

import autograd.numpy as np
import autograd

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
        X0 = np.random.random(size=(1,size[1]))
#         S0 = randMat(N=dx,d=dy)
#     X0 = pynp.arr__rowNorm(X0)
        
    res = getTraj(step,
                  X0,
                  lossFunc=lossFunc,
                  **kwargs)
    
    res.last = proj(res.last)
    return res

# main__grad = entropy__minimise__grad

def entropy__minimise__grad(
    Y,
#     loss,
    X0=None,
    beta=1.0,
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
        res = (1 + beta * X) * np.log(X + eps)
        res = res * Y
        res = np.mean(res)
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
        X0 = np.random.random(size=(1,size[1]))
#         S0 = randMat(N=dx,d=dy)
#     X0 = pynp.arr__rowNorm(X0)
        
    res = getTraj(step,
                  X0,
                  lossFunc=lossFunc,
                  **kwargs)
    
    res.last = proj(res.last)
    return res

main__grad = entropy__minimise__grad