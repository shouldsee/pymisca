from pymisca.iterative.base import getTraj
import pymisca.numpy_extra as pynp

import numpy as np
import pymisca.oop as pyop

def fit__MaxVarAttention(
    Y,S0=None,
    weight = None,
    stepSize=0.0001,
    lossTol = 0.05,
    speedTol = 1E-3,
    beta = 1.,
#     X0=None,
    eps=1E-10,
    **kwargs):
    '''
    Author: Feng Geng
    X: numpy.ndarray (i,j)
    Y: numpy.ndarray (i,k)
    S0: numpy.ndarray (j,k)
    
    Minimise the hellinger distance between Y and X \cdot S
    non-negative matrices x_{ij}, y_{ik}, s_{jk} with constraints
    \sum_j x_{ij} = 1
    \sum_k y_{ik} = 1
    \sum_k s_{jk} = 1 
    minimise  E = - \sum_{i,k} \sqrt{ \sum_j x_{ij} y_{ik} s_{jk} } 
    by iteratively setting 
    \sqrt{ s_{jk} } = d E / d \sqrt{ s_{jk} }
    in a normalised fashion
    '''
#     dx,dy = S.shape
    N,dy = Y.shape
    size = (1,dy)
#     size = (N,dx)
    
    #### re-alias
#     S0 = X0 
#     S_ = S
    if weight is None:
        weight = np.ones((len(Y),1))
    wsum = np.sum(weight)
    
    def proj(S,):
#         return S/pynp.arr__l2norm(S,axis=1)
        return pynp.arr__rowNorm(abs(S))
    def step(S_):
        '''
        Core iterative update
        '''
        
        if 0:
            R = S_
    #         R = S**0.5
            S = R**2
            grad = -2 * Y * (Y * S).sum(axis=1,keepdims=1) + Y  ** 2
            grad = grad * 2 * R
            grad = np.mean( grad, axis=0,keepdims=True)
    #         S = S + (grad - grad.mean())* stepSize
    #         S = np.clip(S,0,None)
            S = proj(S)
#         grad = Y ** 2 - 2* Y.dot(S.T)
        if 1:
            S = S_
            grad = -2 * Y * (Y * S).sum(axis=1,keepdims=1) + Y  ** 2
            grad = grad * beta
            grad = np.sum( grad * weight, axis=0,keepdims=True)/wsum
            S = S + (grad - grad.mean())* stepSize
            S = np.clip(S,0,None)
            S = proj(S)
        
        return S
    
    def lossFunc(S):
#         ll = - pynp.weightedVariance( Y,weight=S.T[None],keepdims=0).max(axis=1).sum()
        ll = pynp.weightedVariance( Y,weight=S.T[None],keepdims=0).max(axis=1).mean()
#         ll = pynp.distance__hellinger(Y, S.dot(S_))
        return  ll
    # dx,dy=size
    if S0 is None:
        S0 = np.random.random(size=size)
#         S0 = randMat(N=dx,d=dy)
    S0 = proj(S0)
#     S0 = pynp.arr__rowNorm(S0) ** 0.5
        
    res = getTraj(step,
                  S0,
                  lossFunc=lossFunc,
                  lossTol = lossTol,
                  speedTol = speedTol,
                  **kwargs)
    return res
main = fit__MaxVarAttention