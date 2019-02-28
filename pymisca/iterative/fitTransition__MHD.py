import pymisca.numpy_extra as pynp

import numpy as np
import pymisca.oop as pyop
import pymisca.iterative.base
from pymisca.iterative.base import getTraj

def fitTransition__MHD(X,Y,S0=None,
                eps=1E-10,
                 **kwargs
               ):
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
    dx = X.shape[1]
    dy = Y.shape[1]
    
    
    def step(S):
        '''
        Core iterative update
        '''
        R = np.sqrt(S)
        C = X.dot(S)
        A = np.sqrt(Y/(C + eps) )
        grad = X.T.dot(A) * R
    #     grad = X.T.dot(A)
        S = pynp.arr__rowNorm(grad**2)
    #     S = proj(grad**2)
        return S
    def lossFunc(S):
        ll = pynp.distance__hellinger(Y,X.dot(S))
        return ll

    # dx,dy=size
    if S0 is None:
        S0 = np.random.random(size=(dx,dy))
#         S0 = randMat(N=dx,d=dy)
    S0 = pynp.arr__rowNorm(S0)
    res = pymisca.iterative.base.getTraj(
        step,
        S0,
        lossFunc=lossFunc,
        **kwargs)   
    return res

main = minimise__HD = fitTransition__MHD
import scipy.optimize
# lst = []
def worker(i,size=None):
    dx,dy=size
    S0 = np.random.random(size=(dx,dy)).ravel()
    res = scipy.optimize.minimize(loss,x0=S0)
    res.s = proj(res.x)
    print res.fun
    return res
def proj(S,size=None):
    S = abs(S)
    S = S.reshape(size)
    S = pynp.arr__rowNorm(S)
    return S
def loss(S,X=None,Y=None):
    S = proj(S)
    loss = pynp.distance__hellinger(Y,X.dot(S))
    return loss