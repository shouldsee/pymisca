from pymisca.iterative.base import getTraj
import pymisca.numpy_extra as pynp

import numpy as np
import pymisca.oop as pyop

def decode__MHD(Y,S,X0=None,
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
    dx,dy = S.shape
    N,dy = Y.shape
    size = (N,dx)
    
    #### re-alias
    S0 = X0 
    S_ = S
    
    
    def step(X):
        '''
        Core iterative update
        '''
#         R = np.sqrt(S)
        C = X.dot(S_)
        A = np.sqrt(Y/(C + eps) )
        grad = A.dot(S_.T) * np.sqrt(X)
#         grad = X.T.dot(A) * R
    #     grad = X.T.dot(A)
        X = pynp.arr__rowNorm(grad**2)
    #     S = proj(grad**2)
        return X
    
    def lossFunc(S):
        ll = pynp.distance__hellinger(Y, S.dot(S_))
        return  ll
    # dx,dy=size
    if S0 is None:
        S0 = np.random.random(size=size)
#         S0 = randMat(N=dx,d=dy)
    S0 = pynp.arr__rowNorm(S0)
        
    res = getTraj(step,
                  S0,
                  lossFunc=lossFunc,
                  **kwargs)
    return res
main = decode__MHD