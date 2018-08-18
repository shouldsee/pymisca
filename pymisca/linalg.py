import numpy as np


def meanNormBasis(L,orthonormal = 0):
    '''Calculate meanNorm operator in matrix form
'''
    I = np.eye(L)
    W = I - np.ones((L,L),dtype='float')/L
    if orthonormal:
        W = gram_schmidt(W)
    return W

def gram_schmidt(vectors,rowVec=1):
    '''
Source: https://gist.github.com/iizukak/1287876/edad3c337844fac34f7e56ec09f9cb27d4907cc7
'''
    basis = []
    if not rowVec:
        vectors= np.transpose(vectors)
    for v in vectors:
        w = v - np.sum( np.dot(v,b)*b  for b in basis )
        if (w > 1e-10).any():  
            basis.append(w/np.linalg.norm(w))
    basis = np.array(basis)
    if not rowVec:
        basis = np.transpose(basis)
    return basis

def norm2basis(x):
    '''calculate matrix from the normal vector of a hyperplane
'''

    L = len(x)
    W = np.zeros((L,L))
    W[:,0] = x[0]
    li = np.arange(L-1)
    W[li,li+1] = - np.array(x[1:])
    W[-1,0]=-x[0] * (L-1)
    return W
