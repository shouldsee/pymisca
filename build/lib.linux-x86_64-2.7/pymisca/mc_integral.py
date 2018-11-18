
# from __future__ import absolute_import

import util as pyutil
# import pymisca.util as pyutil
np = pyutil.np
import scipy.special as spspec

def unif(D,R,N=1000):
    X = (np.random.random(size=(N,D))-0.5)*R
    return X

def cubicRInt(f,D,N=1000,R=5.):
    '''
    Integral on a hypercube
'''
    X = unif(N=N,D=D,R=R)
    val = np.mean(f(X),axis=0) * R**D
    return val

def suf(alpha):
    '''
    Surface area of a (D-1) sphere, with D = 2 * \alpha
'''
    val  = 2 * np.pi ** alpha / spspec.gamma(alpha)
    return val


def surfInt(f,N=10000,D=2):
    '''
    Integration on (the surface of) a (D-1) sphere
'''
    phis = pyutil.random_unitary((N,D))
#     phis = np.random.uniform(0,2 * np.pi,size=Ns)
    ys = f(phis)
    avg = np.mean(ys) * suf(D/2.)
    return avg