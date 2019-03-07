import pymisca.iterative.fitTransition__MHD
import pymisca.numpy_extra as pynp
np = pynp.np
from pymisca.numpy_extra import distance__hellinger

def dist__MHD(X,Y,symmetric=True,**kwargs):
    res = pymisca.iterative.fitTransition__MHD.main(X,Y,**kwargs).hist['loss'][-1]
    if symmetric:
        X,Y = Y,X
        res = max(res,
                  pymisca.iterative.fitTransition__MHD.main(X,Y,**kwargs).hist['loss'][-1])
    return res


if __name__=='__main__':
    np.random.seed(0)
    N = 200
    dx= 10
    dy = 10
    X = np.random.random(size=(N,dx,))
    X = pynp.arr__rowNorm(X)
    Y = np.random.random(size=(N,dy,))
    Y = pynp.arr__rowNorm(Y)
    assert dist__MHD(X,Y) < 0.00524
