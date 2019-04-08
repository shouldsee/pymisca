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
    aggFunc = 'sum',
    **kwargs
):
    aggFunc = getattr(autograd.numpy,aggFunc)
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
        res = aggFunc(res)
#         assert 0
#         /1E10
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
    aggFunc = 'sum',
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
    aggFunc = getattr(autograd.numpy,aggFunc)
    
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
#         res = np.sum(res)
        res = aggFunc(res)
        
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


def MCE__grad(
    Y,
#     loss,
    X0=None,
    beta=0.0,
#     1.0,
    stepSize=1.0,
    eps=1E-10,
    aggFunc = 'sum',
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
    aggFunc = getattr(autograd.numpy,aggFunc)
    
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
#         res = np.sum(res)
        res = aggFunc(res)
        
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


def MCE__surfer(
    Y,
#     loss,
    X0=None,
    beta=0.0,
    stepSize=0.1,
#     1.0,
#     stepSize=1.0,
#     stepSize=[0.1, 0.001],
    eps=1E-10,
    aggFunc = 'sum',
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
    aggFunc = getattr(autograd.numpy,aggFunc)
    
    Y = np.array(Y)
    Y = pynp.arr__rowNorm(Y)
    if not isinstance(stepSize,list):
        stepSize = [stepSize, stepSize/float(Y.shape[1])]
#     if stepSize[1] = None:
#         stepSize[]
    size = Y.shape
    
    #### re-alias
#     X0 
#     S_ = S
    def proj(X):
        X = np.exp(X)
        X = X / np.sum(X,axis=1,keepdims=1)
        return X
#     def _loss(X):
        
    def _loss(X):
        X = proj(X)
        #### beta = 0. reduce to standard EM
        #### beta = 1. corresponding to reduce_entropy = 1.0
        res = (X) * ( np.log(Y + eps) + (beta - 1. ) * np.log(X + eps) )
#         res = np.sum(res)
        res = aggFunc(res)
        return res    
    if X0 is None:
        X0 = np.random.random(size=( size[0],size[1]))
#         X0 = np.log(Y +eps)
#         X0 = X0 + np.std(X0,axis=1,keepdims=True) * 0.1 \
#         * ( np.random.random(size) - 0.5 )
        
    _lossY = _loss(np.log(Y+eps))
#     _lossY = (_loss(np.log(Y+eps)) + _loss(X0))/2.
#     _lossY = _loss(X0)
    
    def lossFunc(X):

#         res = np.square( _lossY - _loss(X))
        res = -_loss(X)
        
        #### we are maximising the sum, hence minimising its negative
#         res = -res 
        return res

    gradFunc = autograd.grad(lossFunc)
#     stepSize = beta
    
    
    
    def step(args):
        '''
        Core iterative update
        '''
        X = args['X']
        hist = args['hist']
        
        grad = -gradFunc(X)
                
        gg = np.ravel(grad)
        
        move = gg*0.
        gn = pynp.arr__msq(gg,keepdims=0)
        gnorm = gg /gn
        
        dd = np.random.random(gg.shape)-0.5
        ddNormVect = dd/pynp.arr__msq(dd,keepdims=0)
        dd = ddNormVect * stepSize[1]
        
        gGrad = gnorm * np.sum(gnorm * dd)
        gTang =  dd - gGrad


#         move += stepSize[0] * gGrad
#         move += gTang
        
        move += stepSize[0] * gnorm
        move += gTang

        
#         dd.dot()
        X = X + np.reshape(move,X.shape)
    
        args['hist']['dd'] = dd
        args['X'] = X
        return args

    def step(args):
        '''
        Core iterative update
        '''
        X = args['X']
        hist = args['hist']   
        
        grad = -gradFunc(X)
        X = X + stepSize[0] * grad
        
#         args['hist']['dd'] = dd
        args['X'] = X
        return args
        
    # dx,dy=size

#         S0 = randMat(N=dx,d=dy)
#     X0 = pynp.arr__rowNorm(X0)
        
    res = getTraj(step,
                  X0,
                  lossFunc=lossFunc,
                  passDict = True,
                  **kwargs)
    
    res.last = proj(res.last)
#     res.lossY = lossY
    
    return res 
