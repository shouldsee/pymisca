import tensorflow as tf
import numpy as np

def getSimp_(shape, name, mode = None,method='l2norm'):
    if mode is None:
        mode = method
    x_raw = tf.get_variable(shape=shape,name = name)
    eps = tf.constant(1E-07)
    if mode == 'l2norm':
        x_simp = tf.square(tf.nn.l2_normalize( 
            x_raw,
            axis= -1,
        ))
    elif mode == 'expnorm':
        x_simp = tf.nn.softmax( 
#                         -tf.nn.softplus(
                x_raw,
#                         ),
            axis= -1,
        )
    elif mode == 'logitnorm':
        x_raw = tf.log_sigmoid(x_raw)
        x_simp = tf.nn.softmax(x_raw,axis=-1)
#                     x_simp = x_raw / (tf.reduce_sum(x_raw,axis=-1,keepdims = True) + eps
    elif mode == 'logit':
        x_raw = tf.sigmoid(x_raw)
        x_simp = x_raw

    elif mode == 'beta':
        x_raw = tf.log_sigmoid(x_raw)
        x_raw = tf.cumsum(x_raw,axis=-1)
#                     x_raw = tf.exp(x_raw)
        ones = tf.ones(shape=x_raw.shape[:-1].as_list() +  [1,] )
#                     ones = tf.ones(shape=x_raw.shape[:-1] + [1,])
        x_p  = tf.concat([ ones, tf.exp(x_raw) + eps ],axis=-1)
#                                tf.gather(x_raw,
#                                          range(0,x_raw.shape[-1]),
#                                         axis=-1)],axis=-1)
        L = x_p.shape[-1]
        x_p = tf.gather(
            x_p, range(0,L-1),axis=-1) - tf.gather(
            x_p,range(1,L),axis=-1) 
#                     x_p = tf.diff(x_p,axis=-1)

        x_simp = x_p
#                     x_simp = tf.gather(x_p,)
#                     x_raw = 
    else:
        raise Exception('mode not implemented:%s'%mode)
    return x_simp

def wrapper_perm(f,perm=None,iperm = None):
    '''Apply two permutation before and after transformation
'''
#     if perm is not None:
#         iperm = np.argsort(perm)
#     else:
#         iperm = perm
    def g(X,*args,**kwargs):
            
        X = tf.transpose(X,perm=perm)
        Y = f(X,*args,**kwargs)
        Y = tf.transpose(Y,perm=iperm)
        return Y
    return g
def take_tril(X,n,k=-1,m=None, transpose=False,):
    '''
    Take tril of the first two dimensions
'''
#     wrapper__perm()
    ul = np.tril_indices(n,-1)
    ulzip = zip(*(ul))
    if transpose:
        ul = ul[::-1]
    Y = tf.gather_nd(X,zip(*(ul)))
    return Y


def batchMaker__movingwindow(batchSize=100,stepSize=50):
    '''Take convolutional windows as batches
'''
    def batchMaker(ts, i):
        L = len(ts)
        i = (stepSize * i) % (L - batchSize)
        return ts[i:i+batchSize]
    return batchMaker

def batchMaker__random(batchSize=100):
    '''Take random subsamples as batches
'''
    def batchMaker(ts, i):
        L = len(ts)
        d = 100
        idx = np.random.randint(0,L,size=(batchSize,))
        return ts[idx]
    return batchMaker

def batchMaker__randomWindow(batchSize=100, windowNumber=None, windowSize=None):
    '''Take random convolutional windows as batches
'''
    errMsg = 'only specify ONE of "windowNumber" or "windowSize"'
    if windowSize is not None:
        assert windowNumber is None, errMsg
        windowNumber = batchSize//windowSize
    else:
        if windowNumber is None:
            windowSize = windowNumber = int(batchSize**0.5)
        else:
#             assert windowNumber is not None, errMsg
            windowSize = batchSize//windowNumber
    def batchMaker(ts, i):
        L = len(ts)
        idx = np.random.randint(0, (L - windowSize),size=(windowNumber,))
        idx = np.hstack([np.arange(i,i+windowSize) for i in idx])
#         i = (stepSize * i) % (L - batchSize)
        return ts[idx]
    return batchMaker


# from pymisca.affine_transform_diag import *