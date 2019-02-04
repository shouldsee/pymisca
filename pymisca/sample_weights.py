# import numpy as np
import pymisca.numpy_extra as pynp
np = pynp.np
def expSD(data):
    x = np.std(data,axis=1)
    x = x - pynp.logsumexp(x)
    sample_weights = np.exp(x)
    return sample_weights
def expVAR(data):
    x = np.std(data,axis=1)
    x = np.square(x)
    x = x - pynp.logsumexp(x)
    sample_weights = np.exp(x)
    return sample_weights
def sd(data):
    SD = np.std(data,axis=1)
    sample_weights = SD
    return sample_weights
def var(data):
    SD = np.std(data,axis=1)
    sample_weights = SD**2
    return sample_weights
def constant(data):
    res = np.ones((len(data),))
    return res
