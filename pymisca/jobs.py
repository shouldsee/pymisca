import numpy  as np
import pymisca.model_collection.mixture_vmf
_mod1 = pymisca.model_collection.mixture_vmf

def job__cluster__mixtureVMF__incr(
                                  tdf,
                                  K = 20,
                                  randomState=0,
                                  nIter=100,
                                  nStart=1,
                                  start=0.1,
                                  end= 24.0,
                                  step = None,
                                  init_method='random',
                                  meanNorm=1,
                                  normalizeSample=0,
    weighted=True,
    
                                   alias = 'mdl',
                                   verbose=0,
):
    data = tdf
    mod = pymisca.model_collection.mixture_vmf
    
    np.random.seed(randomState)
#     betas = lambda i: (i + 1) * 0.00015 + 0.15
    if step is None:
        step = (end-start)/nIter
    callback = mod.callback__stopAndTurn(
        start=start,
        step=step)
#     callback = pyfop.composeF(callback__stopAndTurn(betas=betas),
# #                              callback__stopOnClu(interval=1)
#                              )
    mdl0 = mdl = mod.MixtureVMF(
        init_method = init_method,
                        NCORE = 1,
                        meanNorm=meanNorm,
#                          beta = betas(0),
                        beta = start,
                         weighted =  weighted,
                         normalizeSample=normalizeSample,
                        kappa = None,
                        K = K,)
    res = mdl.fit(
        data,verbose=verbose,
                  nStart=nStart,
                  callback = callback,
                  min_iters = nIter,
                  max_iters = nIter,
                  sample_weights=None,
                 )    
    np.save(alias + '.npy',mdl0,)
    return mdl0



