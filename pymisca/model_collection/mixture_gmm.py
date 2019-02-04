# def mixtue
import pymisca.models 
pymod = pymisca.models 
import mixem
# import sklearn.mixture as skmix 
import numpy as np
import pymisca.sample_weights
import functools
# reload(mixem)
import pymisca.ext as pyext
import pymisca.fop as pyfop

def l2norm(x,axis=None,keepdims=0):
    res = np.sum(x**2,axis=axis,keepdims=keepdims)**0.5
    return res

def oneHot(values):
    values = np.ravel(values)
    n_values = np.max(values) + 1
    res = np.eye(n_values)[values]
    return res



def _MixtureGMM__init(data,
                      init_method,
                      K,
                      beta,
                      meanNorm,
                      seed):
    np.random.seed(seed)
    def random( i=0):
        x = np.random.random(data.shape[1]) - 0.5
        mu = x
#         if i==100:
#             x = x * 0.
#         y = x / l2norm(x,) 
#         sigma = np.diag( abs(x) )
        sigma =  abs(x) + 0.1
        #### [TBC] ####
#         sigma =  np.diag(sigma)
#         dist = mixem.distribution.MultivariateNormalDistribution(
        dist = mixem.distribution.diagMVN(
            mu = mu ,
            sigma = sigma,
            beta = beta,
        )
#             print (dist.mu,dist.kappa)
        return dist        
    if init_method == 'random':
        dists = map(random, range(K))
        init_resp = None
    elif init_method == 'kmeans':
        dists = map(random, range(K))
#             self.dists = [None] * self.K
        import sklearn.cluster
        _mdl = sklearn.cluster.KMeans(n_clusters=K)
        ypred = _mdl.fit_predict(data)
        init_resp = oneHot(ypred)
    return dists,init_resp


def _MixutreGMM__fit__worker( (i,seed),
                             data,
                             init_method,
                             K,
#                              kappa,
                             beta,
                             meanNorm,
                             max_iters,
                             callback,
                             sample_weights,
                             min_iters,
                             fix_weights,
                             verbose,
                             **kwargs
                            ):

    dists,init_resp = _MixtureGMM__init(data,
                                        seed = seed,
                                       init_method=init_method,
                                       K=K,
#                                        kappa = kappa,
                                       beta = beta,
                                       meanNorm=meanNorm
                                       )
    llHist  = []
    def callback2(iteration, weight, distributions, log_likelihood, log_proba):
        llHist.append(log_likelihood)
        if verbose >= 2:
            if not iteration % 10:
                print ('[iter]{iteration},\
                log_likelihood={log_likelihood:.2f}'.format(**locals()))
        return (iteration, weight, distributions, log_likelihood, log_proba)
    if callback is None:
        callback = pyfop.identity 
    callback = pyfop.compositeF(callback,callback2)
    res = mixem.em(
        data, 
        dists,
        max_iters = max_iters,
        progress_callback=callback,
        sample_weights=sample_weights,
        init_resp=init_resp,
        fix_weights = fix_weights,
        min_iters = min_iters,
        **kwargs
#         progress_callback=simple_progress,
    )
    res = list(res)
    weights, distributions, _ =  res
    ll = llHist[-1]
    
    res += [ll]
    msg = '[]:Run {i}. Loglikelihood: {ll}'.format(**locals())
    if verbose >=1:
        print (msg)

#     res[-1] = ll[-1]
    return weights, distributions, np.array(llHist), ll



class MixtureGMM(pymod.MixtureModel):
    def __init__(self, K =30, 
                 kappa=False,
                 beta = 1.,
                 normalizeSample=False,
                init_method = 'kmeans',
                 NCORE=1,
                 weighted = False,
                 meanNorm = 1,
                 *args,
                 **kwargs
#                  normalizeSample=False
                ):
        '''Set kappa=False to ignore the normalisation constant
        beta palys the same role as kappa, but in a global fashion
        '''
        if K is not None:
            n_components = K
        K = n_components  
        self.NCORE = NCORE
        self.K  = K
#         self.kappa = kappa
        self.beta = beta
        self.normalizeSample = normalizeSample
        self.init_method = init_method
        self.weighted = weighted
        self.meanNorm = meanNorm
#         self.dists = None
        super(MixtureGMM,self).__init__(*args,**kwargs)
    def _init(self, data, 
              init_method = 'kmeans',
             ):
        init_method  = self.init_method
        K = self.K
#         kappa = self.kappa
        return MixtureGMM__init(data,init_method,K,kappa)
    @property
    def params(self):
        d = [ d.params for d in self.dists]
        d = pyext.dict__combine(d)
        return d
            
    def _fit(
        self,
        data,
#         n_components = 30,
#         K = None,
        verbose = 0,
        nStart = 5,
        n_print = None,
        callback = None,
#         sample_weights = 'expVAR',
        sample_weights = None,
        min_iters = 100,
#         fix_weights= 1,
#         sample_weights = None,
        max_iters = 3000,
        n_iter = None, ### dummy capturer 
        **kwargs
    ):
        
        self.callback = callback
        K = self.K
        fix_weights = not self.weighted
        ### preprocess data
        if self.meanNorm:
            data = data - np.mean(data, axis=1,keepdims =1)     
            
        if isinstance(sample_weights,basestring):
            _ = sample_weights
            sample_weights = getattr(pymisca.sample_weights,
                                     sample_weights,
                                     None)
            assert callable(sample_weights),'Dont know how to perform weight:%s' % _
        if callable(sample_weights):
            sample_weights = sample_weights(data)

            
        if self.normalizeSample:
            data = data / l2norm(data,axis=1,keepdims=1)        

#         if self.kappa is False:
#             self.kappa = l2norm(np.mean(data,axis=0))
            
        init_method  = self.init_method
        K = self.K
#         kappa = self.kappa
        beta = self.beta
        meanNorm = self.meanNorm
        lst = []

        worker = functools.partial(
            _MixutreGMM__fit__worker,
                 data=data,
                 init_method=init_method,
                 K=K,
#                  kappa=kappa,
                 beta = beta,
                 meanNorm = meanNorm,
                 max_iters = max_iters,
                 callback = callback,
                 sample_weights = sample_weights,
                 fix_weights = fix_weights,
                 verbose= verbose,
                 min_iters = min_iters,
                 **kwargs
         )
        lst = pyext.mp_map(worker, 
                           enumerate(np.random.randint(0, 10000,size=(nStart,))),
                           NCORE=self.NCORE)  

        #### select the best run
        lst = sorted(lst,key=lambda x:x[-1],reverse=True)
        weights, dists, llHist, ll = res = lst[0]
        if verbose >=1:
            msg = '[Best]:Run : Loglikelihood: {ll}'.format(**locals())
            print (msg)
        self.weights = weights
        self.dists = dists
        self.lastLL = ll
        self.histLL = llHist
        return llHist
main = MixtureGMM

from pymisca.callbacks import callback__stopAndTurn