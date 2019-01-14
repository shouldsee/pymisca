# def mixtue
import pymisca.models 
import mixem
# import sklearn.mixture as skmix 
import numpy as np
# reload(mixem)

def mm__get__logProba(dists,weights,data):
    ''' Get log proba for a mixture model
'''
    n_data = len(data)
    n_distr = len(dists)
    log_density = np.empty((n_data, n_distr))
    for d in range(n_distr):
        log_density[:, d] = dists[d].log_density(data)
    logProba = np.log(weights[None,: ]) + log_density    
    return logProba        

class MixtureModel(pymisca.models.BaseModel):
    def __init__(self, 
                 weights= None,
                 dists = None, 
                 **kwargs
                ):
        self.weights = weights 
        self.dists = dists
        super(MixtureModel,self).__init__()
    def _predict_proba(self,data, ):
        return self._log_pdf(data)
    def _log_pdf(self, data):
        logP = mm__get__logProba(self.dists,self.weights, 
                                 data)
        return logP
    def _fit(self,data,**kwargs):
        assert 0,'Not Implemented!'



class MixtureVMF(MixtureModel):
    def _fit(
        self,
        data,
        n_components = 30,
        K = None,
        verbose = 0,
        nStart = 5,
        n_iter = None,
        n_print = None,
    ):
        if K is not None:
            n_components = K
        K = n_components
        def l2norm(x,axis=None):
            res = np.sum(x**2,axis=axis)**0.5
            return res

        def random( i=0):

            x = np.random.random(data.shape[1]) - 0.5
            if i==100:
                x = x * 0.
            y = x / l2norm(x,) 
            y = y 
            dist = mixem.distribution.vmfDistribution(y ,
                                       kappa = 1.,
                                      )
            return dist
        
        ### preprocess data
        data = data- np.mean(data, axis=1,keepdims =1)        
        
        lst = []
        for i in range(nStart):
            dists = map(random, range(K))
            res = mixem.em(
                np.array(data), 
                dists,
                tol=1e-30,
                max_iterations=n_iter,
                progress_callback=None,
        #         progress_callback=simple_progress,
            )
            res = list(res)
            weights, distributions, llHist =  res
            ll = llHist[-1]
            res += [ll]
            msg = '[]:Run {i}. Loglikelihood: {ll}'.format(**locals())
            if verbose >=1:
                print (msg)

        #     res[-1] = ll[-1]
            lst += [res]

        #### select the best run
        lst = sorted(lst,key=lambda x:x[-1],reverse=True)
        weights, dists, llHist, ll = res = lst[0]
        
        self.weights = weights
        self.dists = dists
        return llHist
main = MixtureVMF