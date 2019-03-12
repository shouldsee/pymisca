
import pymisca.numpy_extra as pynp
np = pynp
import pymisca.ext as pyext
import pandas as pd
import collections
import  sys
pymod = sys.modules[__name__]

class BaseModel(object):
    def __init__(self,name='test',lastLL=None, data = None):
        self.name = name
        self.lastLL = lastLL
        self.data  = data
#         print (type(self),self.__dict__)
        pass
    def __getstate__(self):
        d = dict(self.__dict__)
#         del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d) # I *think* this is a safe way to do it
        
    def sanitise(self,X = None):
        if X is None:
            assert self.data is not None
            X = self.data        
        X  = np.asarray(X,np.float32)        
        return X
    def fit(self,X = None,n_iter = 1000, n_print=100,
            **kwargs):
        if X is not None:
            self.data = X
        X = self.sanitise(X)
        res = self._fit(X,n_iter = n_iter,n_print = n_print,
                  **kwargs)
        return res
    
    def predict_proba(self,X= None,norm = 1,log=1, **kwargs):

        X = self.sanitise(X)
        prob = self._predict_proba(X,**kwargs)
        if norm:
            prob = prob - pynp.logsumexp( prob, axis =1,keepdims=1)
        if not log:
            prob = np.exp(prob)
        return prob
    
    def score(self,X,keepdims=0, **kwargs):
        prob = self.predict_proba(X,norm=0,log=1,**kwargs)
        score = pynp.logsumexp( prob, axis =1,keepdims=keepdims)
        return score
    
    def predict(self,X, **kwargs):
        proba = self.predict_proba(X,norm = 0,**kwargs)
        clu = np.argmax(proba,axis = 1)
        return clu
    
    def predictClu(self,X, index = None, **kwargs):
        clu = self.predict(X, **kwargs)
        clu = pd.DataFrame(clu,columns =['clu'],index=index)
        return clu
    
    def expand_input(self,X):
        N = len(X)
        X_bdc = tf.tile(
            tf.reshape(
                X, [N, 1, 1, self.D]),
                   [1, 1, self.K, 1])
        return X_bdc
    
def cache__model4data(mdl,tdf,ofname = None):
    logP = mdl.predict_proba(tdf, norm = 0, log = True)
    logP = pd.DataFrame(logP,index=tdf.index,)    
    logPN = pd.DataFrame(logP.values - 
                         pynp.logsumexp(logP.values,axis=1),
                        index=logP.index
                        )
    score = logPN.max(axis=1).to_frame('score')
    
    if hasattr(mdl,'predictClu'):
        clu = mdl.predictClu(tdf,index=tdf.index,)
    else:
        clu = mdl.predict(tdf)
        clu = pd.DataFrame(clu,index=tdf.index,columns=['clu'])
    stats = pd.concat([clu,score],axis=1)
    res = dict(logP = logP,
               logPN = logPN,
               clu = clu,
               score= score,
               stats= stats)
    if ofname is not None:
        np.save(ofname, res)
        return ofname 
    else:
        return res

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

class MixtureModel(BaseModel):
    def __init__(self, 
                 weights= None,
                 dists = None, 
                 weighted = True,
                 *args,
                 **kwargs
                ):
        self.dists = dists
        self.K = len(dists)
        if weights is None:
            weights = np.ones(self.K).mean(keepdims=1)
        self.weights = weights 
        self.weighted = weighted
        super(MixtureModel,self).__init__(*args,**kwargs)
        
    def _predict_proba(self,data, ):
        return self._log_pdf(data)
    
    def _log_pdf(self, data):
        logP = mm__get__logProba(
            self.dists,
            self.weights, 
            data)
        return logP
    
    def _entropy_by_cluster(self, data):
        log_proba = self._log_pdf(data)
        part = pynp.logsumexp(log_proba,axis=1,keepdims=1)
        proba = np.exp(log_proba - part)
        H = pyext.entropise(proba,normed=1,axis=1).sum(axis=1,keepdims=1)
#         stat = H.std()
        resp = proba/proba.sum(axis=0,keepdims=1) 
        clusterH = (H * resp).sum(axis=0)    
        return clusterH
    
    def _fit(self,data,**kwargs):
        assert 0,'Not Implemented!'
        
    def predict(self,X, entropy_cutoff = None, method='reorder', **kwargs):
        proba = self.predict_proba(X,norm = 0,**kwargs)
        clu = np.argmax(proba,axis = 1)
        
        if method=='orig':
            pass
        elif entropy_cutoff is not None:
            if method=='reorder':
                clusterH = self._entropy_by_cluster(X)
                proj = collections.OrderedDict()
                
                i = 0
                inc = np.where( clusterH < entropy_cutoff)[0]
                od = inc[np.argsort(clusterH[inc])]
                proj = {v:i for i,v in enumerate(od)}
#                 print (proj)
#                 for _,v in enumerate(np.argsort(clusterH)):
#                     if clusterH[v] > entropy_cutoff:
#                         proj[v] = -1
#                     else:
#                         proj[v] = i
#                         i += 1
#                     print(proj)
#                 proj = {v:i for i,v in enumerate(np.argsort(clusterH))}
            elif method=='oorder':
                clusterH = self._entropy_by_cluster(X)
                proj = {v:v for i,v in enumerate(np.argsort(clusterH))}
                
                for v in np.where( clusterH > entropy_cutoff)[0]:
                    proj[v] = -1
#             proj = {x:x for x in np.where( clusterH < entropy_cutoff)[0]}
            clu = np.vectorize( lambda x: proj.get(x, -1 )) (clu)
    
        return clu
    
    
import pymisca.fop as pyfop
class EMMixtureModel(MixtureModel):
#     def __init__(self, 
                 
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
        init_resp = None,
#         fix_weights= 1,
#         sample_weights = None,
        max_iters = 3000,
        n_iter = None, ### dummy capturer 
        **kwargs
    ):
        
            
        self.callback = callback
        fix_weights = not self.weighted
#         not getattr(self,weighted,True)
#         init_method  = self.init_method
        K = self.K
#         kappa = self.kappa
#         beta = self.beta
#         meanNorm = self.meanNorm
#         lst = []
    
        llHist  = []
        def callbackInternal(iteration, weight, distributions, log_likelihood, log_proba):
            llHist.append(log_likelihood)
            if verbose >= 2:
                if not iteration % 10:
                    print ('[iter]{iteration},\
                    log_likelihood={log_likelihood:.2f}'.format(**locals()))
            return (iteration, weight, distributions, log_likelihood, log_proba)
        if callback is None:
            callback = pyfop.identity 
        callback = pyfop.compositeF(callback,callbackInternal)
        res = mixem.em(
            data, 
            self.dists,
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
        
        i = 0
        msg = '[]:Run {i}. Loglikelihood: {ll}'.format(**locals())
        if verbose >=1:
            print (msg)

#         return weights, distributions, np.array(llHist), ll    
    
        return llHist
        

        
import mixem.distribution    

import pymisca.proba as pyprob
import pymisca.numpy_extra as pynp;np = pynp.np
class distribution(mixem.distribution.Distribution):
    def __init__(self,**kwargs):
        pass
    
    def __repr__(self,):
        raise Exception('__repr__() not implmented')
        
    def log_density(self,*a,**kw):        
        return self.log_pdf(*a,**kw)
    
    def estimate_parameters(self,*a,**kw):
        raise Exception('Not Implemented')
    
    def log_pdf(self,X,**kwargs):
        X = np.array(X)
        logP = self._log_pdf(X,**kwargs)
        if np.ndim(logP) == 2:
            logP = np.sum(logP,axis=1)
        return logP
    
    def pdf(self,X,**kwargs):
        res = np.exp(self.log_pdf(X,**kwargs))
#         res = self._pdf(X,**kwargs)
        return res




class normalDist(distribution):
    def __init__(self, 
                 loc=0., 
                 scale=1.,
                 **kwargs):
        self.loc = loc
        self.scale = scale
        self = super(normalDist,self).__init__(**kwargs)
        
    def _log_pdf(self,X):
        X = X -self.loc
        X = X/self.scale
        res = -np.square(X)
        return res

class normNormalDist(distribution):
    '''
    Ugly yet useful.
    '''
    def __init__(self, 
                 loc=0., 
#                  scale=1.,
                 **kwargs):
        self.loc = loc
#         self.scale = scale
        self = super(normNormalDist,self).__init__(**kwargs)
        
    def _log_pdf(self,X):
        Y = self.loc
        sumsq = (X**2 + Y**2 )
#         sumsq = 1/ (1./ X**2 + 1./ Y**2 )
#         sumsq = (Y**2 + 0* X)
        logP = -(X-Y) ** 2 

#         logP = X*Y
#         logP = -(X-Y) ** 2
        isZero = sumsq==0.
#         if isinstance(isZero,bool):
        if np.ndim(X)==0:
            if isZero:
                return -1
            else:
                return logP/sumsq
        else:
            logP[~isZero] = logP[~isZero]/sumsq[~isZero]
            logP[isZero] = -1
            
#         logP = -(X+Y)**2
#         anti = (X + Y)*Y < 0
        anti = X*Y < 0.
#         anti = 0
        logP = logP * ( 1 - anti) + anti * -1
        return logP

    

class simpleJSUDist(distribution):
    '''
    Elegant
    '''
    def __init__(self, 
                 loc=0., 
#                  scale=1.,
                 **kwargs):
        self.loc = loc
#         self.scale = scale
        self = super(simpleJSUDist,self).__init__(**kwargs)
        
    def _log_pdf(self,X):
        Y = self.loc
        
#         sumsq = (X**2 + Y**2 )
#         sumsq = (Y**2 + 0* X)
        d = np.arcsinh(X) - np.arcsinh(Y)
        logP = -np.square(d)

        return logP        
    
import scipy.special
class Binomial(pymod.distribution):
    def __init__(self,N,p,asInt=0):
        self.N = N 
        self.p = p
        self._loggammaN = scipy.special.loggamma(N+1)
        self.asInt = asInt
    def _log_pdf(self, X):
        if self.asInt:
            X = (X+0.5).astype(int)
        count = np.vstack([X, self.N-X]).T
        logP = self._loggammaN - np.sum( scipy.special.loggamma(count+1),axis=1)
#         if self.asInt:
#             logP = np.nan_to_num(logP)
        logP += count.dot(np.log( [self.p, 1.-self.p] ))
#         print logP
        return logP.astype(float)


def nearestGridAppx(X,D):
# def f(X):
    X = X*D
    Y = (X +0.5 ).astype(int)    
    
    arg = np.argsort((Y - X),axis=1)
    res = Y.sum(axis=1,keepdims=1) 
    diff = (res - D)
    FLAG = (arg < diff) * (diff > 0) | (diff < 0) * ((arg - D) >= diff)  
    Y = Y + FLAG * -np.sign(diff)

    return Y

class Multinomial(pymod.distribution):
    def __init__(self, eta,mean,asInt=0):
#         p = mean
        self.eta = eta
        self.mean = np.array(mean)
        self.mean = self.mean
        assert np.allclose(self.mean.sum(),1.)
        self.D = len(self.mean)
        self._loggammaEta = scipy.special.loggamma(eta+1).astype(float)
        self.asInt = asInt
    def _log_pdf(self, X):
        if self.asInt:            
            #### not implemented
#             Y = (X+0.5).astype(int)
            X = nearestGridAppx(X,self.D)
#             X = nearestGrid(X)
        count = X 

        logP = self._loggammaEta - np.sum(scipy.special.loggamma(count+1),
                                        axis=1).astype(float)
        logP += count.dot(np.log(self.mean))
        return logP    

class NormedMultinomial(pymod.Multinomial):
    '''This approximation breaks down when N is small or 
    any of p approaches zero.
    '''
    def _log_pdf(self, X):
        count = (X * (self.eta+1)) - 0.5
#         count = X * self.eta
        return super(NormedMultinomial,self)._log_pdf(count) + (self.D-1) * np.log(self.eta+1)
    def estimate_parameters(self,data,weights):
        wdata = data * weights[:,None]
        mean = wdata.mean(axis=0)
        SUM = mean.sum()
        if SUM!=0:
            mean = mean/SUM
            self.mean = mean
        