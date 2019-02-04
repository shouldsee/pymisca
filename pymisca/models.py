
import pymisca.numpy_extra as pynp
np = pynp
import pymisca.ext as pyext
import pandas as pd

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
            prob = np.exp(log)
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
                 *args,
                 **kwargs
                ):
        self.weights = weights 
        self.dists = dists
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
        
    def predict(self,X, entropy_cutoff = None, **kwargs):
        proba = self.predict_proba(X,norm = 0,**kwargs)
        clu = np.argmax(proba,axis = 1)
        
        if entropy_cutoff is not None:
            clusterH = self._entropy_by_cluster(X)
            proj = {x:x for x in np.where( clusterH < entropy_cutoff)[0]}
            clu = np.vectorize( lambda x: proj.get(x, -1)) (clu)
        return clu
        