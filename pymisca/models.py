
import pymisca.numpy_extra as pynp
np = pynp
class BaseModel(object):
    def __init__(self,name='test'):
        self.name = name
#         print (type(self),self.__dict__)
        pass
    def sanitise(self,X):
        X  = np.asarray(X,np.float32)        
        return X
    def fit(self,X,n_iter = 1000, n_print=100,
            **kwargs):
        X = self.sanitise(X)
        res = self._fit(X,n_iter = n_iter,n_print = n_print,
                  **kwargs)
        return res
    
    def predict_proba(self,X,norm = 1,log=1):
        X = self.sanitise(X)
        prob = self._predict_proba(X)
        if norm:
            prob = prob - pynp.logsumexp( prob, axis =1,keepdims=1)
        if not log:
            prob = np.exp(log)
        return prob
    def score(self,X,keepdims=0):
        prob = self.predict_proba(X,norm=0,log=1)
        score = pynp.logsumexp( prob, axis =1,keepdims=keepdims)
        return score
    
    def predict(self,X):
        proba = self.predict_proba(X,norm = 0)
        clu = np.argmax(proba,axis = 1)
        return clu
    
    def expand_input(self,X):
        N = len(X)
        X_bdc = tf.tile(
            tf.reshape(
                X, [N, 1, 1, self.D]),
                   [1, 1, self.K, 1])
        return X_bdc
    
