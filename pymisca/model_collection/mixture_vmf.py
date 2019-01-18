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

# def mm__get__logProba(dists,weights,data):
#     ''' Get log proba for a mixture model
# '''
#     n_data = len(data)
#     n_distr = len(dists)
#     log_density = np.empty((n_data, n_distr))
#     for d in range(n_distr):
#         log_density[:, d] = dists[d].log_density(data)
#     logProba = np.log(weights[None,: ]) + log_density    
#     return logProba        

# class MixtureModel(pymisca.models.BaseModel):
#     def __init__(self, 
#                  weights= None,
#                  dists = None, 
#                  *args,
#                  **kwargs
#                 ):
#         self.weights = weights 
#         self.dists = dists
#         super(MixtureModel,self).__init__(*args,**kwargs)
        
#     def _predict_proba(self,data, ):
#         return self._log_pdf(data)
    
#     def _log_pdf(self, data):
#         logP = mm__get__logProba(
#             self.dists,
#             self.weights, 
#             data)
#         return logP
#     def _fit(self,data,**kwargs):
#         assert 0,'Not Implemented!'

def _MixtureVMF__init(data,init_method,K,kappa,beta,seed):
    np.random.seed(seed)
    def random( i=0):
        x = np.random.random(data.shape[1]) - 0.5
        if i==100:
            x = x * 0.
        y = x / l2norm(x,) 
        y = y 
        dist = mixem.distribution.vmfDistribution(
            mu = y ,
            kappa = kappa,
            beta = beta,
#                 sample_weights = sample_weights,
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

def _MixutreVMF__fit__worker( (i,seed),
                             data,
                             init_method,
                             K,
                             kappa,
                             beta,
                             max_iters,
                             callback,
                             sample_weights,
                             min_iters,
                             fix_weights,
                             verbose,
                             **kwargs
                            ):

    dists,init_resp = _MixtureVMF__init(data,
                                        seed = seed,
                                       init_method=init_method,
                                       K=K,
                                       kappa = kappa,
                                       beta = beta,)
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

class MixtureVMF(pymod.MixtureModel):
    def __init__(self, K =30, 
                 kappa=False,
                 beta = 1.,
                 normalizeSample=True,
                init_method = 'kmeans',
                 NCORE=1,
                 weighted = False,
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
        self.kappa = kappa
        self.beta = beta
        self.normalizeSample = normalizeSample
        self.init_method = init_method
        self.weighted = weighted
#         self.dists = None
        super(MixtureVMF,self).__init__(*args,**kwargs)
    def _init(self, data, 
              init_method = 'kmeans',
             ):
        init_method  = self.init_method
        K = self.K
        kappa = self.kappa
        return MixtureVMF__init(data,init_method,K,kappa)
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
        sample_weights = 'expVAR',
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

        if self.kappa is False:
            self.kappa = l2norm(np.mean(data,axis=0))
            
        init_method  = self.init_method
        K = self.K
        kappa = self.kappa
        beta = self.beta

        lst = []

        worker = functools.partial(
            _MixutreVMF__fit__worker,
                 data=data,
                 init_method=init_method,
                 K=K,
                 kappa=kappa,
                 beta = beta,
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
main = MixtureVMF

import pymisca.numpy_extra as pynp
import copy
class callback__stopAndTurn(object):
    def __init__(self, interval = 1, cluMin = 2, burnin = 10,
                 afterTurn = 50,
                 turning = False,
                 start=None,step = None,
                betas = None):
        self.interval = interval
        self.cluMin  = cluMin
        self.burnin = burnin
        if betas is None:
            assert start is not None
            assert step is not None
            betas = lambda i: start + i * step
        self.turning = turning
        self.betas = betas
        self.lastTurn = None
        self.right= None
        self.left = None
        self.stats = []
        self.betaHist = []
        self.cluNum = []
        self.H = []
        self.clusterH = []
#         self.mode = 'lr'
        self.mode = 'r'
        self.interval = 1
        self.mdls = []
    def saveModel(self,  *args):
        iteration, weight, distributions, log_likelihood, log_proba = args
        if not iteration % self.interval:
            mdl = pymod.MixtureModel(weights=weight,
                                   dists= distributions,
                                   lastLL = log_likelihood,)
            self.mdls.append( 
                [ iteration, copy.deepcopy(mdl) ]
                      )
        return args      
    def __call__(self,*args):
        iteration, weight, distributions, log_likelihood, log_proba = args
        if iteration > self.burnin:
#             if not iteration % self.interval:
            cluNum = len(set(np.argmax(log_proba,axis=1)))
            self.cluNum.append(cluNum)
#             stat = cluNum
            part = pynp.logsumexp(log_proba,axis=1,keepdims=1)
            proba = np.exp(log_proba - part)
            H = pyext.entropise(proba,normed=1,axis=1).sum(axis=1,keepdims=1)
            stat = H.std()
            resp = proba/proba.sum(axis=0,keepdims=1) 
            clusterH = (H * resp).sum(axis=0)
            beta = distributions[0].beta
            self.betaHist.append(beta)
            self.stats.append(stat)
#             self.H.append(H.ravel())
            self.clusterH.append(clusterH)
            self.saveModel(*args)
            
            if not self.turning:
                beta = self.betas(iteration)
                if self.right is None:
                    if cluNum >= self.cluMin:
                        self.rightIter  = self.lastTurn = iteration

                        print ('[cbk]iter={iteration}: beta={beta:.3E} \
                        Cluster multiplexing'.format(**locals()))
                        self.right = beta
                    
            else:
                if self.right is None:
                    if cluNum >= self.cluMin:
                        self.rightIter  = self.lastTurn = iteration

                        print ('[cbk]iter={iteration}: Now going left.'.format(**locals()))
                        self.right = beta
                elif self.left is None:
                    if cluNum < self.cluMin:
                        self.lastTurn = iteration
                        self.leftIter = self.rightIter *2 - iteration
                        self.left = beta
                        print ('[cbk]iter={iteration}: Now going right.'.format(**locals()))
                        self.going = 'right'

                if self.right is None:
                    beta = self.betas(iteration)
        #             self.turnBeta = beta
                else:
                    if self.left is None:
                        self.going = 'left'

                    if self.going == 'left':
                        vit = self.rightIter - ( iteration -  self.lastTurn)
                        beta = self.betas(vit)
                    else:
                        vit =  self.leftIter + (  iteration - self.lastTurn  )
                        beta = self.betas(vit)

        #                     self.lastTurn = iteration
        #                 beta = np.random.uniform(self.left,self.right)
                    if (beta > self.right) and ('l' in self.mode):
                        self.going = 'left'
                        self.lastTurn = iteration
                        print ('going %s' % self.going)

                    if beta < self.left and ('r' in self.mode):
                        self.going = 'right'
                        self.lastTurn = iteration
                        print ('going %s' % self.going)

#                     args = (-1, ) + args[1:]
        elif iteration <= self.burnin:
            beta = self.betas(iteration)
        
        for d in distributions:
            d.beta = beta
        return args

import pymisca.vis_util as pyvis
plt = pyvis.plt
def qc__vmf(mdl=None,
            callback = None,
            nMax=10000,
            xunit = None,
           ):
    if callback is None:
        assert mdl is not None
        callback = mdl.callback
#     n = 4000
#     xmax = 1.0
    fig,ax = plt.subplots(1,1,figsize=[14,10])
    if xunit is not None:
        xs = getattr(callback,xunit)
    else:
        xs = np.arange(len(callback.betaHist))
    axs = pyvis.qc_2var(*np.broadcast_arrays(np.array(xs)[:,None], 
                                             callback.clusterH)
                       ,nMax=nMax,axs=[None,ax,None,None])

    plt.sca(axs[1])
    # plt.figure()
    ax = plt.gca()
    plt.plot(xs[-nMax:],callback.stats[-nMax:],'ro')
    tax = ax.twinx()
    tax.plot(xs[-nMax:],callback.cluNum[-nMax:],'go')
    # tax.set_xlim(0,0.4)
    tax.set_ylim(0,25)