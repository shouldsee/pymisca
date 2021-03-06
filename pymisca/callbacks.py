
import pymisca.models as pymod
import pymisca.numpy_extra as pynp
import pymisca.ext as pyext
import copy
import functools
import pymisca.fop

import pymisca.proba    



np  = pynp.np
def _betas(i,start=0.,step=1.):
    res = start + i * step
    return res

class callback__stopAndTurn(object):
    def __init__(self, interval = 1, cluMin = 2, 
                 burnin = -1,
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
            betas = functools.partial( _betas, start=start,step=step)
#             betas = lambda i: start + i * step
        elif hasattr(betas,'__getitem__'):
            lst = betas
            betas = functools.partial(
                np.interp,
                fp=lst,
                xp=range(len(lst)),)
#             betas = getattr(betas,'__getitem__')
        elif hasattr(betas,'next'):
            betas = getattr(betas,'next')
            assert 0,'broken cuz not accepting next(i)'
                
            
        self.turning = turning
        self.betas = betas
        self.lastTurn = None
        self.right= None
        self.left = None
        self.lastResp = None
        self.stats = []
        self.betaHist = []
        self.speedHist = []
        self.cluNum = []
        self.H = []
        self.clusterH = []
#         self.mode = 'lr'
        self.mode = 'r'
        self.interval = 1
        self.mdls = []
#     def betas(self,i ):
#         res =  self.start
        
    def __getstate__(self):
        d = dict(self.__dict__)
#         del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d) # I *think* this is a safe way to do it
        
    def saveModel(self,  *args):
        iteration, weight, distributions, log_likelihood, log_proba, wresp = args
        if not iteration % self.interval:
            mdl = pymod.MixtureModel(weights=weight,
                                   dists= distributions,
                                   lastLL = log_likelihood,)
            self.mdls.append( 
                [ iteration, copy.deepcopy(mdl) ]
                      )
        return args      
    def __call__(self,*args):
#         print 'args',len(args),args
#         print 
        iteration, weight, distributions, log_likelihood, log_proba, wresp = args
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
            
            if self.lastResp is not None:
                speed = ((resp - self.lastResp)**2).sum(axis=1).mean(axis=0)
            else:
                speed = 0.
            self.speedHist.append(speed)
            self.lastResp = resp
            
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
            nMax=-1,
            xunit = None,
            XCUT=None,
            YCUT=None,
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
        betas = callback.betaHist
        
    ys = np.nan_to_num(callback.clusterH)
    axs = pyvis.qc_2var(*np.broadcast_arrays(np.array(xs)[:,None], 
                                             ys)
                       ,nMax=nMax,axs=[None,ax,None,None])

    plt.sca(axs[1])
    # plt.figure()
    ax = plt.gca()
    plt.plot( xs, np.nan_to_num( callback.stats),'ro')
    ax.set_xlim(0,None)
#     plt.plot(xs[-nMax:],callback.stats[-nMax:],'ro')
    tax = ax.twinx()
    tax.plot( xs, np.nan_to_num(callback.cluNum),'go')
#     tax.plot(xs[-nMax:],callback.cluNum[-nMax:],'go')
    # tax.set_xlim(0,0.4)
    tax.set_ylim(0,25)
    

    xtk = ax.get_xticks()
#     tay = ax.twiny()
    betaTicks = np.interp(xp=range(len(betas)),fp=betas,x=xtk)
    betaTicks = map(lambda x:'%.2E'%x,betaTicks)
    ax.set_xticklabels(pyext.paste0([xtk,betaTicks],'\n'))    
             
def qc__vmf__speed(mdl=None,
            callback = None,
            nMax=-1,
            xunit = None,
            XCUT=None,
            YCUT=None,
           ):
    def getCallback(mdl):
        callback = getattr(mdl,'callback',None)
        _class = callback__stopAndTurn
        if not isinstance(callback, _class):
            callback = None

        if callback is None:
            callback = [ x for x in mdl.callbacks 
                        if isinstance(x,_class)][0]
        return callback

#             if callback is None or not isinstance:
    if callback is None:
        assert mdl is not None
        callback  = getCallback(mdl)
                            
#     n = 4000
#     xmax = 1.0
    fig,ax = plt.subplots(1,1,figsize=[14,10])
    if xunit is not None:
        xs = getattr(callback,xunit)
    else:
        xs = np.arange(len(callback.betaHist))
        betas = callback.betaHist

    ys = np.nan_to_num(callback.clusterH)
    axs = pyvis.qc_2var(*np.broadcast_arrays(np.array(xs)[:,None], 
                                             ys)
                       ,nMax=nMax,axs=[None,ax,None,None])
        

    plt.sca(axs[1])
    # plt.figure()
    ax = plt.gca()
    plt.plot( xs, np.nan_to_num( callback.stats),'ro')
    ax.set_xlim(0,None)
    ax.set_ylim(0,None)
#     plt.plot(xs[-nMax:],callback.stats[-nMax:],'ro')
    tax = ax.twinx()
#     tax.plot(xs, callback.speedHist,'go')
    tax.plot(xs, np.nan_to_num( callback.speedHist) ,'go')
    tax.set_ylim(0,None)
    
    if mdl is not None:
        tax2 = tax.twinx()
        mdl.hist[-1] = 0.
#         tax2.plot(xs,  mdl.hist[1:], color='violet', marker='o')
        tax2.plot(xs,  mdl.hist[0:], color='violet', marker='o')
    
    ax.set_title( mdl.hist[1 + XCUT])
    
    xtk = ax.get_xticks()
#     tay = ax.twiny()
    betaTicks = np.interp(xp=range(len(betas)),fp=betas,x=xtk)
    betaTicks = map(lambda x:'%.2E'%x,betaTicks)
    ax.set_xticklabels(pyext.paste0([xtk,betaTicks],'\n'))
    
    if YCUT is not None:
        pyvis.abline(y0=YCUT,k=0,ax=ax)
    if XCUT is not None:
        pyvis.abline(x0=XCUT,ax=ax)    
    
    return [ax,tax],betaTicks
#     tax.plot(xs[-nMax:],callback.cluNum[-nMax:],'go')
    # tax.set_xlim(0,0.4)
    
import pymisca.iterative.weight__entropise

class weight__entropise(object):
    def __init__(self,beta = 1.0):
        self.beta = beta
    def __call__(self, *args):
        beta = self.beta
#     def weight__entropise(*args):
        iteration, weight, distributions, log_likelihood, log_proba, wresp = args
    #                                 , speedTol = 0.,**kwargs):
        speedTol = 0.000001
#         weight = pymisca.itera
# dirichlet__minimise__grad

#         weight = pymisca.iterative.weight__entropise.dirichlet__minimise__grad(
#             wresp,
#             speedTol=speedTol, 
#             lossTol =  1E-8,
#             maxIter=50,
#             beta = beta).last[0]

        weight = pymisca.iterative.weight__entropise.main__grad(
            wresp,
            speedTol=speedTol, 
            lossTol =  1E-8,
            maxIter=50,
            beta = beta).last[0]
        
        
#         assert 0
#         weight = pymisca.iterative.weight__entropise.main__grad(
#             [weight], 
#             speedTol=speedTol, 
#             beta = beta).last[0]

        log_likelihood = pyext.entropise(weight)[None,:] + log_likelihood

        args = list(args)
        args[1] = weight
        args[-3] = log_likelihood
        return tuple(args)
    
import pymisca.iterative.resp__entropise    
class MCE(object):
    def __init__(self,stepSize = 1.0, beta = 1.0, 
                 maxIter = 50,
                 speedTol= 1E-6,
                 lossTol =  1E-8,
                 debug = False,
                 **kwargs):
#         self.
        self._debug = debug        
        self._hist = []
        
        self.stepSize = stepSize
        self.beta = beta
        self.maxIter = maxIter
        self.speedTol = speedTol
        self.lossTol = lossTol
        for k,v in kwargs.items():
            setattr(self,k,v)
#         self.debug = debug
            
    def __call__(self, *args):
#         beta = self.beta
#     def weight__entropise(*args):
        iteration, weight, distributions, log_likelihood, log_proba, wresp = args
    #                                 , speedTol = 0.,**kwargs):
        speedTol = 0.000001

#         d  = 
        res = pymisca.iterative.resp__entropise.MCE__grad(
            wresp,
            **{k:v for k,v in vars(self).items() if not k.startswith("_")}
#             speedTol= speedTol, 
#             lossTol =  1E-8,
#             maxIter=50,
#             stepSize=self.stepSize,
#             beta = self.beta,
        )
        if self._debug:
            self._hist.append(res.hist )
        wresp = res.last
        
#         .last[0]
        


        log_likelihood = pyext.entropise(weight)[None,:] + log_likelihood

        args = list(args)
        args[-1]=  wresp
#         args[1] = weight
        args[-3] = log_likelihood
        return tuple(args)    
    
# class sample__latent(object):
class resp__sample(object):
    def __init__(self,
#                  stepSize = 1.0, beta = 1.0, 
                 n_draw = 1,
                 debug = False,
                 **kwargs):
#         self.
        self._debug = debug        
        self._hist = []
        self.n_draw = int(n_draw)
        
            
    def __call__(self, *args):
#         beta = self.beta
#     def weight__entropise(*args):
        iteration, weight, distributions, log_likelihood, log_proba, wresp = args
        
        if self._debug:
            self._hist.append(res.hist )
#         wresp = res.last
        wresp = pymisca.proba.random__categorical(wresp, n_draw=self.n_draw)
        
#         log_likelihood = pyext.entropise(weight)[None,:] + log_likelihood

        args = list(args)
        return tuple(args)        
    
class resp__MCE__surfer(object):
    def __init__(self,
                 stepSize = 0.1,
                 maxIter = 20,
                 beta = 1.0, 
#                  n_draw = 1,
                 debug = False,
                 **kwargs):
#         self.
        self._debug = debug        
        self._hist = []
        self.stepSize = stepSize
        self.maxIter = int(maxIter)
        self.beta = beta
#         self.n_draw = int(n_draw)
        
            
    def __call__(self, *args):
#         beta = self.beta
#     def weight__entropise(*args):
        iteration, weight, distributions, log_likelihood, log_proba, wresp = args
        eps = 1E-8
        res = pymisca.iterative.resp__entropise.MCE__surfer(
            wresp,
#             X0 = np.log(pynp.arr__rowNorm(wresp+eps)),
            **{k:v for k,v in vars(self).items() if not k.startswith("_")}
#                                                             stepSize=self.stepSize,
#                                                              maxIter=self.maxIter,
                                                             
                                                            )
        if self._debug:
            self._hist.append(res.hist )
        wresp = res.last
        

        args = list(args)
        args[-1] = wresp
        return tuple(args)     

class resp__random__dirichlet(object):
    def __init__(self,
                 scale=1.0,
#                  stepSize = 0.1,
#                  maxIter = 20,
#                  beta = 1.0, 
# #                  n_draw = 1,
#                  debug = True,
                 debug = False,
                 **kwargs):
#         self.
        self._debug = debug        
        self._hist = []
        self.scale = scale
#         self.stepSize = stepSize
#         self.maxIter = int(maxIter)
#         self.beta = beta
#         self.n_draw = int(n_draw)
        
            
    def __call__(self, *args):
#         beta = self.beta
#     def weight__entropise(*args):
        iteration, weight, distributions, log_likelihood, log_proba, wresp = args
        eps = 1E-16
#         wresp = pymisca.proba.random__dirichlet(wresp)        
        log_proba += -pynp.logsumexp(log_proba,axis=1)
        proba = np.exp(log_proba)
        
        p_unif = 1./ len(proba.T)
        
#         temp = 0.25
#         temp = 0.45
        temp = 0.25
    
#         temp = 1.00
#         scale = len(proba.T)
#         scale = 1.0
#         scale = 1./ (len(proba.T)**2)
    
    
        K = float(len(proba.T))
        scale = 1./ K
        
#         temp =  K / (K + 1.)
# #         print 'K=',K
#         proba = (( 1. - temp) * proba + temp*p_unif ) 
#         wresp = proba
        
    
        
# #         proba = (( 1. - temp) * proba + temp* ) 
#         proba = ( 1. * proba + K )/ (K+1) 
#         wresp = proba        
        
        temp =  0.50
#         temp =  0.00
        
        proba = (( 1. - temp) * proba + temp*p_unif ) 

        wresp = proba
#         proba 
        
        wresp = pymisca.proba.random__dirichlet(wresp * scale) 
        wresp = wresp + eps
#         + eps

#         wresp = pymisca.proba.random__dirichlet(proba * scale + eps) 

#         + eps
        
        if self._debug:
            print (np.min(wresp),np.max(wresp))

        args = list(args)
        args[-1] = wresp
        return tuple(args)   
    
class resp__momentum(object):
    def __init__(self,
#                  alpha = 0.5,
                 alpha = 0.5,
#                  scale=1.0,
#                  stepSize = 0.1,
#                  maxIter = 20,
#                  beta = 1.0, 
# #                  n_draw = 1,
#                  debug = True,
                 debug = False,
                 **kwargs):
#         self.
        self._debug = debug        
        self._hist = []
#         self.scale = scale
        self._last_resp  = None
        self.alpha = alpha
        
            
    def __call__(self, *args):
        iteration, weight, distributions, log_likelihood, log_proba, wresp = args
        K = float(len(wresp.T))
        L = len(wresp)
        eps = 1E-16
        
        alpha = self.alpha
        if self._last_resp is None:
            #### reinitialised memory to a 
#             self._last_resp = wresp * 0. + 1./ K
#             self._last_resp = wresp * 0. +  1./K
            self._last_resp = wresp * 0.
            Ka = 15
            Ka = int(K)
            self._last_resp[:,:Ka] = 1./Ka
            alpha = 0.            
            
            alpha  =1.
#             self._last_resp=  pymisca.proba.random__dirichlet(self._last_resp * K)
#             self._last_resp=  pymisca.proba.random__dirichlet(self._last_resp * K)
        else:
            alpha = alpha
            
        log_proba += - np.log(weight)
        log_proba += - pynp.logsumexp(log_proba,axis=1)
        
        proba  = proba_c = proba_current= np.exp(log_proba)
        

        if 0:
            pass
        
        if 0:
            ### using a mixutre instead of summation
    #         gamma  = 0.2
            random_proba = pymisca.proba.random__dirichlet(wresp * 0 + 1. )

            gamma = 1./K 
            gamma_ = np.random.random(size=(L,1))
            proba = random_proba * (gamma_ > gamma) + proba_current * (gamma_ <= gamma)
        
        
        ### mix at the param level
#         proba = pymisca.proba.random__dirichlet( proba_c * K + 2 )
#         proba = pymisca.proba.random__dirichlet( (proba_c + 1. )* K )
#         proba = pymisca.proba.random__dirichlet( ((proba_c  ) + 1./K) * K )
#         proba = ( ((proba_c  ) + 1./K)/2. )
#         alpha = 1.0
        alpha = self.alpha
#         alpha = self.alpha = 0.5
        proba =  ( (alpha) * (proba_c  ) + (1. - alpha) * 1./K) 
    
    
#         proba = pymisca.proba.random__dirichlet( (proba_c  )* K + 1.)
        
#         proba = (1.- alpha) * self._last_resp + ( alpha ) * ( proba  )/2.      
#         proba = (1.- alpha) * proba + ( alpha) * self._last_resp
        
        wresp = proba
        self._last_resp = wresp
                                                                             
        
        if self._debug:
            print (np.min(wresp),np.max(wresp))

        args = list(args)
        args[-1] = wresp
        return tuple(args)     
    
class resp__entropise(object):
    def __init__(self,beta = 1.0):
        self.beta = beta
    def __call__(self, *args):
        beta = self.beta
#     def weight__entropise(*args):
        iteration, weight, distributions, log_likelihood, log_proba, wresp = args
    #                                 , speedTol = 0.,**kwargs):
        speedTol = 0.000001
#         weight = pymisca.itera
# dirichlet__minimise__grad

        wresp = pymisca.iterative.resp__entropise.dirichlet__minimise__grad(
            wresp,
            speedTol=speedTol, 
            lossTol =  1E-8,
            maxIter=50,
            beta = beta).last

#         weight = pymisca.iterative.weight__entropise.main__grad(
#             wresp,
#             speedTol=speedTol, 
#             lossTol =  1E-8,
#             maxIter=50,
#             beta = beta).last[0]
        
        
#         assert 0
#         weight = pymisca.iterative.weight__entropise.main__grad(
#             [weight], 
#             speedTol=speedTol, 
#             beta = beta).last[0]
        weight = wresp.mean(axis=0)

        log_likelihood = pyext.entropise(weight)[None,:] + log_likelihood

        args = list(args)
        args[-1] = wresp
        args[1] = weight
        args[-3] = log_likelihood
        return tuple(args)

def verbose__callback( *args ):
    iteration, weight, distributions, log_likelihood, log_proba, wresp = args
#     log_likeli
#                                 , speedTol = 0.,**kwargs):
    print('[weight]',weight[:5])
    print('[log_proba]',log_proba.ravel()[:5])
    args = list(args)
    return tuple(args)
#     return iteration, weight, distributions, log_likelihood, log_proba
             