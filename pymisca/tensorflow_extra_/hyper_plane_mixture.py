# import models; reload(models);from models import * 

# import tensorflow_probability.python.edward2 as tfed
import tensorflow_probability.python.distributions as tfdist

import pymisca.tensorflow_extra as pytf
import pymisca.tensorflow_util as pytfu
ed = edm = pytf.ed
tf = pytf.tf;

import pymisca.util as pyutil
np = pyutil.np

from pymisca.models import BaseModel


# def GammaRadialTheta( self, 
#                       gamma_concentration = None,
#                       gamma_rate = None,
#                       vm_concentration = None,
#                       vm_direction = None,
#                       D = None
#                     ):
 
#     #### Use a bijector to calculate P(x) from P(r^2)
#     dist_xyz = mdl = pytf.AsRadialTheta(
#         distribution=mdl,
#         D=D)

def mixture_logProbComponents(self, x):
    '''Hacked to return logprob before logsumexp across components
    '''
    with tf.control_dependencies(self._assertions):
        x = tf.convert_to_tensor(x, name="x")
        distribution_log_probs = [d.log_prob(x) for d in self.components]
        cat_log_probs = self._cat_probs(log_probs=True)
        final_log_probs = [
          cat_lp + d_lp
          for (cat_lp, d_lp) in zip(cat_log_probs, distribution_log_probs)
        ]
        concat_log_probs = tf.stack(final_log_probs, -1)
#      log_sum_exp = tf.reduce_logsumexp(concat_log_probs, [0])
    return concat_log_probs
pytf.mixture_logProbComponents = mixture_logProbComponents
#     return mdl

class HyperPlaneMixture_VIMAP(BaseModel):
    
    bkd    = tfdist
    emDist = pytf.HyperPlane
    
    def __init__(self,D=None,K=20,
                 debug=False,NCORE=1,L2loss=0.,
                 meanNorm = 1,
                 variable_scope=None,
                 alpha = None,
                 nConv=  1,
                 normalize = False,
                 threshold = None,
                 weighted=  True,
                 *args,**kwargs):
        super(
            HyperPlaneMixture_VIMAP,
            self).__init__(*args,**kwargs)
        self.NCORE= NCORE
        self.K = K
        self.D = D
        self.L2loss = float(L2loss)
        self.initialised = False
        self.sess = None
        self.feed_dict = None
        self.debug = debug
        self.meanNorm  = meanNorm
        self.nConv = nConv
        self.emission = None
        self.normalize = normalize
        self.threshold = threshold
        self.weighted = weighted
#         if alpha is None:
        if alpha is None:
            alpha = float(self.K)
        self.alpha = float(alpha)
        if variable_scope is None:
            self.variable_scope = self.getScope()
        else:
            self.variable_scope =  variable_scope
        if D is not None:
            self.init_model(D=D)  
                        
    em_key =[
        'L2loss',
        'mean',
        'bias',
        ]
    mix_key = [
            'weight',
        ]
    def getScope(self):
        try:
            tf.get_variable(self.name+'/post',[1])
            reuse = None
        except:
            reuse = True
        print ('reuse',reuse)
        variable_scope =  tf.variable_scope(self.name, reuse=reuse)
        return variable_scope
    
    def random(self,size):
        '''
        random initialiser
'''
        
        return np.random.normal(0.,1.,size=size).astype(np.float32)
    


        
    def make_prior(self):
        D = self.D
        K = self.K
        alpha = self.D/2.
#         diriAlpha = 1.
        diriAlpha = self.alpha
#         diriAlpha = 1.
#         diriAlpha = 0.001
#         diriAlpha = 0.00001
#         diriAlpha = 0.0000000000000000000000000000000001        
#         diriAlpha = 10.

        name = self.name
        self.prior = prior = pyutil.util_obj()

        with self.variable_scope:
            
            uspan = [-1E5,1E5]
            ##### Prior
#             prior.gamma_concentration = edm.Normal(tf.zeros(D), tf.ones(D), sample_shape=K)            
#             prior.loc =  edm.Uniform(*uspan,sample_shape=(K,D))

            prior.weight = pi = tfdist.Dirichlet( float(diriAlpha) * tf.ones(K)/float(K) )            
#             prior.weight = 
#             prior.cat = edm.Categorical(weight = post.weight)
        return prior
    
    def make_post(self):
        D = self.D
        K = self.K
        alpha = self.D/2.
        name = self.name
        self.post = post = pyutil.util_obj()
        prior = self.prior

        with self.variable_scope:
            
            uspan = [-1E5,1E5]
            ##### Posterior
            i = -1
            i += 1
#             post.weight =  tf.ones(shape=[K,]) * 1.
            if self.weighted:
                post.weight = pytfu.getSimp_(shape=[K],name = 'weight',method='expnorm')
            else:
                post.weight = tf.constant([1.] * K, name ='weight')

            
            i += 1            
#             post.mean = tf.get_variable(str(i), shape =[K,self.D])
#             post.mean = pytfu.getSimp_(shape=[K,self.D],name = str(i),method='l2norm')
    
#             if self.normalize:
# #                 post.mean = pytfu.getSimp
#                 post.mean = tf.get_variable(str(i), shape =[K,self.D])
#                 meanSq  = tf.square(post.mean)
#                 l2_mean = tf.reduce_mean(meanSq,
#                                       axis=-1,keepdims=True) 
#                 if self.meanNorm:
#                     ### Make sure each signature sums to zero
#                     post.mean = post.mean - tf.reduce_mean(post.mean,axis=-1,keepdims=True)
#                 post.mean = meanSq/tf.sqrt(l2_mean)
#                 post.mean = tf.abs(post.mean)
#                 #### make sure sq sum to 1.
#             else:
#             if self.normalize:
            if 1:
                post.mean = tf.get_variable(str(i), shape =[K,self.D])
                if self.meanNorm:
                    ### Make sure each signature sums to zero
                    post.mean = post.mean - tf.reduce_mean(post.mean,axis=-1,keepdims=True)
                l2_mean = tf.reduce_sum(tf.square(post.mean),
                                      axis=-1,keepdims=True) 
                post.mean = post.mean/tf.sqrt(l2_mean)
#                 if self.normalize
#                 post.L2loss = tf.ones(shape=[K,]) * self.L2loss
            if self.threshold is not None:
                post.mean  = tf.concat([post.mean[:-1], 
                                        tf.zeros(shape=[1,self.D])],axis=0)
                post.bias = tf.constant( [0.] * (self.K - 1) + [self.threshold], )
    
#             i += 1
#             post.vm_direction = edm.PointMass(
#                 tf.nn.l2_normalize(
#                     tf.get_variable(str(i), [K,D]),
#                     axis = -1,
#                     name = "vm_direction",
#                 ),
#             )
            
#             post.rate  = edm.PointMass(
#                 tf.nn.softplus(
# #                     tf.Variable(name="rate",initial_value = self.random([K]) ),
#                     tf.get_variable('rate',shape=[K,])
#                               ),
#             )
        return post
    
    def init_model(self,D=None,K = None,alpha = 1.0):
        if self.initialised:
            return self
        
        self.D = D = self.D if D is None else D
        assert D is not None
        self.K = K = self.K if K is None else K
        assert K is not None
        
        prior = self.make_prior()
        post = self.make_post()
        
        ##### Dictonary for constructing self.emDist(**self.param)
#         self.

        self.param_key = (self.em_key + 
                          self.mix_key)

        self.paramDict = {getattr(prior,name,None):
                          getattr(post,name,None) 
                          for name in self.param_key}
        
        self.paramDict = {k:v 
                          for k,v in self.paramDict.items() 
                          if k is not None and v is not None}
        
        
        ### Prior components
        cDicts = [
            {key: v[k] 
             for key,v in prior.__dict__.items() 
             if key in self.em_key} 
            for k in range(K)]
#         self.prior.components = [self.emDist(**d) for d in cDicts]

        
        ### Posterior generative
#         edm.Mixture
        cDicts = [
            {key: v[k] 
             for key,v in post.__dict__.items() 
             if key in self.em_key} 
            for k in range(K)]
        const  = {'normalize':self.normalize}
        [d.update(const) for d in cDicts]
        self.post.components = [self.emDist(**d) for d in cDicts]
        

    
        self.initialised = True; return self
    
    def build_likelihood(self,X,env=None):
        if env is None:
            env = self.prior
        elif isinstance(env,str):
            env = getattr(self,env)
        K = self.K
#         N = len(X)
#         env,cat = bkd

        ### build 
        env.cat = self.bkd.Categorical(
            probs = env.weight,
#             sample_shape=N
        )
        
#         env.cat = edm.PointMass(
#             tf.nn.softmax(
#                 tf.get_variable("cat",[N,K]),
# #                     tf.Variable(name="q_pi",initial_value = self.random([K]) ),
#             )
#         )            


        env.emission = self.bkd.Mixture(
            cat = env.cat, 
            components=env.components,
#             sample_shape=N,
        )       
        self.emission = env.emission
        return env.emission
    
    def _fit(self,X, 
             n_iter = 1000, 
             n_print=100, 
             env=None,
             optimizer = None, 
             **kwargs):
        
        K = self.K
        N = len(X)
        D = X.shape[-1]
        if self.D is not None:
            assert D == self.D
        else:
            self.D = D
#         emission = self.build_likelihood(X,env=env)
        self.init_model()
    
        mdl, (last_vars, hist_loss, opt) = self._fit_MAP(
            X,
            MAX_ITER = n_iter,
            optimizer = optimizer,
            **kwargs)
        
        return hist_loss
        


    @property
    def means_(self):
        res = self.x_post.mean().eval()
        return res
    @property
    def covariances_(self):
        res = self.x_post.covariance().eval()
        return res
    @property
    def weights_(self):
        res = self.post.weight.eval(session=self.sess)
        return res
    def _predict_proba(self,X, N=None, norm = 0, nConv = None):
        ''' self.emission does not work, manually build posterior
'''
        assert self.sess is not None,"\"self.sess\" is None, please fit the model first with a tf.Session()"
    
        N = len(X)
#         X_bdc = self.expand_input(X)
        

#         ll = tf.concat([ comp.log_prob(X)[:,None]
#                         for comp in self.post.components],axis=1) 
#         ll = ll + tf.log( self.post.weight )
        ll = self.getProba(X,nConv = nConv)
##         ll = tf.reduce_mean(ll,axis=1)  ### over posterior samples
    #     ll = tf.reduce_sum(ll,axis=-1)  ### over dimensions
        logP = ll.eval(session=self.sess,
                      feed_dict=self.feed_dict)   
        return logP
    
#     import tensorflow as tf
    def freeVarDict(self, env):
        if isinstance(env,dict):
            idict = env
        else:
            if isinstance(env,str):
                env = getattr(self,env)
            else:
                raise Exception('"env" not recognised:%s'%env)
            idict = env.__dict__
        odict = {k:
#                  x.value() ### for edward.PointMass()
                 x
                 for k,x in 
                       idict.iteritems() 
                       if not isinstance(x,list) and k in self.param_key}            
        return odict
        
    def getProba(self,x,nConv= None):
        if self.emission is None:
            self.emission  = mdl = self.build_likelihood(X=x,env='post')
        if nConv is None:
            nConv = self.nConv
            
#             self.lp_data = 
        proba = pytf.mixture_logProbComponents(self.emission,x)
        if nConv > 1:

            im  =  tf.expand_dims( proba, axis = 0)
            fir =  tf.tile([tf.eye(self.K,)],(nConv,1,1))/float(nConv)
#                 assert 0
            proba = tf.nn.convolution(
                im,fir,
#                     tf.ones((self.nConv,self.K,self.K)) / float(self.nConv),
                padding= 'SAME',
                strides=None,
                dilation_rate=None,
                name=None,
                data_format=None
            )
            proba = proba[0]
#         proba = lp_data
        return proba
            
    def _fit_MAP(self,x_obs, optimizer = None, batchMaker = None, nConv = None, **kwargs):
        if batchMaker == 'auto':
            batchMaker = pytfu.batchMaker__random(batchSize=500, 
#                                                         windowSize=20,
                                                 )
#             batchMaker = pytfu.batchMaker__randomWindow(batchSize=100, 
#                                                         windowSize=20,)
            
#         if nConv is None:
#             nConv = self.nConv
        with self.variable_scope:
#             self.emission  = mdl = self.build_likelihood(X=x_obs,env='post')
#             self.emission = 
    #         x_place = tf.placeholder(dtype='float32')
            x_place = tf.placeholder(shape=(None, ) + x_obs.shape[1:],dtype='float32')


            ### prior likelihood
            self.lp_param = lp_param = [
                tf.reduce_sum(
                    k.log_prob(v)  ### ed.RandomVariable.value()
                ) 
                if k.__class__.__name__ != 'Uniform' 
                else 0.
                for k,v in self.paramDict.items()]
    #         print (tf.reduce_sum(lp_param))
            ### data likelihood
#             self.lp_data = lp_data = mdl.log_prob(x_place)
            self.proba = self.getProba(x_place, nConv = nConv)
            self.lp_data = tf.reduce_logsumexp(self.proba, [-1])
            lp = tf.reduce_sum(
                map(tf.reduce_sum,[self.lp_param,
                                   self.lp_data])
            )
            loss = -lp
    #         loss = 0.
    #         loss += tf.reduce_sum(-lp_param) + tf.reduce_sum()

            self.feed_dict = {x_place:x_obs}
    #         fitted_vars_dict = {k:x.value() for k,x in 
    #                        self.post.__dict__.iteritems() 
    #                        if not isinstance(x,list) and k in self.param_key}
            self.sess = pytf.newSession(NCORE=self.NCORE)
            sess, last_vars, hist_loss, opt = pytf.op_minimise(
                loss,
                self.freeVarDict('post').values(),
                optimizer = optimizer,
                feed_dict = self.feed_dict,
                sess = self.sess,
                batchMaker = batchMaker,
                **kwargs

            )
            post = self.getPost()
    #         self.sess = sess
        return self.emission,(last_vars, hist_loss, opt)

    def getPost(self):
        '''Move to tfModel in the future'''
        assert self.sess is not None
        with self.sess.as_default():
            post = pyutil.util_obj(**{k:pytf.quick_eval(v) for k,v in self.params.items()})
            self.post.__dict__.update(post.__dict__)
            return self.post
        
    @property
    def params(self): 
        '''[FRAGILE] move distributions to a separate dict'''
        d = {}
        for k,v in self.post.__dict__.items():
            if not (isinstance(v, tfdist.Distribution) or  isinstance(v,list)):
                d[k] = v
        return d
        
#         params = {k:pytf.quick_eval(self.post[k])
#                 for k in self.post.__dict__ 
#                 if k != 'components'}
#         return params
main = HyperPlaneMixture_VIMAP
    