
import tensorflow_probability as tfp
import tensorflow_probability.python.distributions as tfdist
import pymisca.tensorflow_extra as pytf
ed = edm = pytf.ed
tf = pytf.tf;

import pymisca.util as pyutil
np = pyutil.np

from pymisca.models import BaseModel

import tensorflow_probability.python.bijectors as tfbjt

bkd = tfdist
class affineGaussian__VIMAP(BaseModel):
    bkd    = tfdist
    emDist = tfdist.Normal
    
    em_key =[
        'loc',
        'scale',
        ]
    mix_key = [
            'mix_x',
            'mix_y'
        ]
    
    def __init__(self,D=None,K=20,
                 debug=False,NCORE=1,alpha = None,
                 ncatx = 1,
                 ncaty = 1,
                 mode = 'left',
                 diri = 1,
                 transpose=0,
                 *args,**kwargs):
        super(
            affineGaussian__VIMAP,
            self).__init__(*args,**kwargs)
        self.NCORE= NCORE
        self.K = K
        self.D = D
        self.initialised = False
        self.sess = None
        self.feed_dict = None
        self.debug = debug
        self.alpha = 0.1 if alpha is None else alpha
        self.ncaty = self.ncatx = ncatx
        self.mode = mode
        self.diri = diri
        self.transpose= transpose
#         self.ncatx = ncatx
#         self.ncaty = ncaty
        if D is not None and not debug:
            self.init_model(D=D)  
            
#     def __init__(self,D=None,K = None,alpha = 1.0):
    def init_model(self,D=None,K = None,alpha = 1.0):
        if self.initialised:
            return self
        prior = self.make_prior()
        post = self.make_post()
        ##### Dictonary for constructing self.emDist(**self.param)
        self.param_key = (self.em_key + 
                          self.mix_key)

        self.initialised = True; 
        return self
    
    def make_prior(self):
        self.alpha = 1.
        ncatx,ncaty = self.ncatx,self.ncaty
        if self.debug:
            print ('[making prior]')
        self.prior = prior = pyutil.util_obj()        
        
        return prior
    
    def make_post(self):
        D = self.D
        K = self.K
        ncatx,ncaty = self.ncatx,self.ncaty
#         alpha = self.D/2.
    #     name = 'test'
        self.post = post = pyutil.util_obj()
        self.postScope = self.get_variable_scope('post')
        
        return post       

    def make_mixture(self,X=None,env=None):
    #     K = self.K
#         N = len(X)    
        shape = X.shape
#         env = self.get_env(env)
        env = post = self.post
        prior = self.prior 
        ncaty = self.ncaty
        ncatx = self.ncatx
        
        self.postScope = self.get_variable_scope('mixed')
        
        with self.postScope:       
            
            ncat = ncatx
#             scale = float(0.9) 
            scale = tf.nn.softplus( tf.get_variable(shape=(1,),name='scale') 
                                       )
            
            post.scale = scale  * tf.ones(shape=(ncatx,ncaty))
            post.scale = tf.clip_by_value(post.scale, 0, 2.5)
#             post.scale = tf.clip_by_value(post.scale, 0, 0.5*2.5)
#             post.scale = tf.ones((1.))
#             post.xcat_scale = tf.nn.softplus( tf.get_variable(shape=(ncatx,),name='xcat_scale') 
#                                        )
    

#             post.encoder = tf.get_variable(shape=(shape[1],ncat),name = 'encoder')
#             post.encoder = post.encoder - tf.reduce_mean(post.encoder,axis=0,keepdims=True)
#             post.decoder = tf.get_variable(shape=(ncat,shape[1]),name = 'decoder')
            
#             if 1:
#                 post.decoder = tf.nn.l2_normalize(post.decoder, axis=1)
            

#             post.latent = tf.matmul(X,post.encoder)

# #             post.signal_scale = tf.exp(-tf.nn.softplus(post.sigraw))
#             alpha = 0.01
# #             prior.signal_scale = bkd.Beta(1., alpha/float(ncat))
# #             alpha = 1.
# #             prior.signal_scale = bkd.Beta(alpha/float(ncat),
# #                                           alpha/float(ncat))
            
# #             #### simple convolution
# #             post.latent = (post.latent[:,1:] *  post.signal_scale 
# #                            + post.latent[:,:-1] * (1-post.signal_scale)
# #                           )
#             nlat = ncat
#             post.latentScaleNorm = pytf.getSimp_(shape=(nlat,),
#                                              name='post_latentScale',
# #                                                  method='expnorm',
#                                              method='expnorm',
# #                                                  method='l2norm',
#                                                 )
# #         [None] * tf.ones((shape[0],1))
# #             post.latentScale = (tf.ones((1,))/ float(self.ncatx) ) 
# #             post.latentScale = post.latentScale * float( 1200. ) 

# #             dsize = float(2.)
# #             dsize = float(0.5)
# #             dsize = float(5)
# #             dsize = float(20.)
# #             dsize = float(0.00001)
# #             dsize = float(30.)
# #             dsize = float(20.)
#             dsize = float(20.)
# #             dsize = float(60.)
# #             dsize = float(20.)
# #             dsize = float(.1)
# #             dsize = float(0.5)

# #             conc = float(0.000000001)
# #             conc = float(1E-6)
#             conc = float(1E0)
#             dist = bkd.Dirichlet([conc/float(nlat)]*(nlat) )
#             bjt = tfbjt.AffineScalar(scale=dsize) 
#             dist = bkd.TransformedDistribution(bijector=bjt,
#                                                distribution=dist)
            
            
#             post.latentScaleBase = (post.latentScaleNorm)* dsize            
#             post.latentScale = post.latentScaleBase
            
#             bjt = tfbjt.Square()
#             dist = bkd.TransformedDistribution(bijector=bjt,distribution=dist)
#             post.latentScale = tf.square(post.latentScale)             
#             prior.encoder = tfdist.Normal([0.]*ncat,[1.]*ncat)

#             post.latent =  tf.get_variable(shape=(shape[0],ncat),name='latent')
            post.encoder = tf.get_variable(shape=(ncat,shape[1]),name = 'encoder')

#             bdist = dist = tfdist.Normal([0.]*ncat, [1.]*ncat)
            post.beta = tf.exp(tf.get_variable(shape=(ncat,),name = 'beta'))
    
            baseDist = dist = tfdist.BetaWithSoftplusConcentration(
                concentration0=post.beta,
                concentration1=post.beta,
                )
        
            bjt = tfp.bijectors.AffineScalar(shift=-0.5)
            baseDist = tfdist.TransformedDistribution(distribution=baseDist,bijector=bjt)

            post.latent =  tf.sigmoid(
                tf.get_variable(shape=(shape[0],ncat),name='latent')
#                 ,0,1
            )  - 0.5
            
#             post.latent =  tf.nn.sigmoid(
#                 tf.get_variable(shape=(shape[0],ncat),name='latent'),
#             )  - 0.5
            
#             baseDist = dist = tfdist.Normal([0.]*ncat, [1.]*ncat)
#             post.latent =  tf.get_variable(shape=(shape[0],ncat),name='latent')


            bjt = pytf.NonSquareLinearTransform(weights=post.encoder)
            prior.locPer = dist \
                = bkd.TransformedDistribution(bijector=bjt,distribution=baseDist)
            post.locPer = tf.matmul(post.latent,post.encoder)

#             post.latentScale = tf.exp(post.latentScale)             
            
#             shift = -1.0
#             bjt = tfbjt.AffineScalar(shift=shift,)
#             dist = bkd.TransformedDistribution(bijector=bjt,distribution=dist)                        

# #             x = 4.
            
#             bjt = tfp.bijectors.Affine()            
#             prior.locPer = bkd.TransformedDistribution(bijector=bjt,distribution=dist,)
#             post.locPer = tf.matmul()
#             post.locPer = tf.matmul(post.latent, post.encoder)
            
            
            #### Centering at Zero
#             prior.xbase = bkd.Normal(0.,1.)
#             post.xbase =  tf.get_variable(shape=(shape[0],1),name = 'x_intercept')            
            
            xmean_est = X.mean(axis=1,keepdims=1)
            x_se = X.std(axis=1,keepdims=1) / np.sqrt(shape[0])
            
            env.emission = self.bkd.Normal(post.locPer 
                                           + xmean_est
#                                            + post.xbase * x_se
                                           , 
                                           
                                         post.scale[0,0])
        return env.emission
    
    def get_variable_scope(self,ext, name=None):
        if name is None:
            name = self.name
        scope_name = name+'/%s'%ext
        try:
            tf.get_variable(scope_name,[1])
            reuse = None
        except:
            reuse = True
        print ('reuse',reuse)
        scope = tf.variable_scope(scope_name, reuse=reuse)
        return scope
    
    def get_env(self,env=None):
        if env is None:
            env = self.prior
        elif isinstance(env,str):
            env = getattr(self,env)
        return env
    
    def _fit(self,X, 
             n_iter = 1000, 
             n_print=100, 
             env=None,
             optimizer = None, 
             **kwargs):
        
        K = self.K
        N = len(X)
#         emission = self.build_likelihood(X,env=env)
        self.init_model()
    
        mdl, (last_vars, hist_loss, opt) = self._fit_MAP(
            X,
            MAX_ITER = n_iter,
            optimizer = optimizer,
            **kwargs)
        
        return hist_loss
        
#     _class = matrixDecomp__VIMAP





# _class.build_likelihood = make_mixture


    def _fit_MAP(self,x_obs, optimizer = None, **kwargs):
        self.init_model()
        mdl = self.make_mixture(X=x_obs,env='post')
        x_place = tf.placeholder(dtype='float32')

        ### prior likelihood
        self.lp_param = lp_param = [
            tf.reduce_sum(
                prv.log_prob(self.post[k])  ### ed.RandomVariable.value()
            ) 
            if prv.__class__.__name__ != 'Uniform' 
            else 0.
            for k,prv in self.prior.__dict__.items()]
    #         print (tf.reduce_sum(lp_param))
    
        ### data likelihood
        self.lp_data = lp_data = tf.reduce_sum(
            mdl.log_prob(x_place)
        )
        lp = tf.reduce_sum(
            map(tf.reduce_sum,[
                lp_param,
                lp_data])
        )
        loss = -lp
    #         loss = 0.
    #         loss += tf.reduce_sum(-lp_param) + tf.reduce_sum()s

        self.feed_dict = {x_place:x_obs}
    #         fitted_vars_dict = {k:x.value() for k,x in 
    #                        self.post.__dict__.iteritems() 
    #                        if not isinstance(x,list) and k in self.param_key}
        self.sess = pytf.newSession(NCORE=self.NCORE)

#         tf.global_variables_initializer().run()
        
        sess, last_vars, hist_loss, opt = pytf.op_minimise(
            loss,
            self.freeVarDict('post').values(),
            optimizer = optimizer,
            feed_dict = self.feed_dict,
            sess = self.sess,
            **kwargs

        )
    #         self.sess = sess
        return mdl,(last_vars, hist_loss, opt)

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
    
main = affineGaussian__VIMAP