
import tensorflow_probability.python.distributions as tfdist
import pymisca.tensorflow_extra as pytf
ed = edm = pytf.ed
tf = pytf.tf;

import pymisca.util as pyutil
np = pyutil.np

from pymisca.models import BaseModel


bkd = tfdist
class matrixDecomp__VIMAP(BaseModel):
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
            matrixDecomp__VIMAP,
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
    #         self.
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
        
        bkd = self.bkd

        shape=(ncatx,ncaty)

        prior.xbase = bkd.Normal(0.,1.)
        return prior

#         prior.loc = bkd.Normal(0.,1.,shape=(ncatx,ncaty))
        
#         prior
#         bkd.Dirichlet(self.alpha)
#         self.blk.
    def make_post(self):
        D = self.D
        K = self.K
        ncatx,ncaty = self.ncatx,self.ncaty
#         alpha = self.D/2.
    #     name = 'test'
        self.post = post = pyutil.util_obj()
        self.postScope = self.get_variable_scope('post')
        
        with self.postScope:

            
#             post.locLR =  tf.get_variable(shape=(ncatx,ncaty),name='factor')
#             post.loc = 2 * tf.nn.sigmoid(
#                post.locLR
#             ) - 1.
#             post.loc = tf.nn.l2_normalize(post.locLR,axis=0)
            
            ##### Attention mechanism
            post.yconc  = tf.nn.softplus(tf.get_variable(shape=(ncaty,1),name='y_beta_conc'))             
#             init = 
#             init = tf.constant(np.random.random((ncatx,)).astype('float32')*2. -1. )
            
#             post.gate = tf.square(tf.nn.l2_normalize( gateOrig))
            
#             post.gate = tf.nn.softplus(gateOrig)
#             post.gate = tf.clip_by_value(gateOrig,-5,5)
#     
#             post.gate = tf.ones(shape=(ncatx,))
            #### Scale sigma
#             scale = 0.5            
#             post.loc = tf.get_variable(shape=)
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
        
        self.postScope = self.get_variable_scope('emission')
        
        with self.postScope:       
            scale = tf.nn.softplus( tf.get_variable(shape=(1,),name='scale') 
                                       )
            post.scale = scale  * tf.ones(shape=(ncatx,ncaty))
            post.scale = tf.clip_by_value(post.scale, 0, 2.)
            post.xcat_scale = tf.nn.softplus( tf.get_variable(shape=(ncatx,),name='xcat_scale') 
                                       )

            ncat = ncatx
            post.encoder = tf.get_variable(shape=(shape[1],ncat),name = 'encoder')
            post.decoder = tf.get_variable(shape=(ncat,shape[1]),name = 'decoder')

            post.latent = tf.matmul(X,post.encoder)
            post.locPer = tf.matmul(post.latent, post.decoder)

            post.xbase =  tf.get_variable(shape=(shape[0],1),name = 'x_intercept')            
            xmean_est = X.mean(axis=1,keepdims=1)
            x_se = X.std(axis=1,keepdims=1) / np.sqrt(shape[0])
            env.emission = self.bkd.Normal(post.locPer 
                                           + xmean_est
                                           + post.xbase * x_se, 
                                           post.scale[0,0])

#         comp = np.ravel(env.components).tolist()
#         env.emission = self.bkd.Mixture(
#             cat = env.cat, 
# #             components_distribution= comp,
#             components= comp,
# #             sample_shape=shape,
#         )        
    
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