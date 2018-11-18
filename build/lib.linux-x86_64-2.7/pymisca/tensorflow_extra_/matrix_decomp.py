
import tensorflow_probability.python.distributions as tfdist
import pymisca.tensorflow_extra as pytf
ed = edm = pytf.ed
tf = pytf.tf;

import pymisca.util as pyutil
np = pyutil.np

from pymisca.models import BaseModel


# 
# import tensorflow_probability.python.distributions as tfdist
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
                 mode=None,
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
        self.ncatx = ncatx
        self.ncaty = ncaty
        if D is not None and not debug:
            self.init_model(D=D)  
            
#     def __init__(self,D=None,K = None,alpha = 1.0):
    def init_model(self,D=None,K = None,alpha = 1.0):
        if self.initialised:
            return self

#         self.D = D = self.D if D is None else D
#         assert D is not None
#         self.K = K = self.K if K is None else K
#         assert K is not None

        prior = self.make_prior()
        post = self.make_post()

        ##### Dictonary for constructing self.emDist(**self.param)
    #         self.

        self.param_key = (self.em_key + 
                          self.mix_key)

#         self.paramDict = {getattr(prior,name,None):
#                           getattr(post,name,None) 
#                           for name in self.param_key}

#         self.paramDict = {k:v 
#                           for k,v in self.paramDict.items() 
#                           if k is not None and v is not None}

        self.initialised = True; 
        return self
    
    def make_prior(self):
        self.alpha = 1.
        ncatx,ncaty = self.ncatx,self.ncaty
        if self.debug:
            print ('[making prior]')
        self.prior = prior = pyutil.util_obj()
        
        bkd = self.bkd
        
        prior.mix_x = bkd.Dirichlet(tf.ones(ncatx) * self.alpha)
        
        prior.mix_y = bkd.Dirichlet(tf.ones(ncaty) * self.alpha)
        

        shape=(ncatx,ncaty)
#         prior.loc = bkd.Normal(tf.zeros(shape),tf.ones(shape),)
        prior.locLR  = bkd.Logistic(0.,1.)
        prior.xcat = bkd.Normal(0.,1.)
#         prior.ycat = bkd.Dirichlet(tf.ones(ncaty,dtype='float32')/ncaty * self.alpha)
        
        prior.ycat = bkd.Beta(concentration0= self.alpha * 0.01,
                              concentration1= self.alpha * 0.01)

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

            post.mix_x = tf.square(tf.nn.l2_normalize( 
                tf.get_variable(shape=(ncatx,),name = 'mix_x',)
            ))
            post.mix_y = tf.square(tf.nn.l2_normalize( 
                tf.get_variable(shape=(ncaty,),name ='mix_y')
            ))
            
            post.locLR =  tf.get_variable(shape=(ncatx,ncaty),name='factor')
            post.loc = 2 * tf.nn.sigmoid(
               post.locLR
            ) - 1.
#             post.loc = tf.nn.l2_normalize(post.locLR,axis=0)

            post.mix_xy  = post.mix_x[:,None] * post.mix_y [None]
            post.weight = post.mix_xy_flat = tf.reshape(post.mix_xy,[-1])
            
#             scale = 0.5
            scale = tf.nn.softplus( tf.get_variable(shape=(1,),name='scale') 
                                       )
    
            post.scale = scale  * tf.ones(shape=(ncatx,ncaty))
#             post.loc = tf.get_variable(shape=)
        return post       

    def make_mixture(self,X=None,env=None):
    #     K = self.K
#         N = len(X)    
        shape = X.shape
#         env = self.get_env(env)
        env = post = self.post
        mixShape   = env.mix_xy.shape.as_list()
        env.components = components = np.ndarray(dtype='O',shape=mixShape)
        for x, y in np.ndindex(tuple(mixShape)):
            pdict = {k:env[k][x,y]
                                   for k in self.em_key}
            comp = self.emDist(**pdict)
            components[x,y] = comp

        
#         env.cat = self.bkd.Categorical(
#             probs = env.weight,
#     #             sample_shape=N
#         )

#         env.emission = self.bkd.Mixture(
#             cat = env.cat, 
#             components= np.ravel(env.components).tolist(),
#     #             sample_shape=N,
#         )        
        
        with self.get_variable_scope('training'):
#             post.xcat = tf.square(tf.nn.l2_normalize( 
#                 tf.get_variable(shape=(shape[0],self.ncatx),name ='mix_x'),
#                 axis=-1,
#             ))

            post.xcat = tf.nn.softplus((
                tf.get_variable(shape=(shape[0], self.ncatx),name ='mix_x')
#                 axis=-1,
            ))
            

            post.xbase =  tf.get_variable(shape=(shape[0],1),name = 'x_intercept')

            post.ycat = (tf.nn.sigmoid( 
                tf.get_variable(shape=(shape[1],self.ncaty),name ='mix_y')
#                 axis= -1,
            ))       
            
            
#             post.ycat = tf.square(tf.nn.l2_normalize( 
#                 tf.get_variable(shape=(shape[1],self.ncaty),name ='mix_y'),
#                 axis= -1,
#             ))
            post.locPer = reduce(tf.matmul, [post.xcat, post.loc, tf.transpose(post.ycat)])

            
#         post.xcat.T[None]
            mix_xy =  post.xcat[:,None,:,None] * post.ycat[None,:,None,:]
            mix_xy = tf.reshape(mix_xy,list(shape) + [-1])        
            post.mixPer = mix_xy
            
        env.cat = self.bkd.Categorical(
            probs = mix_xy,
#     #             sample_shape=N
        )
        
#         bkd.Mixture()
#         return 
#         print 'testing'
#         return 
#         loc = tf.matmul(
#             tf.matmul(post.xcat ,post.loc)
#         loc = .matmul(tf.transpose(post.ycat))
        xbase_init = X.mean(axis=1,keepdims=1)
        env.emission = self.bkd.Normal(post.locPer 
                                       + xbase_init
                                       + post.xbase * post.scale[0,0]/np.sqrt(shape[0]), 
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
    #         loss += tf.reduce_sum(-lp_param) + tf.reduce_sum()

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
    
main= matrixDecomp__VIMAP