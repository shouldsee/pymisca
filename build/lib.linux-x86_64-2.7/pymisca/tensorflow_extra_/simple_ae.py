
import tensorflow_probability.python.distributions as tfdist
import pymisca.tensorflow_extra as pytf
ed = edm = pytf.ed
tf = pytf.tf;

import pymisca.util as pyutil
np = pyutil.np

from pymisca.models import BaseModel

import tensorflow_probability.python.bijectors as tfbjt

bkd = tfdist
class simpleAE__VIMAP(BaseModel):
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
            simpleAE__VIMAP,
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
            
#             scale = tf.nn.softplus( tf.get_variable(shape=(1,),name='scale') 
#                                        )
            scale = float(0.9) 
            post.scale = scale  * tf.ones(shape=(ncatx,ncaty))
#             post.scale = tf.clip_by_value(post.scale, 0, 2.)
#             post.scale = tf.ones((1.))
            post.xcat_scale = tf.nn.softplus( tf.get_variable(shape=(ncatx,),name='xcat_scale') 
                                       )

            ncat = ncatx
            post.encoder = tf.get_variable(shape=(shape[1],ncat),name = 'encoder')
            post.encoder = post.encoder - tf.reduce_mean(post.encoder,axis=0,keepdims=True)
            post.decoder = tf.get_variable(shape=(ncat,shape[1]),name = 'decoder')
            
            if 1:
                post.decoder = tf.nn.l2_normalize(post.decoder, axis=1)
#                 post.decoder = tf.square(post.decoder) 




            
#             * float(shape[1])
#             post.encoder =  (post.encoder)

#             prior.encoder = bkd.Normal(0.,1.)
#             post.decoderScale = tf.nn.softplus( tf.get_variable(shape=(1,),name='de_scale') 
#                                        )
#             post.decoderScale  = tf.ones((1,)) * float(1.)
#             prior.decoder = bkd.Normal(0.,post.decoderScale)

            post.latent = tf.matmul(X,post.encoder)
#         * 0.5 * float(ncat)
#         * 5.
        
#             noise = tf.random_normal(post.latent.shape) * 5.
#             noise = tf.random_normal(post.latent.shape) * post.noise_scale
    
#             post.latent = post.latent  + noise
#             post.noise_scale = tf.square(tf.nn.l2_normalize(
#                 tf.get_variable(shape=(1,self.ncatx),name='noise_scale')
#             )
#             )
#             post.signal_scale = 1. - post.noise_scale
#             post.signal_scale = tf.minimum(1.,post.noise_scale * ncat)
#             prior.sigraw = bkd.Normal(0.,1.)
#             post.sigraw = tf.get_variable(shape=(1,ncat-1),name = 'signal_scale')
#             post.signal_scale = tf.nn.sigmoid(post.sigraw)

#             post.signal_scale = tf.exp(-tf.nn.softplus(post.sigraw))
            alpha = 0.01
#             prior.signal_scale = bkd.Beta(1., alpha/float(ncat))
#             alpha = 1.
#             prior.signal_scale = bkd.Beta(alpha/float(ncat),
#                                           alpha/float(ncat))
            
#             #### simple convolution
#             post.latent = (post.latent[:,1:] *  post.signal_scale 
#                            + post.latent[:,:-1] * (1-post.signal_scale)
#                           )
            nlat = ncat
            post.latentScaleNorm = pytf.getSimp_(shape=(nlat,),
                                             name='post_latentScale',
#                                                  method='expnorm',
                                             method='expnorm',
#                                                  method='l2norm',
                                                )
#         [None] * tf.ones((shape[0],1))
#             post.latentScale = (tf.ones((1,))/ float(self.ncatx) ) 
#             post.latentScale = post.latentScale * float( 1200. ) 

#             dsize = float(2.)
#             dsize = float(0.5)
#             dsize = float(5)
#             dsize = float(20.)
#             dsize = float(0.00001)
#             dsize = float(30.)
#             dsize = float(20.)
            dsize = float(20.)
#             dsize = float(60.)
#             dsize = float(20.)
#             dsize = float(.1)
#             dsize = float(0.5)

#             conc = float(0.000000001)
#             conc = float(1E-6)
            conc = float(1E0)
            dist = bkd.Dirichlet([conc/float(nlat)]*(nlat) )
            bjt = tfbjt.AffineScalar(scale=dsize) 
            dist = bkd.TransformedDistribution(bijector=bjt,
                                               distribution=dist)
            
            
            post.latentScaleBase = (post.latentScaleNorm)* dsize            
            post.latentScale = post.latentScaleBase
            
#             bjt = tfbjt.Square()
#             dist = bkd.TransformedDistribution(bijector=bjt,distribution=dist)
#             post.latentScale = tf.square(post.latentScale)             
            
            bjt = tfbjt.Exp()
            dist = bkd.TransformedDistribution(bijector=bjt,distribution=dist)
            post.latentScale = tf.exp(post.latentScale)             
            
            shift = -1.0
            bjt = tfbjt.AffineScalar(shift=shift,)
            dist = bkd.TransformedDistribution(bijector=bjt,distribution=dist)                        
            post.latentScale = post.latentScale + shift            
        
            prior.latentScale = dist
#             prior.latentScaleBase = dist
            post.latMean = tf.get_variable(name='latMean',shape=(nlat,))
            prior.latent = bkd.Normal(post.latMean, post.latentScale)
            post.latent = post.latent[:,:]
        
#             prior.latentDiff = bkd.Normal(   0. ,   
#                                           post.latentScale)
#             post.latentDiff = (post.latent[:,1:]
#                                - post.latent[:,:-1] )
    
#                                              tf.exp(post.latentScale))
    
#             prior.latentDiff = bkd.Normal(post.latentScale,float(1.))
#             post.latentAvg = post.latent - tf.reduce_mean(post.latent,axis=-1,keepdims=True)
#             prior.latentAvg = bkd.Normal(0., 3 * post.latentScale)

#             post.latent=post.latentAvg
    
    
#             fir =  tf.get_variabler
#             fir = tf.nn.l2_normalize(post.sigraw[0,:3,None,None])

#             #####=====
#             nf = 5
#             prior.fir = bkd.Dirichlet(concentration=[float(nf)] * nf)
#             post.fir = pytf.getSimp_(shape=(1,nf),method='expnorm',name='conv') * tf.ones(shape=(shape[0],1))
#             post.latent = tf.nn.conv1d(post.latent[:,:,None],
#                                        filters=post.fir[0,:,None,None],padding='SAME',stride=1,)[:,:,0]
#             #####-----
            
#             prior.crossprod = bkd.Normal(0.,5.)
#             post.crossprod = (tf.matmul(tf.transpose(post.latent),post.latent))/float(shape[0])
            
#             lst = []
#             curr = post.latent[:,0:1]
#             lst += [curr]
#             for i in range(1,ncat):
#                 decay = post.signal_scale[:,i:i+1]
#                 new   = post.latent[:,i:i+1]
#                 curr  = new * decay + curr * (1-decay) #### prefer decay=0
#                 lst  += [curr]
#             post.latent = tf.concat(lst,axis=1)

#             post.signal_scale = prior.sigraw.cdf(post.sigraw)

            #####=====
#             MEAN =  tf.reduce_mean(post.latent,axis=0,keepdims=True)
#             latentRMSD = tf.sqrt(tf.reduce_sum(tf.square(post.latent- MEAN),
#                                                axis=0,keepdims=True))
#             post.noise_scale = pytf.getSimp_(shape=(1,ncat),
#                                         method = 'expnorm',
#                                         name = 'noise_scale')
#             post.signal_scale = post.noise_scale
#             noise = tf.random_normal(post.latent.shape)  * latentRMSD
#             post.latent = post.latent *  post.signal_scale  + noise * (1. - post.signal_scale)
            ####-----
            

        
            ##################
            if 0:
                lst = [] 
                for i in range(ncat):
                    cat = bkd.Categorical(probs=(0.5,0.5),)
                    loc,scale =  post.latentLoc[i],post.latentScale[min(i,post.latentScale.shape[0]-1)]
                    comp = [
                        bkd.Normal(loc,scale),
                        bkd.Normal(-loc,scale),
                    ]
                    pDim = bkd.Mixture(cat=cat,
                                components=comp)
                    lst.append(pDim)
                prior.latent = pytf.JointDist(subDists=lst)


#             prior.latent = self.bkd.Mixture(
#                 cat = env.cat, 
#                 components= comp,
#         )        
            ###############
#             prior.latSquare = bkd.Dirichlet([1./float(ncat)]*ncat)
#             post.latSquare = tf.square(tf.nn.l2_normalize(post.latent,axis=-1))
#             post.latScale = 
#             x =post.latentScale[0]
            x = 4.
#             x  = 1./x
#             prior.latentMod = bkd.Normal(0.,x/6.)

#             prior.latentMod = bkd.Normal(0.,x/8.)
            post.latentMod = tf.floormod( post.latent + x/2. ,x) - x/2.
            post.latentRMSD = tf.sqrt( tf.reduce_sum(tf.square(post.latent),axis=0))
#             post. latentDistRMSD = tf.square(tf.nn.l2_normalize(post.latentRMSD))        
#             prior.latentDistRMSD = bkd.Dirichlet([1./float(ncat)]*ncat)
            
#             prior.latentMod = bkd.Normal(x/2.,x/8.)
#             post.latentMod = tf.floormod( post.latent ,x)
#             prior.latentMod = bkd.Uniform(0.,x)
#             post.latentMod = tf.floormod( post.latent ,x) 
        
#             prior.latentMod = bkd.Normal(x/2.,x/8.)

#             post.latentMod = tf.floormod( post.latent/x + 0.5, 1) -0.5
#             prior.latentMod =  bkd.Normal(0.,1./4)
            
            
            
            
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
    
main = simpleAE__VIMAP