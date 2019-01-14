# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The Normal (Gaussian) distribution class."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# del absolute_import
from pymisca.tensorflow_extra_.tfp_transformed_distribution import TransformedDistribution


import math


from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.ops.distributions import special_math


__all__ = [
        "Normal",
        "NormalWithSoftplusScale",
]

import tensorflow as tf
def newSession(NCORE=4):
    config = tf.ConfigProto()
    if NCORE > 1:
        config.intra_op_parallelism_threads = NCORE
        config.inter_op_parallelism_threads = NCORE
    sess = tf.Session(config=config)
    return sess

def op_minimise(loss,
                free_params,                
                optimizer=None,
                sess = None,
                MAX_ITER = 4000,
#                 LEARNING_RATE = 0.1,
                TOL_LOSS = 1e-2,
                TOL_PARAM = 1e-8,
                TOL_GRAD = 1e-8,
                feed_dict= {},
                variable_scope= None,
                batchMaker = None,
                autoStop = True,
                ):
    '''
    #### Adapted from: http://kyleclo.github.io/maximum-likelihood-in-tensorflow-pt-1/
'''
    if batchMaker is False:
        batchMaker = None
#     if variable_scope is None:
#         name = 'op_minimise'
#         variable_scope = tf.variable_scope(name, reuse=False)
#     else:
#         pass
#     with variable_scope:
    if 1:
        if optimizer is None:
            LEARNING_RATE = 1.0
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=LEARNING_RATE)
        train_op = optimizer.minimize(loss=loss)
        grad = tf.gradients( loss,free_params)
        grad = [x for x in grad if x is not None]

        if sess is None:
            sess = tf.get_default_session()
            if sess is None:
                sess = tf.Session()

        def getFeedDict(i):
            if batchMaker is not None:
                feed_curr = {k:batchMaker(v,i) for k,v in feed_dict.iteritems()}
            else:
                feed_curr = feed_dict
            return feed_curr
        if 1:
    #     with tf.Session() as sess:

                # initialize
            sess.run(fetches=tf.global_variables_initializer())
            
            
#             i = 0
#             obs_loss = sess.run(fetches=[loss], feed_dict=getFeedDict(0))
#             obs_vars = sess.run(fetches=free_params)
        
            obs_loss = []
            obs_vars = []
            for i in range(MAX_ITER):
                feed_curr = getFeedDict(i)
                
               # gradient step
                sess.run(fetches=train_op, feed_dict=feed_curr)    
                new_loss = sess.run(fetches=loss, feed_dict=feed_curr)
    #             if grad is not None:
                new_grad = sess.run(fetches=grad, feed_dict=feed_curr)
                obs_loss.append(new_loss)
                if i!=0:
                    loss_diff = abs(new_loss - obs_loss[-1])

                    if not i%100:
                        print ('Iter %d'%i, new_loss)
                    if autoStop:
                        if i > 50:
            #             if max(obs_loss[-10:]) - TOL_LOSS < new_loss:
                            minA = min(obs_loss[-20:])
                            minB = min(obs_loss[-10:])
            #                 if minA + TOL_LOSS < minB:
                            if minB - minA > TOL_LOSS:
                                    print('Loss function converged\
                                    in {} iterations!: {}'.format(i,new_loss))
                                    print( minA,minB)
                                    break

            last_vars = sess.run(fetches=free_params)

    return sess,last_vars, obs_loss, optimizer





def bulkTransform(linop,x):
    y = tf.transpose(
        linop.matmul(tf.transpose(x))
    )
    return y
    
def kernelDist( linop,x):
    '''
    y = (L x)^T (L x)
      = x^T L^T L x 
      L: linop
      x: x
'''
    y = bulkTransform(linop,x)
    res = tf.reduce_sum(tf.square(y),axis=-1)
    return res





from pymisca.tensorflow_extra_.hyper_plane import *
from pymisca.tensorflow_extra_.gaussian_field import GaussianField
from pymisca.tensorflow_extra_.affine_transform import *
from pymisca.tensorflow_extra_.affine_transform_diag import *
from pymisca.tensorflow_extra_.affine_transform_diag_plus_low_rank import *
from pymisca.tensorflow_extra_.radial_transform import *
from pymisca.tensorflow_extra_.radial_theta_transform import *
from pymisca.tensorflow_extra_.radial_cosine_transform import *
from pymisca.tensorflow_extra_.joint_dist import *
from pymisca.tensorflow_extra_.von_mises_fisher import *
from pymisca.tensorflow_extra_.von_mises_fisher_cosine import *

# edm.sphereUniformLRP = type('sphereUniformLRP',(edm.RandomVariable,
#                                                   sphereUniformLRP),{})



try:
    import edward.models as edm
    hasEdm = 1
except Exception as e:
    print ("[WARN]unable to import edward.models")
    hasEdm = 0
    
if hasEdm:

    lst = ['sphereUniformLRP',
           'AsAffine',
           'AsRadial',
           'AsRadialTheta',
          ]
    for _name in lst:
        _candidate = eval(_name)
        _class = type(_name, (edm.RandomVariable, _candidate),{})
        setattr(edm, _name, _class)

import tensorflow_probability.python.edward2 as ed
        
import pymisca.util as pyutil
##### Keep PointMass() statements
dummyF = lambda x: x
ed.models = pyutil.util_obj()
ed.models.PointMass = dummyF
ed.PointMass = dummyF



import tensorflow_probability.python.distributions as tfdist
import numpy as np

def quick_eval(v):
    if isinstance(v,np.ndarray):
        res =  v
    elif isinstance(v, tfdist.Distribution):
        res = v
    else:
        res = v.eval()
    return res


from pymisca.tensorflow_util import *

