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

def op_minimise(loss,
                free_params,                
                optimizer=None,
                sess = None,
                MAX_ITER = 4000,
#                 LEARNING_RATE = 0.1,
                TOL_LOSS = 1e-8,
                TOL_PARAM = 1e-8,
                TOL_GRAD = 1e-8,
                feed_dict= {}
                ):
    '''
    #### Adapted from: http://kyleclo.github.io/maximum-likelihood-in-tensorflow-pt-1/
'''
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
    if 1:
#     with tf.Session() as sess:
        
            # initialize
        sess.run(fetches=tf.global_variables_initializer())
        obs_loss = sess.run(fetches=[loss], feed_dict=feed_dict)
        obs_vars = sess.run(fetches=free_params)
        for i in range(MAX_ITER):
            # gradient step
            sess.run(fetches=train_op, feed_dict=feed_dict)    
            new_loss = sess.run(fetches=loss, feed_dict=feed_dict)
#             if grad is not None:
            new_grad = sess.run(fetches=grad, feed_dict=feed_dict)
            loss_diff = abs(new_loss - obs_loss[-1])

            obs_loss.append(new_loss)
            if not i%100:
                print ('Iter %d'%i, new_loss)
            if loss_diff < TOL_LOSS:
                print('Loss function convergence in {} iterations!: {}'.format(i,new_loss))
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

# from pymisca.affine_transform_diag import *
