# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops

from tensorflow_probability.python import distributions ;
distribution = distributions

# import pymisca.tensorflow_extra as pytf
import pymisca.tensorflow_util as pytfu
import tensorflow as tf

from tensorflow.python.framework import tensor_util

import numpy as np
import functools

__all__ = [
    "SimplexCosine",
#     "GaussianField",
#     "JointScalar",
]


def _static_value(x):
  """Returns the static value of a `Tensor` or `None`."""
  return tensor_util.constant_value(ops.convert_to_tensor(x))

def _logical_not(x):
  """Convenience function which attempts to statically apply `logical_not`."""
  x_ = _static_value(x)
  if x_ is None:
    return math_ops.logical_not(x)
  return constant_op.constant(np.logical_not(x_))

def _logical_equal(x, y):
  """Convenience function which attempts to statically compute `x == y`."""
  x_ = _static_value(x)
  y_ = _static_value(y)
  if x_ is None or y_ is None:
    return math_ops.equal(x, y)
  return constant_op.constant(np.array_equal(x_, y_))

def _ndims_from_shape(shape):
  """Returns `Tensor`'s `rank` implied by a `Tensor` shape."""
  if shape.get_shape().ndims not in (None, 1):
    raise ValueError("input is not a valid shape: not 1D")
  if not shape.dtype.is_integer:
    raise TypeError("input is not a valid shape: wrong dtype")
  if shape.get_shape().is_fully_defined():
    return constant_op.constant(shape.get_shape().as_list()[0])
  return array_ops.shape(shape)[0]

def _is_scalar_from_shape(shape):
  """Returns `True` `Tensor` if `Tensor` shape implies a scalar."""
  return _logical_equal(_ndims_from_shape(shape), 0)


class SimpldexCosine(distribution.Distribution):
    '''
    Assumes P(x,y,z,...) = P(x) * P(y) * P(z), ... . Note (x,y,z,...) NEEDS NOT sit on R^n
    
    Currently assume all distributions have batch_shape = ()    
    #### Collapse subdist.batch_shape into self.event_shape
    #### batch_shape is always []
'''
    def __init__(self,
                 mean,
                 L2loss = 0.,
                 radius = None,
#                  scale,
                 D = None,
#                  mean = 
#                  subDists,
                 validate_args=False,
                 allow_nan_stats=True,
                 fixScale= False,
                 name="SimplexCosine"
                ):
        parameters = locals()
#         self._subDists = list(subDists)
#         assert D is not None
#         self.D = D
        
#         for d in subDists:
#             assert isinstance(d,distribution.Distribution),'%s is not a tensorflow "Distribution"'%d
#         #### Managing event shapes
        self.mean = tf.convert_to_tensor(mean,
                                        dtype='float32')
        self.D = int(self.mean.shape[0])
        self.L2loss = tf.convert_to_tensor(L2loss,)
#         if not fixScale:
#             self.scale = tf.convert_to_tensor(scale,
#                                               dtype='float32')
#         else:
#             self.scale = self.loc ** 0.5
#         self.Normals = distributions.Normal(loc=self.loc, 
#                                             scale=self.scale)
#         assert self.loc.shape[0] == self.scale.shape[0]== (self.D * (self.D-1))//2

        super(HyperPlane, self).__init__(
            dtype = self.mean.dtype,
            reparameterization_type=distribution.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            graph_parents=[self.mean],
            name=name)
    def _L2_penalty(self,):
        penalty = self.L2loss *(tf.reduce_mean(tf.square(self.mean),) - 1.)**2
        return penalty
    
    def _log_prob(self, X):
#         X = array_ops.transpose(X)
#         sample_shape = ops.convert_to_tensor(
#           sample_shape, dtype=dtypes.int32, name="sample_shape")
        
        sample_shape = array_ops.shape(X)[:-1]
#         sample_shape, n = self._expand_sample_shape_to_vector(
#           sample_shape, "sample_shape")        
#         V = array_ops.reshape(X,shape = array_ops.concat( [[n], self.event_shape],0))
#         assert X.shape[-1] ==

        assert sample_shape.shape[0] == 1,'sample_shape:{sample_shape}\nX:{X}'.format(**locals())
    
#         X = X - self.mean[None]
#         X = X - self.mean[None]
#         tf.sqrt(self.mean)
        logP = tf.tensordot(X, self.mean, axes=1)
#         dotProd = tf.tensordot(X, self.mean,axes=1)
#         logP = dotProd   - self._L2_penalty()

#         pdX = X[:,:,None] - X[:,None,:]
#         f = functools.partial(pytfu.take_tril, 
#                       n = self.D, k=-1,transpose=False)
#         Y = pytfu.wrapper_perm(f,perm=(1,2,0),iperm=(1,0))(pdX)
#         logP = math_ops.reduce_sum( 
#             self.Normals.log_prob(Y),axis=-1
#         )

        return logP
            
    def subdist_is_scalar(self,i):
        return self._isScalar[i]
    
    def _sample_n(self,n,seed=None):
        lst = []
        for i in range(len(self.subDists)):
            ts = self.subDists[i].sample(n)
            if self.subdist_is_scalar(i):
                ts = array_ops.expand_dims(ts,-1)
            lst += [ts]
        out = array_ops.concat(lst,axis=-1)
        return out
    
    
    def _batch_shape_tensor(self):
        shape = constant_op.constant( [], dtype=dtypes.int32)
        return shape

    def _batch_shape(self):
        shape = tensor_shape.scalar()
        return shape
    
    def _event_shape_tensor(self):
        return tf.constant([self.D])
    def _event_shape(self):
#         shape = tensor_shape.scalar(self.D)
        shape = tensor_shape.TensorShape([self.D])
        return shape
#         return math_ops.reduce_sum(
#                 self._sample_shape_list,0,
#             )

#     def _event_shape(self):
#         return np.sum(self._sample_shape_list_static, 0,)

# JointScalar = JointDist
#     