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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util

from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor_shape


from tensorflow.python.ops import nn
from tensorflow_probability.python import bijectors; bijector = bijectors
from tensorflow_probability.python.distributions import TransformedDistribution,Normal

from pymisca.tensorflow_extra_.tfp_transformed_distribution import TransformedDistribution
from pymisca.tensorflow_extra_.transjector import Transjector

__all__ = [
    "RadialThetaTransform",
    "AsRadialTheta",
]


class RadialThetaTransform(Transjector):
  """
  """

  def __init__(self,
               D =2,
               power=0.,
               event_ndims=1,
               validate_args=False,
               debug = False,
               name="power_transform"):
    """Instantiates the `PowerTransform` bijector.

    Args:
      power: Python `float` scalar indicating the transform power, i.e.,
        `Y = g(X) = (1 + X * c)**(1 / c)` where `c` is the `power`.
      event_ndims: Python scalar indicating the number of dimensions associated
        with a particular draw from the distribution.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.

    Raises:
      ValueError: if `power < 0` or is not known statically.
    """
    self._graph_parents = []
    self._name = name
    self._validate_args = validate_args
#     self._D = tf.constant(D,dtype='int32')
    self.debug = debug
    with self._name_scope("init", values=[D]):
      D = ops.convert_to_tensor(D, name="dimension")
      self._D = D
      self.alpha = self._alpha = math_ops.cast(self._D,'float32') / array_ops.constant(2.)

#     if power is None or power < 0:
#       raise ValueError("`power` must be a non-negative TF constant.")
#     self._power = power
    
    super(RadialThetaTransform, self).__init__(
        forward_min_event_ndims = 1,
        inverse_min_event_ndims = 1,        
#         event_ndims=event_ndims,
        validate_args=validate_args,
        name=name)

  @property
  def D(self):
    """D as in (D-1)-sphere S^{D-1}"""
    return tensor_util.constant_value(self._D)

  def _forward(self, x):
    raise Exception ("Not implemented yet")        
    x = self._maybe_assert_valid_x(x)
    normal = Normal(loc=0.,scale=1.,)

    sp = array_ops.shape(x) ### use dynamical shape
    #### thanks to https://pgaleone.eu/tensorflow/2018/07/28/understanding-tensorflow-tensors-shape-static-dynamic/
#     sp = x.shape
    if self.debug:
        print (type(x),x)
        print ('xshape',x.shape)
    
    y = normal.sample( sp,)
    y = nn.l2_normalize(y,dim=-1) 
    
    if self.debug:
        print ('forward_sp',sp)
    if self.debug:
        print ('forward_xy',x.shape,y.shape)
        
        
    y = y *  math_ops.sqrt(x)
    return y

#   def  _forward_event_shape_tensor(self,input_shape):    
# #     sp = input_shape - 1 +  self.D
#     sp = tf.concat([input_shape[:-1], [self.D] ],0)
# #     sp = tf.concat([input_shape[:-1], [self.D] ],0)
# #     sp = tf.concat([input_shape.shape[:-1], [self.D] ],0)
#     print ('sp',sp)
# #     print (input_shape.shape)
# #     print (input_shape[0].eval())
#     return sp

#   def  _backward_event_shape_tensor(self,x):
#     sp = tf.concat([x.batch_shape_tensor(), [self.D] ],0)
#     return sp
#     if self.power == 0.:
#       return math_ops.exp(x)
#     # If large x accuracy is an issue, consider using:
#     # (1. + x * self.power)**(1. / self.power) when x >> 1.
#     return math_ops.exp(math_ops.log1p(x * self.power) / self.power)

  def _inverse(self, y):
    y = self._maybe_assert_valid_y(y)
    rsq = math_ops.reduce_sum( math_ops.square(y),
                              axis = -1,
                              keepdims = True)
    theta = y/math_ops.sqrt(rsq)

    x = array_ops.concat( [rsq, theta], -1 )
    return x

  @property
  def forward_constant(self,):
    return self._forward_constant()

  def _forward_constant(self,):
    ''' log( pi^ - \alpha * \Gamma(alpha) )
'''
    pi = math_ops.acos(-1.)
    cst = - self.alpha * math_ops.log( pi ) + math_ops.lgamma(self.alpha)
    return cst

  def _inverse_log_det_jacobian(self, y):
    y = self._maybe_assert_valid_y(y)
    x = self.inverse(y)
    val =  -self._forward_log_det_jacobian(x)
    return val

  def _forward_log_det_jacobian(self, x):
    x = self._maybe_assert_valid_x(x)
    
    rsq = array_ops.gather(x,0,axis=-1)    
    logDet = ( self.alpha -1 ) * math_ops.log(rsq) - math_ops.log(2.)
    
    return logDet


#     return math_ops.reduce_sum(x,)
#     if self.power == 0.:
#       return math_ops.reduce_sum(x, axis=event_dims)
#     return (1. / self.power - 1.) * math_ops.reduce_sum(
#         math_ops.log1p(x * self.power),
#         axis=event_dims)

  def _maybe_assert_valid_x(self, x):
    if not self.validate_args or self.power == 0.:
      return x
    is_valid = check_ops.assert_non_negative(
        1. + self.power * x,
        message="Forward transformation input must be at least {}.".format(
            -1. / self.power))
    return control_flow_ops.with_dependencies([is_valid], x)

  def _maybe_assert_valid_y(self, y):
    if not self.validate_args:
      return y
    is_valid = check_ops.assert_positive(
        y, message="Inverse transformation input must be greater than 0.")
    return control_flow_ops.with_dependencies([is_valid], y)

class AsRadialTheta(TransformedDistribution):
    ''' Transform a distribution on R^+ to distribution on R^D 
'''
    def __init__(self, distribution=None,D=None,debug=False,name='AsRadial'):
#         udist = tf.contrib.distributions.Uniform(low=0.,high=3.) 
        
        bjt = RadialThetaTransform(D = D,debug=debug,)
        super(AsRadialTheta,self).__init__(bijector=bjt,
                                         distribution=distribution,
#                                          event_shape = [D],
                                         simple=False,
                                         name = name,
                                        )
    @property    
    def D(self,):
        return self.bijector.D