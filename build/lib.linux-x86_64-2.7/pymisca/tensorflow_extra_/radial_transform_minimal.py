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
from tensorflow_probability.python import bijectors; Transjector = bijectors.Bijector
from tensorflow_probability.python.distributions import TransformedDistribution,Normal


__all__ = [
    "RadialTransform",
    "AsRadial",
]


class RadialTransform(Transjector):
  """
  given x = r^2 , return a random point that satisfies y^T y = r^2
  """

  def __init__(self,
               D =2,
               power=0.,
#                event_ndims=1,
               validate_args=False,
               debug = False,
               name="radial_transform"):
    """Instantiates the `RadialTransform` bijector.

    Args:
      D: number of dimensions associated
        with a particular draw from the distribution
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    self._graph_parents = []
    self._name = name
    self._validate_args = validate_args
    self.debug = debug
    
    with self._name_scope("init", values=[D]):
      D = ops.convert_to_tensor(D, name="dimension")
      self._D = D
      self.alpha = self._alpha = math_ops.cast(self._D,'float32') / array_ops.constant(2.)
    
    super(RadialTransform, self).__init__(
        forward_min_event_ndims = 0,
        inverse_min_event_ndims = 1,
        validate_args=validate_args,
        name=name)


  @property
  def D(self):
    """D as in (D-1)-sphere S^{D-1}"""
    return tensor_util.constant_value(self._D)

  def  _forward_event_shape(self,input_shape):    
    e = input_shape.concatenate(
        tensor_shape.as_dimension(self.D))
    return e

  def _call_forward(self, x, name, **kwargs):
    with self._name_scope(name, [x]):
      x = ops.convert_to_tensor(x, name="x")
      self._maybe_assert_dtype(x)

  def _forward(self, x):
#     x = self._maybe_assert_valid_x(x)
    sp = array_ops.shape(x) ### use dynamical shape        
    sp = array_ops.concat([sp, 
                           self._D[None],
                              ],
                           0)
    
    normal = Normal(loc=0.,scale=1.,)
    y = normal.sample( sp,)
    y = nn.l2_normalize(y, axis=-1) 
    x = array_ops.expand_dims(x,-1)
    y = y *  math_ops.sqrt(x)
    return y

  def _inverse(self, y):
#     y = self._maybe_assert_valid_y(y)
    x = math_ops.reduce_sum( math_ops.square(y),axis = -1)
    return x


  def _call_forward(self, x, name, **kwargs):
    with self._name_scope(name, [x]):
      x = ops.convert_to_tensor(x, name="x")
      self._maybe_assert_dtype(x)
      if 1: #### No cahcing since this is non-injective
        return self._forward(x, **kwargs)        

  def  _forward_event_shape(self,input_shape):    
    e = input_shape.concatenate(
        tensor_shape.as_dimension(self.D))
    return e

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
    y   = self._maybe_assert_valid_y(y)
    rsq = self.inverse(y)
    val = - self._forward_log_det_jacobian(rsq)
    return val

  def _forward_log_det_jacobian(self, x):
    x = self._maybe_assert_valid_x(x)
    rsq = x

    logDet = ( -self.alpha + 1. ) * math_ops.log(rsq) + self.forward_constant    
    logDet = -logDet
    
    return logDet

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


class AsRadial(TransformedDistribution):
    ''' Transform a distribution on R^+ to distribution on R^D 
'''
    def __init__(self, distribution=None,
                 D=None,
                 debug=False,
#                  simple=False,
                 name='AsRadial'):
#         udist = tf.contrib.distributions.Uniform(low=0.,high=3.) 
        
        bjt = RadialTransform(D = D,debug=debug,)
        super(AsRadial,self).__init__(  bijector=bjt,
                                         distribution=distribution,
#                                          event_shape = [D],
#                                          simple=simple,
                                         name = name,
                                        )
    @property    
    def D(self,):
        return self.bijector.D
    
if __name__=='__main__':
    import numpy as np
    from tensorflow.python.ops import nn
    import tensorflow as tf
    import sys
    pytf = sys.modules[__name__]


    D = 3
    x = tf.placeholder(dtype=tf.float32)


    def makeEMD( (mu,phi,) ):
        dist_rsq = tf.contrib.distributions.Gamma(concentration=mu,rate=phi)

        #### Use a bijector to calculate P(x) from P(r^2)
        dist_xyz = mdl = pytf.AsRadial(distribution=dist_rsq,
    #                                    simple=False,
                                       debug = False,
                                       D=D)

        return mdl

    mu = 2.
    phi = 1.

    fitted_vars = [mu,phi,]
    emission = makeEMD(fitted_vars)


    Xmd = np.random.random((100,200,D)).astype(np.float32)
    bjt = emission.bijector

    #### One dimensional samples
    xTheta = bjt.inverse(Xmd)
    print ('R^2 shape',xTheta.shape)

    #### MultiDimensional samples
    Xmd = bjt.forward(xTheta)
    print ('Multidimension shape',Xmd.shape)

    fldj = val = bjt.forward_log_det_jacobian(xTheta,event_ndims=0)
    print ('forward_log_det shape',val.shape,)

    ildj = bjt.inverse_log_det_jacobian(Xmd,event_ndims=1,)
    print ('inverse_log_det shape',val.shape)

    sess = tf.InteractiveSession()
    for i in range(3):
        a = Xmd.eval().flat[:3]
        rsq = np.sum(np.square(a))
        print ('forward() run %d'%i,)
        print ('xyz:',a)
        print ('R^2',rsq)
        print ( )
    #     print ('forward() 2nd run',Xmd.eval().flat[:3])

    assert 0,'[Done]'