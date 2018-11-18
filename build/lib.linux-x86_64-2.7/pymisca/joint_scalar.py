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


from tensorflow.python.framework import tensor_util

import numpy as np

__all__ = [
    "JointDist",
    "JointScalar",
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


class JointDist(distribution.Distribution):
    '''
    Assumes P(x,y,z,...) = P(x) * P(y) * P(z), ... . Note (x,y,z,...) NEEDS NOT sit on R^n
    
    Currently assume all distributions have batch_shape = ()    
    #### Collapse subdist.batch_shape into self.event_shape
    #### batch_shape is always []
'''
    def __init__(self,
                 subDists,
                 validate_args=False,
                 allow_nan_stats=True,
                 name="JointScalar"
                ):
        parameters = locals()
        self._subDists = list(subDists)
        
        #### Managing event shapes
        self._event_shape_list = [ 
                array_ops.concat(
                    [x.batch_shape_tensor(),
                     x.event_shape_tensor()
                    ],0)
            for x in subDists]
        self._isScalar  = [
            _static_value( _is_scalar_from_shape(x) )
           for x in self._event_shape_list ]
        
        for i,val in enumerate(self._isScalar):
            assert val is not None
            if val:
                shape = constant_op.constant([1],dtype="int32")
                self._event_shape_list[i] = shape
        self._event_shape_list_static = map(_static_value, self._event_shape_list,)
                
               
                
            
            
        for d in subDists:
            assert isinstance(d,distribution.Distribution),'%s is not a tensorflow "Distribution"'%d
        graph_parents = sum([d._graph_parents for d in self._subDists],[])
        super(JointDist, self).__init__(
            dtype= graph_parents[0].dtype,
            reparameterization_type=distribution.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            graph_parents=graph_parents,
            name=name)
    @property
    def subDists(self):
        '''Component distributions
'''
        return self._subDists
    
    def _log_prob(self,X):
#         X = array_ops.transpose(X)
#         sample_shape = ops.convert_to_tensor(
#           sample_shape, dtype=dtypes.int32, name="sample_shape")

        sample_shape = array_ops.shape(X)[:-1]
        sample_shape, n = self._expand_sample_shape_to_vector(
          sample_shape, "sample_shape")        
        V = array_ops.reshape(X,shape = array_ops.concat( [[n], self.event_shape],0))
        lst = []        
        
        seps = [0,] + np.cumsum( self._event_shape_list_static ).tolist()
        for i in range(len(self.subDists)):            
            data = V[:,seps[i]:seps[i+1]] 
            lp = self.subDists[i].log_prob(data)
            
            if not self.subdist_is_scalar(i):
                
                #### Either Remove singleton dimension from scalars
#                 data = array_ops.gather(data,0,axis=-1)

                ##### Or Add singleton dimension for non-scalar dists 
                ## for later concatenation
                lp = array_ops.expand_dims(lp,axis=-1)
            lst += [lp]
        logP = math_ops.reduce_sum(array_ops.concat(lst,1),axis=-1)
        
        logP = array_ops.reshape(logP, shape=sample_shape )
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
        shape = constant_op.constant( len(self.subDists), dtype=dtypes.int32)
#         shape = reduce(array_ops.broadcast_dynamic_shape,
#           [x.batch_shape_tensor() for x in self.subDists])
        return shape

    def _batch_shape(self):
#         shape = reduce(array_ops.broadcast_static_shape,
#           [x.batch_shape for x in self.subDists])
        shape = tensor_shape.scalar()
        return shape
    
    def _event_shape_tensor(self):

        return math_ops.reduce_sum(
                self._event_shape_list,0,
            )
#         return constant_op.constant( len(self.subDists), dtype=dtypes.int32)

    def _event_shape(self):
        return np.sum(self._event_shape_list_static, 0,)
#         shape = reduce(array_ops.broadcast_static_shape,
#           [x.batch_shape for x in self.subDists])
#         return shape
#         return _static_value( self._event_shape_tensor() )
    
#         return tensor_shape.TensorShape(len(self.subDists))  
JointScalar = JointDist
    