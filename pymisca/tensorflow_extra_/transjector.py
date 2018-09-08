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
from tensorflow_probability.python import bijectors
# from tensorflow_probability.python.distributions import TransformedDistribution,Normal


__all__ = [
    "Transjector",
]

class Transjector(bijectors.Bijector):
  def _call_forward(self, x, name, **kwargs):
    with self._name_scope(name, [x]):
      x = ops.convert_to_tensor(x, name="x")
      self._maybe_assert_dtype(x)
      if 1: #### No cahcing since this is non-injective
        
#   @property
#   def _is_injective(self):
#     return False
        
        
       
        return self._forward(x, **kwargs)