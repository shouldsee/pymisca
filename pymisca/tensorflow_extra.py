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


class sphereUniformDiag(distribution.Distribution):
    """The Normal distribution with location `loc` and `scale` parameters.

    #### Mathematical details

    The probability density function (pdf) is,

    ```none
    pdf(x; mu, sigma) = exp(-0.5 (x - mu)**2 / sigma**2) / Z
    Z = (2 pi sigma**2)**0.5
    ```

    where `loc = mu` is the mean, `scale = sigma` is the std. deviation, and, `Z`
    is the normalization constant.

    The Normal distribution is a member of the [location-scale family](
    https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
    constructed as,

    ```none
    X ~ Normal(loc=0, scale=1)
    Y = loc + scale * X
    ```

    #### Examples

    Examples of initialization of one or a batch of distributions.

    ```python
    # Define a single scalar Normal distribution.
    dist = tf.distributions.Normal(loc=0., scale=3.)

    # Evaluate the cdf at 1, returning a scalar.
    dist.cdf(1.)

    # Define a batch of two scalar valued Normals.
    # The first has mean 1 and standard deviation 11, the second 2 and 22.
    dist = tf.distributions.Normal(loc=[1, 2.], scale=[11, 22.])

    # Evaluate the pdf of the first distribution on 0, and the second on 1.5,
    # returning a length two tensor.
    dist.prob([0, 1.5])

    # Get 3 samples, returning a 3 x 2 tensor.
    dist.sample([3])
    ```

    Arguments are broadcast when possible.

    ```python
    # Define a batch of two scalar valued Normals.
    # Both have mean 1, but different standard deviations.
    dist = tf.distributions.Normal(loc=1., scale=[11, 22.])

    # Evaluate the pdf of both distributions on the same point, 3.0,
    # returning a length 2 tensor.
    dist.prob(3.0)
    ```

    """

    def __init__(self,
#                              D=2,
                             loc=[],
                             scale=[],
                             validate_args=False,
                             allow_nan_stats=True,
                             name="sphereUnif",
                             detPenalty = 0.125 ,
#                              detPenalty = 0.25 ,
                ):
        """Construct Normal distributions with mean and stddev `loc` and `scale`.

        The parameters `loc` and `scale` must be shaped in a way that supports
        broadcasting (e.g. `loc + scale` is a valid operation).

        Args:
            loc: Floating point tensor; the means of the distribution(s).
            scale: Floating point tensor; the stddevs of the distribution(s).
                Must contain only positive values.
            validate_args: Python `bool`, default `False`. When `True` distribution
                parameters are checked for validity despite possibly degrading runtime
                performance. When `False` invalid inputs may silently render incorrect
                outputs.
            allow_nan_stats: Python `bool`, default `True`. When `True`,
                statistics (e.g., mean, mode, variance) use the value "`NaN`" to
                indicate the result is undefined. When `False`, an exception is raised
                if one or more of the statistic's batch members are undefined.
            name: Python `str` name prefixed to Ops created by this class.

        Raises:
            TypeError: if `loc` and `scale` have different `dtype`.
        """
        parameters = locals()
#         self._event_shape_tensor_ = 
#         self.D = D
#         self._scale
        with ops.name_scope(name, values=[loc, scale]):
            with ops.control_dependencies([check_ops.assert_positive(scale)] if
                                                                        validate_args else []):
#                 self._loc = array_ops.identity(loc, name="loc",dtype='float32')
                self._detPenalty = ops.convert_to_tensor(detPenalty, name="norm",dtype="float32")
                
                self._scale = ops.convert_to_tensor(scale, name="scale",dtype="float32")
#                 self._scale = tf.matrix_diag( self._scale )
                loc = tf.zeros( self._scale.shape[-1:],dtype='float32') if loc == [] else loc
                self._loc = ops.convert_to_tensor(loc, name="loc",dtype='float32')
#                 self._scale = array_ops.identity(scale, name="scale",dtype='float32')
                check_ops.assert_same_float_dtype([self._loc, self._scale])
                
        super(sphereUniformDiag, self).__init__(
                dtype=self._scale.dtype,
                reparameterization_type=distribution.FULLY_REPARAMETERIZED,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                parameters=parameters,
#                 graph_parents=[self._loc, self._scale],
                name=name)

    @staticmethod
    def _param_shapes(sample_shape):
        return dict(
                zip(("loc", "scale","norm"), ([ops.convert_to_tensor(
                        sample_shape, dtype=dtypes.int32)] * 2)))

    @property
    def loc(self):
        """Distribution parameter for the mean."""
        return self._loc

    @property
    def scale(self):
        """Distribution parameter for standard deviation."""
        return self._scale

    @property
    def detPenalty(self):
        """Distribution parameter for standard deviation."""
        return self._detPenalty

    def _batch_shape_tensor(self):
        return array_ops.broadcast_dynamic_shape(
                array_ops.shape(self.loc),
                array_ops.shape(self.scale))

    def _batch_shape(self):
        return array_ops.broadcast_static_shape(
                self.loc.get_shape(),
                self.scale.get_shape())
#                 self.scale.get_shape())

    def _event_shape_tensor(self):
#         res 
#         res = self.scale.shape[-1:]
        res = ops.convert_to_tensor(
        constant_op.constant([], dtype=dtypes.int32))
        return res
#         return constant_op.constant([], dtype=dtypes.int32)
    def _event_shape(self):
#         return self._scale.shape[-1:]
#         return tensor_shape.TensorShape(self._scale.shape[-1])
        return tensor_shape.scalar()


    def _z(self, x):
        """Standardize input `x` to a unit normal."""
        with ops.name_scope("standardize", values=[x]):
#             x = ops.convert_to_tensor(x,dtype="float32",)
            y = nn.l2_normalize(x=x,dim=-1)
            return y
    

    def _sample_n(self, n, seed=None):
        shape = array_ops.concat([[n], self.batch_shape_tensor(), self.event_shape_tensor()], 0)
        sampled = random_ops.random_normal(
            shape=shape, mean=0., stddev=1., 
            dtype=self.loc.dtype, seed=seed)
        return self._z(sampled)
#         return sampled * self.scale + self.loc

    def input2batch_shape(self,x):
#         assert math_ops.equal( x.shape[-1],self.event_shape_tensor())
        return tf.concat( [x.shape[:-1],
                                 self.event_shape_tensor()],
                             0)


    def _log_unnormalized_prob(self, x):
        x = self._z(x)

        res = tf.expand_dims(x,axis=-1) * tf.expand_dims(x,axis=-2)
        scale = self.scale                
        scale = tf.matrix_diag( scale )
        res = res * scale
        res = tf.reduce_sum( res,axis=(-1,-2))
        return -tf.log(res)
#         return tf.zeros(shape = self.input2batch_shape(x))

    def _log_normalization(self):
        #### The 
        det = tf.reduce_prod(self.scale)
#         c = 0.5
        v =  - math_ops.log(det) * tf.constant([0.5],dtype="float32")
        res = - math_ops.log(det) *  self.detPenalty
        res = tf.square(res) + v        
        return res
    def _log_prob(self, x):
        return self._log_unnormalized_prob(x) - self._log_normalization()

    def _prob(self, x):
        return math_ops.exp(self._log_prob(x))

    
    def _entropy(self):
        # Use broadcasting rules to calculate the full broadcast scale.
        scale = self.scale * array_ops.ones_like(self.loc)
        return 0.5 * math.log(2. * math.pi * math.e) + math_ops.log(scale)
    
    def _mean(self):
        return tf.concat( [x.shape[:-1],
                                 self.event_shape_tensor()],
                             0)




    def _log_cdf(self, x):
        raise Exception("Not implemented")
#         return special_math.log_ndtr(self._z(x))

    def _cdf(self, x):
        raise Exception("Not implemented")
#         return special_math.ndtr(self._z(x))

    def _log_survival_function(self, x):
        raise Exception("Not implemented")
#         return special_math.log_ndtr(-self._z(x))

    def _survival_function(self, x):
        raise Exception("Not implemented")
#         return special_math.ndtr(-self._z(x))
        

    def _quantile(self, p):
        raise Exception("Not implemented")
#         return self._inv_z(special_math.ndtri(p))

    def _stddev(self):
        raise Exception("Not implemented")        
#         return self.scale * array_ops.ones_like(self.loc)

    def _mode(self):
        raise Exception("Not implemented")
#         return self._mean()

    
    
import edward.models as edm
# edm.RandomVariable

edm.sphereUniformDiag = type('sphereUniformDiag',(edm.RandomVariable,
                                                  sphereUniformDiag),{})
# edm.sphereUniformDiag = sphereUniformDiag

from tensorflow.contrib import linalg
from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.contrib.distributions.python.ops import mvn_linear_operator as mvn_linop
from tensorflow.python.framework import ops
from tensorflow.python.ops import check_ops

class sphereUniformLRP(distribution.Distribution):
#         mvn_linop.MultivariateNormalLinearOperator):
    def __init__(self,
                             loc=None,
                             scale_diag=None,
                             scale_identity_multiplier=None,
                             scale_perturb_factor=None,
                             scale_perturb_diag=None,
                             detPenalty = 0.125,
                             concentration = None,
                             rate = None,
                             validate_args=False,
                             allow_nan_stats=True,
                             name="MultivariateNormalDiagPlusLowRank"):
        """Construct Multivariate Normal distribution on `R^k`.

        The `batch_shape` is the broadcast shape between `loc` and `scale`
        arguments.

        The `event_shape` is given by last dimension of the matrix implied by
        `scale`. The last dimension of `loc` (if provided) must broadcast with this.

        Recall that `covariance = scale @ scale.T`. A (non-batch) `scale` matrix is:

        ```none
        scale = diag(scale_diag + scale_identity_multiplier ones(k)) +
                scale_perturb_factor @ diag(scale_perturb_diag) @ scale_perturb_factor.T
        ```

        where:

        * `scale_diag.shape = [k]`,
        * `scale_identity_multiplier.shape = []`,
        * `scale_perturb_factor.shape = [k, r]`, typically `k >> r`, and,
        * `scale_perturb_diag.shape = [r]`.

        Additional leading dimensions (if any) will index batches.

        If both `scale_diag` and `scale_identity_multiplier` are `None`, then
        `scale` is the Identity matrix.

        Args:
            loc: Floating-point `Tensor`. If this is set to `None`, `loc` is
                implicitly `0`. When specified, may have shape `[B1, ..., Bb, k]` where
                `b >= 0` and `k` is the event size.
            scale_diag: Non-zero, floating-point `Tensor` representing a diagonal
                matrix added to `scale`. May have shape `[B1, ..., Bb, k]`, `b >= 0`,
                and characterizes `b`-batches of `k x k` diagonal matrices added to
                `scale`. When both `scale_identity_multiplier` and `scale_diag` are
                `None` then `scale` is the `Identity`.
            scale_identity_multiplier: Non-zero, floating-point `Tensor` representing
                a scaled-identity-matrix added to `scale`. May have shape
                `[B1, ..., Bb]`, `b >= 0`, and characterizes `b`-batches of scaled
                `k x k` identity matrices added to `scale`. When both
                `scale_identity_multiplier` and `scale_diag` are `None` then `scale` is
                the `Identity`.
            scale_perturb_factor: Floating-point `Tensor` representing a rank-`r`
                perturbation added to `scale`. May have shape `[B1, ..., Bb, k, r]`,
                `b >= 0`, and characterizes `b`-batches of rank-`r` updates to `scale`.
                When `None`, no rank-`r` update is added to `scale`.
            scale_perturb_diag: Floating-point `Tensor` representing a diagonal matrix
                inside the rank-`r` perturbation added to `scale`. May have shape
                `[B1, ..., Bb, r]`, `b >= 0`, and characterizes `b`-batches of `r x r`
                diagonal matrices inside the perturbation added to `scale`. When
                `None`, an identity matrix is used inside the perturbation. Can only be
                specified if `scale_perturb_factor` is also specified.
            validate_args: Python `bool`, default `False`. When `True` distribution
                parameters are checked for validity despite possibly degrading runtime
                performance. When `False` invalid inputs may silently render incorrect
                outputs.
            allow_nan_stats: Python `bool`, default `True`. When `True`,
                statistics (e.g., mean, mode, variance) use the value "`NaN`" to
                indicate the result is undefined. When `False`, an exception is raised
                if one or more of the statistic's batch members are undefined.
            name: Python `str` name prefixed to Ops created by this class.

        Raises:
            ValueError: if at most `scale_identity_multiplier` is specified.
        """
        parameters = locals()
        def _convert_to_tensor(x, name):
            return None if x is None else ops.convert_to_tensor(x, name=name)
        with ops.name_scope(name):
            with ops.name_scope("init", values=[
                    loc, 
                    scale_diag, 
                    scale_identity_multiplier, 
                    scale_perturb_factor,
                    scale_perturb_diag,
                    detPenalty]):
                
                self._detPenalty = ops.convert_to_tensor(detPenalty, name="norm",dtype="float32")
                
                self.has_low_rank = has_low_rank = (scale_perturb_factor is not None or
                                                scale_perturb_diag is not None)
                self.has_gamma = ( concentration is not None and
                             rate is not None ) 

                scale = distribution_util.make_diag_scale(
                        loc=loc,
                        scale_diag=scale_diag,
                        scale_identity_multiplier=scale_identity_multiplier,
                        validate_args=validate_args,
                        assert_positive=has_low_rank)
                


                scale_perturb_factor = _convert_to_tensor(
                        scale_perturb_factor,
                        name="scale_perturb_factor")
                scale_perturb_diag = _convert_to_tensor(
                        scale_perturb_diag,
                        name="scale_perturb_diag")

                if self.has_low_rank:
                    scale = linalg.LinearOperatorUDVHUpdate(
                            scale,
                            u=scale_perturb_factor,
                            diag_update=scale_perturb_diag,
                            is_diag_update_positive=scale_perturb_diag is None,
                            is_non_singular=True,    # Implied by is_positive_definite=True.
                            is_self_adjoint=True,
                            is_positive_definite=True,
                            is_square=True)
                else:
                    raise Exception("must specify Low-rank perturbation...(to be implemented)")
                if self.has_gamma:
                    self._concentration = array_ops.identity(
                        concentration, name="concentration")
                    self._rate = array_ops.identity(rate, name="rate")
                    check_ops.assert_same_float_dtype(
                        [self._concentration, self._rate])
                else:
                    #### Assume Uniform distribution on radius xT K x
                    self._concentration = None
                    self._rate = None
                                
#         seflffffffffffffffff
        self._scale = scale
        if loc is not None:
            self._loc = ops.convert_to_tensor(loc, name="loc",dtype='float32') 
        else:
            self._loc = tf.zeros(scale.shape[0])
            
        batch_shape, event_shape = distribution_util.shapes_from_loc_and_scale(
          self._loc, 
#             scale,
            scale,
        )
#         self.__batch_shape = batch_shape
#         self.__event_shape = event_shape

#         super(sphereUniformLRP, self).__init__(
#                 loc=loc,
#                 scale=scale,
#                 validate_args=validate_args,
#                 allow_nan_stats=allow_nan_stats,
#                 name=name)
          
        super(sphereUniformLRP, self).__init__(
                dtype=self._scale.dtype,
                reparameterization_type=distribution.FULLY_REPARAMETERIZED,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                parameters=parameters,
                graph_parents= [self._loc, self._detPenalty ] + self._scale.graph_parents,
                name=name)
        
        
        self._parameters = parameters

#         @staticmethod
#         def _param_shapes(sample_shape):
#             return dict(
#                     zip(("loc", "scale","norm"), ([ops.convert_to_tensor(
#                             sample_shape, dtype=dtypes.int32)] * 2)))

    @property
    def loc(self):
        """Distribution parameter for the mean."""
        return self._loc

    @property
    def scale(self):
        """Distribution parameter for standard deviation."""
        return self._scale
    @property
    def forwardMat(self):
        I = tf.eye( self.scale.shape[-1].value)
        res = self.scale.solve(I)
#         res = linalg.LinearOperator(res)
        return res

    @property
    def detPenalty(self):
        """Distribution parameter for standard deviation."""
        return self._detPenalty

    @property
    def concentration(self):
        """Concentration parameter."""
        return self._concentration

    @property
    def rate(self):
        """Rate parameter."""
        return self._rate    

    def _batch_shape_tensor(self):
        return array_ops.broadcast_dynamic_shape(
#                 array_ops.shape(self.loc),
                array_ops.shape(self.scale[:-2]),
                array_ops.shape(self.scale[:-2]),
        )

    def _batch_shape(self):
        return array_ops.broadcast_static_shape(
#                 self.loc.get_shape(),
                self.scale.shape[:-2],
                self.scale.shape[:-2],
#                 self.scale.get_shape()
        )

    def _event_shape_tensor(self):
#         res 
        res = self.scale.shape[-1:]
#         res = ops.convert_to_tensor(
#         constant_op.constant([], dtype=dtypes.int32))
        return res
#         return constant_op.constant([], dtype=dtypes.int32)
    def _event_shape(self):
        return self._scale.shape[-1:]

#         return tensor_shape.TensorShape(self._scale.shape[-1])
#         return tensor_shape.scalar()

    
    def _z(self, x):
        """Standardize input `x` to a unit normal."""
        with ops.name_scope("standardize", values=[x]):
#             x = ops.convert_to_tensor(x,dtype="float32",)
            y = nn.l2_normalize(x=x,dim=-1)
            return y

    def _sample_n(self, n, seed=None):
        shape = array_ops.concat([[n], 
#                                   self.batch_shape_tensor(), 
                                  self.event_shape_tensor(),
                                 ], 0)
        sampled = random_ops.random_normal(
            shape=shape, mean=0., stddev=1., 
            dtype=self.loc.dtype, 
            seed=seed)
        
#         print ( self.scale.shape)
#         print ( self.batch_shape_tensor().eval(), 
#                self.event_shape_tensor().eval(),)
#         print (self.forwardMat.shape,
#                tf.transpose(sampled).shape)
        
        y = tf.matmul(self.forwardMat, tf.transpose(sampled))
        sampled_mvn = tf.transpose(y)
        d = tf.expand_dims(self.kernelDist(sampled_mvn),-1)
        sampled_mvn = sampled_mvn/tf.sqrt(d)
#         d1 = self.kernelDist(sampled_mvn )
#         print (tf.reduce_mean(d1).eval(),tf.reduce_max(d1).eval(),tf.reduce_min(d1).eval())
        
        if self.has_gamma:
            gamma = random_ops.random_gamma(
                shape=[n],
                alpha=self.concentration,
                beta=self.rate,
                dtype=self.dtype,
                seed=seed)
            sampled_mvn = sampled_mvn  * tf.sqrt(gamma)

#         bulkTransform(self.forwardMat,sampled)
        return sampled_mvn
#         return self._z(sampled)
#         return sampled * self.scale + self.loc

    def input2batch_shape(self,x):
#         assert math_ops.equal( x.shape[-1],self.event_shape_tensor())
        return tf.concat( [x.shape[:-1],
                                 self.event_shape_tensor()],
                             0)


    def _log_unnormalized_prob(self, x):
        d0 = kernelDist(self.scale, x)
        l2 = tf.reduce_sum(tf.square(x),axis=-1)
        res =  tf.log(l2) - tf.log(d0)
        
        if self.has_gamma:
            logGa = (self.concentration - 1.) * math_ops.log(d0) - self.rate * d0
            res = res +logGa
        
        return res
#         return -tf.log(res)
#         return tf.zeros(shape = self.input2batch_shape(x))


    def _log_normalization(self):
        #### The 
#             det = tf.reduce_prod(self.scale)
#             det = gen_linalg_ops.matrix_determinant(self.scale)
        det = self.scale.determinant()
#         c = 0.5
        v =  - math_ops.log(det) * tf.constant([1.0],dtype="float32")
#         v =  - math_ops.log(det) * tf.constant([0.5],dtype="float32")

        res = - math_ops.log(det) *  self.detPenalty * 2.
        res = tf.square(res) + v 
        
        if self.has_gamma:
            logGa = (math_ops.lgamma(self.concentration)
                        - self.concentration * math_ops.log(self.rate))
            res = res + logGa
            
        return res
    
    def _log_prob(self, x):
        return self._log_unnormalized_prob(x) - self._log_normalization()

    def _prob(self, x):
        return math_ops.exp(self._log_prob(x))


    def _entropy(self):
        # Use broadcasting rules to calculate the full broadcast scale.
        scale = self.scale * array_ops.ones_like(self.loc)
        return 0.5 * math.log(2. * math.pi * math.e) + math_ops.log(scale)

    def _mean(self):
        return tf.concat( [x.shape[:-1],
                                 self.event_shape_tensor()],
                             0)
    
    def _covariance(self):
        res = tf.matmul( self.forwardMat, tf.transpose(self.forwardMat))
#         res = self.scale.matmul(self.scale.to_dense())
#         return self.scale.to_dense()
        return res
#         return tf.concat( [x.shape[:-1],
#                                  self.event_shape_tensor()],
#                              0)
    @property
    def precision(self):
        res = self.scale.matmul(self.scale.to_dense())
        return res
    def kernelDist(self,x):
        res = kernelDist(self.scale, x)
        return res
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

edm.sphereUniformLRP = type('sphereUniformLRP',(edm.RandomVariable,
                                                  sphereUniformLRP),{})
