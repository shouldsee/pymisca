import tensorflow as tf
import tensorflow_probability.python.distributions as tfdist
# VonMisesFisher = tfdist.VonMisesFisher
from pymisca.von_mises_fisher import VonMisesFisher

__all__ = ['VonMisesFisherCosine']

class VonMisesFisherCosine(tfdist.Distribution):
    def __init__(self,
                 D,
                 concentration,
                 allow_nan_stats=True,
                 validate_args=False,
                 name = 'VonMisesFisherCosine'):
        '''
        Distribution that takes ( x^T * mu  ) as its input
        instead of (x) itself
    '''
        parameters = dict(locals())
        with tf.name_scope(name, values=[concentration,D]) as name:
            self._one = tf.constant([],dtype='float32')
            
            D = tf.convert_to_tensor(D,name='dimension',
                                     dtype="int32")
            concentration = tf.convert_to_tensor(concentration,
                                                name='concentration')
            dtype = concentration.dtype
            self._D = D
            self._Dfloat = tf.cast(D,'float32')
            self._concentration = concentration
            self._mean_direction= tf.concat(
                [[1.],tf.zeros(self.D - 1,dtype=dtype)], 0)
            dtype = self._concentration.dtype

            self._vm = VonMisesFisher(
                concentration = self._concentration,
                mean_direction = self._mean_direction,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
            )

            super(VonMisesFisherCosine, self).__init__(
              dtype=self._concentration.dtype,
              validate_args=validate_args,
              allow_nan_stats=allow_nan_stats,
              reparameterization_type=tfdist.FULLY_REPARAMETERIZED,
              parameters=parameters,
              graph_parents= [self._concentration],
              name=name)

    @property
    def mean_direction(self):
        """Mean direction parameter."""
        return self._mean_direction

    @property
    def concentration(self):
        """Concentration parameter."""
        return self._concentration
    
    @property
    def D(self):
        """
        Dimension D for the (D - 1)-sphere"""
        return self._D

    def _log_prob(self, x):
        x = tf.expand_dims(x,-1)
        xr= tf.sqrt( (self._one - tf.square(x) )/ 
                    (self._Dfloat - self._one ))        
        xall = tf.concat([x,xr],axis=-1)
        logP = self._vm._log_prob(xall)
        return logP
    
    def _sample_n(self,n,seed=None):
        xall = self._vm._sample_n(n,seed=seed)
        x = tf.gather(xall,axis=-1,indices=0)
        return x
    
    def _batch_shape_tensor(self):
        return tf.shape(self.concentration)
    
    def _event_shape_tensor(self):
        return tf.constant([], dtype=tf.int32)
    
    

