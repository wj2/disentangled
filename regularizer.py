import tensorflow as tf
import tensorflow_probability as tfp

@tf.keras.utils.register_keras_serializable(name='var9')
class VarianceRegularizer(tf.keras.regularizers.Regularizer):

    def __init__(self, weights=(.1, .1), **kwargs):
        if kwargs:
            raise TypeError('Argument(s) not recognized: %s' % (kwargs,))
        self.l2_weight = float(weights[0])
        self.var_weight = float(weights[1])

    def get_config(self):
        return {'weights':(self.l2_weight, self.var_weight)}
        
    def __call__(self, x):
        l2_r = tf.math.reduce_sum(tf.math.square(x))
        var_r = tf.math.reduce_variance(tf.math.reduce_mean(x**2, axis=0))
        return self.l2_weight*l2_r + self.var_weight*var_r

class L2PRRegularizer(tf.keras.regularizers.Regularizer):

    def __init__(self, weights=(.1, .1), **kwargs):
        self.weights = float(weights)

    def get_config(self):
        return {'weights':self.weights}

    def __call__(self, x):
        ev, _ = tf.linalg.eigh(tf.tensordot(tf.transpose(x), x, axes=1))
        pr = (np.sum(ev)**2)/np.sum(ev**2)
        l2 = tf.math.reduce_sum(tf.math.square(x))
        return l2*self.weights[0] + pr*self.weights[1]
