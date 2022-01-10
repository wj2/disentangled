import tensorflow as tf
import tensorflow_probability as tfp

@tf.keras.utils.register_keras_serializable(name='var13')
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

@tf.keras.utils.register_keras_serializable(name='l2pr3')
class L2PRRegularizer(tf.keras.regularizers.Regularizer):

    def __init__(self, weights=(.1, .1), **kwargs):
        self.weights = (float(weights[0]), float(weights[1]))

    def get_config(self):
        return {'weights':self.weights}

    def __call__(self, x):
        ev, _ = tf.linalg.eigh(tf.tensordot(tf.transpose(x), x, axes=1))
        pr = tf.math.divide(tf.math.square(tf.math.reduce_sum(ev)),
                            tf.math.reduce_sum(tf.math.square(ev)))
        pr_ratio = (x.shape[1] - pr)/x.shape[1]
        l2 = tf.math.reduce_sum(tf.math.square(x))
        return l2*self.weights[0] + pr_ratio*self.weights[1]

@tf.keras.utils.register_keras_serializable(name='l2pr_inv')
class L2PRRegularizerInv(tf.keras.regularizers.Regularizer):

    def __init__(self, weights=(.1, .1), **kwargs):
        self.weights = (float(weights[0]), float(weights[1]))

    def get_config(self):
        return {'weights':self.weights}

    def __call__(self, x):
        ev, _ = tf.linalg.eigh(tf.tensordot(tf.transpose(x), x, axes=1))
        pr = tf.math.divide(tf.math.square(tf.math.reduce_sum(ev)),
                            tf.math.reduce_sum(tf.math.square(ev)))
        l2 = tf.math.reduce_sum(tf.math.square(x))
        return l2*self.weights[0] - pr*self.weights[1]

class ConfusionRegularizer(tf.keras.regularizers.Regularizer):

    def __init__(self, weight=.1, n=10, **kwargs):
        self.weight = float(weight)
        self.n = n

    def get_config(self):
        return {'weight':self.weight}

    def __call__(self, x):
        n_eg = x.shape[0]
        cats = np.zeros(n_eg)
        cats[:int(n_eg/2)] = 1
        scores = np.zeros(self.n)
        for i in range(self.n):
            classes = np.random.choice(cats, n_eg, replace=False)
            svc = skc.LinearSVC()
            svc.fit(x, classes)
            scores[i] = svc.score(x, classes)
        out = self.weight*(scores - .5)**2
        return out
