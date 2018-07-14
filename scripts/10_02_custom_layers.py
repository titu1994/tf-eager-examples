import os
import numpy as np

import tensorflow as tf
from tensorflow.python.keras.datasets import boston_housing
from tensorflow.contrib.eager.python import tfe

# enable eager mode
tf.enable_eager_execution()
tf.set_random_seed(0)
np.random.seed(0)

if not os.path.exists('weights/'):
    os.makedirs('weights/')

# constants
batch_size = 128
epochs = 24

# dataset loading
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# normalization of dataset
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)

x_train = (x_train - mean) / (std + 1e-8)
x_test = (x_test - mean) / (std + 1e-8)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print('x train', x_train.shape, x_train.mean(), x_train.std())
print('y train', y_train.shape, y_train.mean(), y_train.std())
print('x test', x_test.shape, x_test.mean(), x_test.std())
print('y test', y_test.shape, y_test.mean(), y_test.std())


# A "Custom" layer which mimics the Dense layer from Keras
class CustomLayer(tf.keras.layers.Layer):

    def __init__(self, dim, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        self.dim = dim

    # change the "_" for the input shape to some variable name, and build on first call !
    def build(self, input_shape):
        # add variable / add_weights works inside Layers but not Models !
        self.kernel = self.add_variable('kernel',
                                        shape=[input_shape[-1], self.dim],
                                        dtype=tf.float32,
                                        initializer=tf.keras.initializers.he_normal())

        # Do NOT forget to call this line, otherwise multiple model variables will be built with the same name
        # This cannot happen inside Keras layers, and therefore the model will not be Checkpoint-able.
        # It also wont train properly.
        #
        self.built = True

    def call(self, inputs, **kwargs):
        return tf.matmul(inputs, self.kernel)


# model definition
class CustomRegressor(tf.keras.Model):
    def __init__(self):
        super(CustomRegressor, self).__init__()
        # self.add_variable and self.add_weight are not yet supported inside a Model
        # However, since we created a custom layer (Dense layer), we *can* attach it to this model
        # just like other layers !
        self.hidden1 = CustomLayer(1)

        # we also use a keras layer along with a custom weight matrix
        self.hidden2 = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        output1 = self.hidden1(inputs)
        output1 = tf.keras.activations.relu(output1)

        output2 = self.hidden2(inputs)

        output = output1 + output2  # goofy model ; just for demonstration purposes
        return output


device = '/cpu:0' if tfe.num_gpus() == 0 else '/gpu:0'

with tf.device(device):
    # build model and optimizer
    model = CustomRegressor()
    model.compile(optimizer=tf.train.AdamOptimizer(1.), loss='mse')

    # suggested fix for TF <= 2.0; can be incorporated inside `_eager_set_inputs` or `_set_input`
    # Fix = Use exactly one sample from the provided input dataset to determine input/output shape/s for the model
    dummy_x = tf.zeros((1, 13))
    model.call(dummy_x)

    # Now that we have a "proper" Keras layer, we can rely on Model utility functions again !

    # train
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(x_test, y_test))

    # evaluate on test set
    scores = model.evaluate(x_test, y_test, batch_size, verbose=2)
    print("Test MSE :", scores)

    saver = tfe.Saver(model.variables)
    saver.save('weights/10_02_custom_layers/weights.ckpt')
