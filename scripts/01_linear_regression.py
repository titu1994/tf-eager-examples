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
epochs = 25

# dataset loading
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# normalization of dataset
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)

x_train = (x_train - mean) / (std + 1e-8)
x_test = (x_test - mean) / (std + 1e-8)

print('x train', x_train.shape, x_train.mean(), x_train.std())
print('y train', y_train.shape, y_train.mean(), y_train.std())
print('x test', x_test.shape, x_test.mean(), x_test.std())
print('y test', y_test.shape, y_test.mean(), y_test.std())

# model definition (canonical way)
class Regressor(tf.keras.Model):

    def __init__(self):
        super(Regressor, self).__init__()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        output = self.dense(inputs)
        return output


device = '/cpu:0' if tfe.num_gpus() == 0 else '/gpu:0'

with tf.device(device):
    # build model and optimizer
    model = Regressor()
    model.compile(optimizer=tf.train.GradientDescentOptimizer(0.05), loss='mse')

    # suggested fix for TF <= 1.9; can be incorporated inside `_eager_set_inputs` or `_set_input`
    # Fix = Use exactly one sample from the provided input dataset to determine input/output shape/s for the model
    dummy_x = np.zeros((1, 13))
    model._set_inputs(dummy_x)

    # train
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(x_test, y_test))

    # evaluate on test set
    scores = model.evaluate(x_test, y_test, batch_size, verbose=2)
    print("Test MSE :", scores)

    saver = tfe.Saver(model.variables)
    saver.save('weights/01_linear_regression/weights.ckpt')
