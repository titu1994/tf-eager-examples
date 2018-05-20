import os
import numpy as np

import tensorflow as tf
from tensorflow.python.keras.datasets import mnist
from tensorflow.contrib.eager.python import tfe

from utils.basic_lstm import BasicLSTM

# enable eager mode
tf.enable_eager_execution()
tf.set_random_seed(0)
np.random.seed(0)

if not os.path.exists('weights/'):
    os.makedirs('weights/')

# constants
units = 128
batch_size = 100
epochs = 2
num_classes = 10

# dataset loading
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((-1, 28, 28))  # 28 timesteps, 28 inputs / timestep
x_test = x_test.reshape((-1, 28, 28))  # 28 timesteps, 28 inputs / timeste

# one hot encode the labels. convert back to numpy as we cannot use a combination of numpy
# and tensors as input to keras
y_train_ohe = tf.one_hot(y_train, depth=num_classes).numpy()
y_test_ohe = tf.one_hot(y_test, depth=num_classes).numpy()

print('x train', x_train.shape)
print('y train', y_train_ohe.shape)
print('x test', x_test.shape)
print('y test', y_test_ohe.shape)


class BasicBidirectionalLSTMModel(tf.keras.Model):
    def __init__(self, units, num_classes):
        super(BasicBidirectionalLSTMModel, self).__init__()
        self.units = units
        self.forward_lstm = BasicLSTM(units)
        self.backward_lstm = BasicLSTM(units)
        self.classifier = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=None, mask=None):
        reverse_inputs = inputs[:, ::-1, :]  # reverse the timesteps

        h_forward = self.forward_lstm(inputs)  # forward
        h_backward = self.backward_lstm(reverse_inputs)  # backward

        # concatenate mode
        h = tf.keras.layers.concatenate([h_forward, h_backward], axis=-1)
        output = self.classifier(h)

        # softmax op does not exist on the gpu, so always use cpu
        with tf.device('/cpu:0'):
            output = tf.nn.softmax(output)

        return output


device = '/cpu:0' if tfe.num_gpus() == 0 else '/gpu:0'

with tf.device(device):
    # build model and optimizer
    model = BasicBidirectionalLSTMModel(units, num_classes)
    model.compile(optimizer=tf.train.AdamOptimizer(0.01), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # suggested fix ; can be incorporated inside `_eager_set_inputs` or `_set_input`
    # Fix = Use exactly one sample from the provided input dataset to determine input/output shape/s for the model
    dummy_x = tf.zeros((1, 28, 28))
    model.call(dummy_x)

    # train
    model.fit(x_train, y_train_ohe, batch_size=batch_size, epochs=epochs,
              validation_data=(x_test, y_test_ohe), verbose=1)

    # evaluate on test set
    scores = model.evaluate(x_test, y_test_ohe, batch_size, verbose=1)
    print("Final test loss and accuracy :", scores)

    saver = tfe.Saver(model.variables)
    saver.save('weights/07_02_bi_rnn/weights.ckpt')