import os
import numpy as np

import tensorflow as tf
from tensorflow.python.keras.datasets import mnist
from tensorflow.contrib.eager.python import tfe

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


class RNN(tf.keras.Model):
    def __init__(self, units, num_classes, num_layers=1):
        super(RNN, self).__init__()
        self.cells = [tf.keras.layers.LSTMCell(units, implementation=2) for _ in range(num_layers)]
        self.rnn = tf.keras.layers.RNN(self.cells, unroll=True)  # extremely slow if not unrolled - probably because it is using K.rnn() internally.
        self.classifier = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=None, mask=None):
        # this approach is canonical, but slow
        # probably will be fixed in later versions of Tensorflow

        x = self.rnn(inputs)
        output = self.classifier(x)

        # softmax op does not exist on the gpu, so always use cpu
        with tf.device('/cpu:0'):
            output = tf.nn.softmax(output)

        return output


device = '/cpu:0' if tfe.num_gpus() == 0 else '/gpu:0'

with tf.device(device):
    # build model and optimizer
    model = RNN(units, num_classes, num_layers=2)
    model.compile(optimizer=tf.train.AdamOptimizer(0.01), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # suggested fix ; can be incorporated inside `_eager_set_inputs` or `_set_input`
    # Fix = Use exactly one sample from the provided input dataset to determine input/output shape/s for the model
    dummy_x = tf.zeros((1, 28, 28))
    model._set_inputs(dummy_x)

    # train
    model.fit(x_train, y_train_ohe, batch_size=batch_size, epochs=epochs,
              validation_data=(x_test, y_test_ohe), verbose=1)

    # evaluate on test set
    scores = model.evaluate(x_test, y_test_ohe, batch_size, verbose=1)
    print("Final test loss and accuracy :", scores)

    saver = tfe.Saver(model.variables)
    saver.save('weights/06_01_rnn/weights.ckpt')