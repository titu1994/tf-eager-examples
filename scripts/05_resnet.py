import os
import numpy as np

import tensorflow as tf
from tensorflow.python.keras.datasets import fashion_mnist
from tensorflow.contrib.eager.python import tfe

# enable eager mode
tf.enable_eager_execution()
tf.set_random_seed(0)
np.random.seed(0)

if not os.path.exists('weights/'):
    os.makedirs('weights/')

# constants
image_size = 28
batch_size = 128
epochs = 10
num_classes = 10

# dataset loading
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape((-1, image_size, image_size, 1))
x_test = x_test.reshape((-1, image_size, image_size, 1))
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# one hot encode the labels. convert back to numpy as we cannot use a combination of numpy
# and tensors as input to keras
y_train_ohe = tf.one_hot(y_train, depth=num_classes).numpy()
y_test_ohe = tf.one_hot(y_test, depth=num_classes).numpy()

# 3x3 convolution
def conv3x3(channels, stride=1, kernel=(3, 3)):
    return tf.keras.layers.Conv2D(channels, kernel, strides=(stride, stride), padding='same', use_bias=False,
                                  kernel_initializer=tf.variance_scaling_initializer())

class ResnetBlock(tf.keras.Model):

    def __init__(self, channels, strides=1, residual_path=False):
        super(ResnetBlock, self).__init__()
        self.channels = channels
        self.strides = strides
        self.residual_path = residual_path

        self.conv1 = conv3x3(channels, strides)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = conv3x3(channels)
        self.bn2 = tf.keras.layers.BatchNormalization()

        if residual_path:
            self.down_conv = conv3x3(channels, strides, kernel=(1, 1))
            self.down_bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        residual = inputs

        x = self.bn1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)

        if self.residual_path:
            residual = self.down_bn(inputs, training=training)
            residual = tf.nn.relu(residual)
            residual = self.down_conv(residual)

        x = x + residual
        return x


class ResNet(tf.keras.Model):

    def __init__(self, block_list, num_classes, initial_filters=16, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.num_blocks = len(block_list)
        self.block_list = block_list

        self.in_channels = initial_filters
        self.out_channels = initial_filters
        self.conv_initial = conv3x3(self.out_channels)

        self.blocks = []

        # build all the blocks
        for block_id in range(len(block_list)):
            for layer_id in range(block_list[block_id]):
                key = 'block_%d_%d' % (block_id + 1, layer_id + 1)
                if block_id != 0 and layer_id == 0:
                    block = ResnetBlock(self.out_channels, strides=2, residual_path=True)
                else:
                    if self.in_channels != self.out_channels:
                        residual_path = True
                    else:
                        residual_path = False
                    block = ResnetBlock(self.out_channels, residual_path=residual_path)

                self.in_channels = self.out_channels

                # "register" this block to this model ; Without this, weights wont update.
                setattr(self, key, block)

                self.blocks.append(block)

            self.out_channels *= 2

        self.final_bn = tf.keras.layers.BatchNormalization()
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=None, mask=None):
        out = self.conv_initial(inputs)

        # forward pass through all the blocks
        # build all the blocks
        for block in self.blocks:
            out = block(out, training=training)

        out = self.final_bn(out)
        out = tf.nn.relu(out)

        out = self.avg_pool(out)
        out = self.fc(out)

        # softmax op does not exist on the gpu, so always use cpu
        with tf.device('/cpu:0'):
            output = tf.nn.softmax(out)

        return output


device = '/cpu:0' if tfe.num_gpus() == 0 else '/gpu:0'

with tf.device(device):

    # build model and optimizer
    model = ResNet([2, 2, 2], num_classes)
    model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # suggested fix ; can be incorporated inside `_eager_set_inputs` or `_set_input`
    # Fix = Use exactly one sample from the provided input dataset to determine input/output shape/s for the model
    dummy_x = tf.zeros((1, image_size, image_size, 1))
    model._set_inputs(dummy_x)

    print("Number of variables in the model :", len(model.variables))
    model.summary()

    # train
    model.fit(x_train, y_train_ohe, batch_size=batch_size, epochs=epochs,
              validation_data=(x_test, y_test_ohe), verbose=1)

    # evaluate on test set
    scores = model.evaluate(x_test, y_test_ohe, batch_size, verbose=1)
    print("Final test loss and accuracy :", scores)

    saver = tfe.Saver(model.variables)
    saver.save('weights/05_resnet/weights.ckpt')