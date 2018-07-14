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
image_size = 28
batch_size = 512
epochs = 6
num_classes = 10

# dataset loading
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((-1, image_size, image_size, 1))
x_test = x_test.reshape((-1, image_size, image_size, 1))

# one hot encode the labels. convert back to numpy as we cannot use a combination of numpy
# and tensors as input to keras
y_train_ohe = tf.one_hot(y_train, depth=num_classes).numpy()
y_test_ohe = tf.one_hot(y_test, depth=num_classes).numpy()


class ConvBNRelu(tf.keras.Model):
    def __init__(self, channels, strides=1, kernel=3, padding='same'):
        super(ConvBNRelu, self).__init__()
        self.conv =  tf.keras.layers.Conv2D(channels, (kernel, kernel), strides=(strides, strides), padding=padding,
                                            use_bias=False, kernel_initializer='he_normal')
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x


class InceptionBlock(tf.keras.Model):
    def __init__(self, channels, strides=1):
        super(InceptionBlock, self).__init__()
        self.channels = channels
        self.strides = strides

        self.conv1 = ConvBNRelu(channels, strides, kernel=1)
        self.conv2 = ConvBNRelu(channels, strides, kernel=3)
        self.conv3_1 = ConvBNRelu(channels, strides, kernel=3)
        self.conv3_2 = ConvBNRelu(channels, 1, kernel=3)
        self.maxpool = tf.keras.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')
        self.maxpool_conv = ConvBNRelu(channels, strides, kernel=1)

    def call(self, inputs, training=None, mask=None):
        x1 = self.conv1(inputs, training=training)

        x2 = self.conv2(inputs, training=training)

        x3_1 = self.conv3_1(inputs, training=training)
        x3_2 = self.conv3_2(x3_1, training=training)

        x4 = self.maxpool(inputs)
        x4 = self.maxpool_conv(x4, training=training)

        x = tf.keras.layers.concatenate([x1, x2, x3_2, x4], axis=-1)
        return x


class InceptionCIFAR(tf.keras.Model):
    def __init__(self, num_layers, num_classes, initial_filters=16, **kwargs):
        super(InceptionCIFAR, self).__init__(**kwargs)

        self.in_channels = initial_filters
        self.out_channels = initial_filters
        self.num_layers = num_layers
        self.initial_filters = initial_filters

        self.conv1 = ConvBNRelu(initial_filters)

        self.blocks = []

        # build all the blocks
        for block_id in range(num_layers):
            for layer_id in range(2):
                key = 'block_%d_%d' % (block_id + 1, layer_id + 1)
                if layer_id == 0:
                    block = InceptionBlock(self.out_channels, strides=2)
                else:
                    block = InceptionBlock(self.out_channels)

                self.in_channels = self.out_channels

                # "add" this block to this model. This is important
                setattr(self, key, block)

                self.blocks.append(block)

            self.out_channels *= 2

        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=None, mask=None):
        out = self.conv1(inputs, training=training)

        for block in self.blocks:
            out = block(out, training=training)

        out = self.avg_pool(out)
        out = self.fc(out)

        # softmax op does not exist on the gpu, so always use cpu
        with tf.device('/cpu:0'):
            output = tf.nn.softmax(out)

        return output


device = '/cpu:0' if tfe.num_gpus() == 0 else '/gpu:0'

with tf.device(device):
    # build model and optimizer
    model = InceptionCIFAR(2, num_classes)
    model.compile(optimizer=tf.train.AdamOptimizer(0.01), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # suggested fix ; can be incorporated inside `_eager_set_inputs` or `_set_input`
    # Fix = Use exactly one sample from the provided input dataset to determine input/output shape/s for the model
    dummy_x = tf.zeros((1, image_size, image_size, 1))
    model._set_inputs(dummy_x)

    # train
    model.fit(x_train, y_train_ohe, batch_size=batch_size, epochs=epochs,
              validation_data=(x_test, y_test_ohe), verbose=1)

    # evaluate on test set
    scores = model.evaluate(x_test, y_test_ohe, batch_size, verbose=1)
    print("Final test loss and accuracy :", scores)

    saver = tfe.Saver(model.variables)
    saver.save('weights/05_inception/weights.ckpt')