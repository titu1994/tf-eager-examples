import os
import numpy as np
from collections import OrderedDict

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
epochs = 26

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

# model definition
class CustomRegressor(tf.keras.Model):
    def __init__(self):
        super(CustomRegressor, self).__init__()
        # self.add_variable and self.add_weight are not yet supported
        self.custom_variables = OrderedDict()

        # we also use a keras layer along with a custom weight matrix
        self.hidden2 = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        if 'hidden' not in self.custom_variables:
            # this is equivalent to a Dense layer from Keras (same as hidden2)
            hidden = tf.get_variable('hidden1', shape=[inputs.shape[-1], 1], dtype=tf.float32,
                                     initializer=tf.keras.initializers.he_normal())
            self.custom_variables['hidden'] = hidden

        output1 = tf.matmul(inputs, self.custom_variables['hidden'])
        output1 = tf.keras.activations.relu(output1)

        output2 = self.hidden2(inputs)

        output = output1 + output2  # goofy layer ; just for demonstration purposes
        return output


def gradients(model, x, y):
    with tf.GradientTape() as tape:
        outputs = model(x)
        loss = tf.losses.mean_squared_error(y, outputs[:, 0])
        loss = tf.reduce_mean(loss)

    gradients = tape.gradient(loss, model.variables + list(model.custom_variables.values()))
    grad_vars = zip(gradients, model.variables + list(model.custom_variables.values()))
    return loss, grad_vars


device = '/cpu:0' if tfe.num_gpus() == 0 else '/gpu:0'

with tf.device(device):
    # build model and optimizer
    model = CustomRegressor()

    dummy_x = tf.zeros([1] + [x_train.shape[-1]])
    model.call(dummy_x)

    # Can no longer use Keras utility functions since we could not register the variable to keras properly
    # Whenever TF allows the addition of variables using Keras APIs, this will become easier like before
    optimizer = tf.train.AdamOptimizer(1.0)

    # wrap with datasets to make life slightly easier
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(batch_size).shuffle(100).repeat().prefetch(20)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size)

    # train the model
    num_batch_per_epoch = len(x_train) // batch_size + 1
    for e in range(epochs):

        # measure the losses
        train_loss = tfe.metrics.Mean()
        test_loss = tfe.metrics.Mean()

        for b, (x, y) in enumerate(train_dataset):
            loss, grads = gradients(model, x, y)
            optimizer.apply_gradients(grads, tf.train.get_or_create_global_step())

            # update the running training loss
            train_loss(loss)

            if b >= num_batch_per_epoch:
                break

        # evaluate after epoch
        iterator = test_dataset.make_one_shot_iterator()  # dont repeat any values from test set
        for x, y in iterator:
            preds = model(x)
            loss = tf.losses.mean_squared_error(y, preds[:, 0])

            test_loss(loss)

        print("Epoch %d: Train Loss = %0.4f | Test Loss = %0.4f\n" % (e + 1, train_loss.result(), test_loss.result()))

    # Make sure to add not just the "model" variables, but also the custom variables we added !
    saver = tfe.Saver(model.variables + list(model.custom_variables.values()))
    saver.save('weights/10_01_custom_models/weights.ckpt')
    print("Model saved")

    # Here we need to reset the keras internal backend first
    tf.keras.backend.clear_session()

    # Now we restore the model and predict again on test set
    model2 = CustomRegressor()

    # we need to run the model at least once to build all of the variables and the custom variables
    # make sure to build the model the same way, otherwise it wont find the weights in the checkpoints properly
    # safest option is to call model.call(tf_input_batch) explicitly
    model2.call(dummy_x)

    # ensure that you are loading both the Keras variables AND the custom variables
    saver2 = tfe.Saver(model2.variables + list(model2.custom_variables.values()))
    saver2.restore('weights/10_01_custom_models/weights.ckpt')
    print("Weights restored")

    # evaluate the results
    iterator = test_dataset.make_one_shot_iterator()  # dont repeat any values from test set
    test_loss = tfe.metrics.Mean()

    for x, y in iterator:
        preds = model2(x)
        loss = tf.losses.mean_squared_error(y, preds[:, 0])

        test_loss(loss)

    print("Test Loss = %0.4f\n" % (test_loss.result()))

