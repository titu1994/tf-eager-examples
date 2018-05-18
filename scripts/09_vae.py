import os
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.python.keras.datasets import mnist
from tensorflow.contrib.eager.python import tfe

# enable eager mode
tf.enable_eager_execution()
tf.set_random_seed(0)
np.random.seed(0)

if not os.path.exists('weights/'):
    os.makedirs('weights/')

# image grid
new_im = Image.new('L', (280, 280))

# Hyper-parameters
image_size = 784
h_dim = 512
z_dim = 5  # set to larger values for slightly better reconstructions but poorer sampling
num_epochs = 15
batch_size = 100
learning_rate = 1e-3

# dataset loading
(x_train, y_train), (_, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape((x_train.shape[0], 28 * 28))


class VAE(tf.keras.Model):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = tf.keras.layers.Dense(h_dim)
        self.fc2 = tf.keras.layers.Dense(z_dim)
        self.fc3 = tf.keras.layers.Dense(z_dim)
        self.fc4 = tf.keras.layers.Dense(h_dim)
        self.fc5 = tf.keras.layers.Dense(image_size)

    def encode(self, x):
        h = tf.nn.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        std = tf.exp(log_var * 0.5)
        eps = tf.random_normal(std.shape)
        return mu + eps * std

    def decode_logits(self, z):
        h = tf.nn.relu(self.fc4(z))
        return self.fc5(h)

    def decode(self, z):
        return tf.nn.sigmoid(self.decode_logits(z))

    def call(self, inputs, training=None, mask=None):
        mu, log_var = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        x_reconstructed_logits = self.decode_logits(z)
        return x_reconstructed_logits, mu, log_var


device = '/cpu:0' if tfe.num_gpus() == 0 else '/gpu:0'

with tf.device(device):
    # build model and optimizer
    model = VAE()
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # create the database iterator
    dataset = tf.data.Dataset.from_tensor_slices((x_train,))
    dataset = dataset.shuffle(batch_size * 5)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(10)

    num_batches = x_train.shape[0] // batch_size

    for epoch in range(num_epochs):
        for batch, (images,) in enumerate(dataset):

            with tf.GradientTape() as tape:
                # Forward pass
                x_reconstruction_logits, mu, log_var = model(images)

                # Compute reconstruction loss and kl divergence
                # For KL divergence, see Appendix B in VAE paper or http://yunjey47.tistory.com/43
                # Scaled by `image_size` for each individual pixel.
                reconstruction_loss = image_size * tf.nn.sigmoid_cross_entropy_with_logits(labels=images, logits=x_reconstruction_logits)
                kl_div = -0.5 * tf.reduce_sum(1. + log_var - tf.square(mu) - tf.exp(log_var), axis=-1)

                # Backprop and optimize
                loss = tf.reduce_mean(reconstruction_loss) + kl_div

            gradients = tape.gradient(loss, model.variables)
            grad_vars = zip(gradients, model.variables)
            optimizer.apply_gradients(grad_vars, tf.train.get_or_create_global_step())

            if (batch + 1) % 10 == 0:
                reconstruction_loss = tf.reduce_mean(reconstruction_loss)
                kl_div = tf.reduce_sum(kl_div)
                print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
                      .format(epoch + 1, num_epochs, batch + 1, num_batches, reconstruction_loss.numpy(), kl_div.numpy()))

            if batch > num_batches:
                break

        saver = tfe.Saver(model.variables)
        saver.save('weights/09_vae/weights.ckpt')

        # evaluate
        z = tf.random_normal((batch_size, z_dim))
        out = model.decode(z)  # decode with sigmoid
        out = tf.reshape(out, [-1, 28, 28]).numpy() * 255
        out = out.astype(np.uint8)

        index = 0
        for i in range(0, 280, 28):
            for j in range(0, 280, 28):
                im = out[index]
                im = Image.fromarray(im, mode='L')
                new_im.paste(im, (i, j))
                index += 1

        new_im.save('images/vae_sampled_epoch_%d.png' % (epoch + 1))

        # Save the reconstructed images of last batch
        out_logits, _, _ = model(images[:batch_size // 2])
        out = tf.nn.sigmoid(out_logits)  # out is just the logits, use sigmoid
        out = tf.reshape(out, [-1, 28, 28]).numpy() * 255

        images = tf.reshape(images[:batch_size // 2], [-1, 28, 28])

        x_concat = tf.concat([images, out], axis=0).numpy() * 255.
        x_concat = x_concat.astype(np.uint8)

        index = 0
        for i in range(0, 280, 28):
            for j in range(0, 280, 28):
                im = x_concat[index]
                im = Image.fromarray(im, mode='L')
                new_im.paste(im, (i, j))
                index += 1

        new_im.save('images/vae_reconstructed_epoch_%d.png' % (epoch + 1))
        print('New images saved !')
