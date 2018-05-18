import os
import numpy as np

import tensorflow as tf
from tensorflow.contrib.eager.python import tfe

from utils.data_utils import Corpus
from utils.basic_lstm import BasicLSTM

# enable eager mode
tf.enable_eager_execution()
tf.set_random_seed(0)
np.random.seed(0)

if not os.path.exists('weights/'):
    os.makedirs('weights/')

# Hyper-parameters
embed_size = 128
rnn_units = 1024
num_epochs = 10
num_samples = 1000  # number of words to be sampled
batch_size = 20
seq_length = 30
learning_rate = 0.002
global_clipping_norm = 0.5

# dataset loading
corpus = Corpus()
train_corpus = corpus.get_data('../data_ptb/train', batch_size)
vocab_size = len(corpus.dictionary)
num_batches = train_corpus.shape[-1] // seq_length

train_corpus = tf.constant(train_corpus, dtype=tf.int32)

print("Dataset shape : ", train_corpus.shape)
print("Vocabulary size : ", vocab_size)
print("Number of batches : ", num_batches)


class RNNLanguageModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, rnn_units):
        super(RNNLanguageModel, self).__init__()
        self.units = rnn_units
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.enbedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size)
        self.lstm = BasicLSTM(self.units, return_states=True, return_sequence=True)
        self.classifier = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, training=None, mask=None, initial_states=None):
        embeds = self.enbedding(inputs)

        output, [h, c] = self.lstm(embeds, initial_states=initial_states)

        # preserve the states
        self.states = [h, c]

        # Reshape output to (batch_size * sequence_length, hidden_size)
        output = tf.reshape(output, [-1, output.shape[2]])

        # Decode hidden states of all time steps
        output = self.classifier(output)

        return output


device = '/cpu:0' if tfe.num_gpus() == 0 else '/gpu:0'
with tf.device(device):
    # build model and optimizer
    model = RNNLanguageModel(vocab_size, embed_size, rnn_units)
    optimizer = tf.train.AdamOptimizer(0.001)

    # suggested fix ; can be incorporated inside `_eager_set_inputs` or `_set_input`
    # Fix = Use exactly one sample from the provided input dataset to determine input/output shape/s for the model
    dummy_x = np.zeros((1, 1))
    model._set_inputs(dummy_x)

    best_perplexity = 1e6
    saver = tfe.Saver(model.variables)

    if os.path.exists('weights/08_02_rnn_lm/') and tf.train.checkpoint_exists('weights/08_02_rnn_lm/weights.ckpt'):
        saver = tfe.Saver(model.variables)
        saver.restore('weights/08_02_rnn_lm/weights.ckpt')
        print("Restored model !")

    # train loop
    for epoch in range(num_epochs):
        # Set initial hidden and cell states
        initial_states = (tf.zeros([batch_size, rnn_units]), tf.zeros([batch_size, rnn_units]))

        for i in range(0, train_corpus.shape[1] - seq_length, seq_length):
            # Get mini-batch inputs and targets
            inputs = train_corpus[:, i:i + seq_length]
            targets = train_corpus[:, (i + 1):(i + 1) + seq_length]
            targets = tf.reshape(targets, [-1])

            # Forward pass
            with tf.GradientTape() as tape:
                outputs= model(inputs, initial_states=initial_states)
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=targets)
                loss = tf.reduce_mean(loss)

            # use only the final state
            h, c = model.states
            initial_states = [h[:, -1, :], c[:, -1, :]]

            # get and clip gradients
            gradients = tape.gradient(loss, model.variables)
            grad_vars = zip(gradients, model.variables)

            # update weights
            optimizer.apply_gradients(grad_vars, tf.train.get_or_create_global_step())

            step = (i + 1) // seq_length
            if step % 100 == 0:
                perplexity = np.exp(loss.numpy())

                print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                      .format(epoch + 1, num_epochs, step, num_batches, loss.numpy(), perplexity))

                if best_perplexity > perplexity:
                    best_perplexity = perplexity
                    saver.save('weights/08_02_rnn_lm/weights.ckpt')
                    print("Perplexity improved. Saving weights...")

    saver = tfe.Saver(model.variables)
    saver.restore('weights/08_02_rnn_lm/weights.ckpt')

    if not os.path.exists('language_model/'):
        os.makedirs('language_model/')

    # evaluation of model
    with open('language_model/sample.txt', 'w') as f:
        # Set intial hidden ane cell states
        initial_states = (tf.zeros([1, rnn_units]), tf.zeros([1, rnn_units]))

        # Select one word id randomly
        prob = tf.ones([1, vocab_size])
        input = tf.multinomial(prob, num_samples=1)

        for i in range(num_samples):
            # Forward propagate RNN
            output = model(input, initial_states=initial_states)

            # use only the final state
            h, c = model.states
            initial_states = [h[:, -1, :], c[:, -1, :]]

            # Sample a word id
            prob = tf.exp(output)
            word_id = tf.multinomial(prob, num_samples=1)[0, 0]

            # Fill input with sampled word id for the next time step
            input = tf.fill(input.shape, word_id)

            # File write
            word = corpus.dictionary.idx2word[word_id.numpy()]
            word = '\n' if word == '<eos>' else word + ' '
            f.write(word)

            if (i + 1) % 100 == 0:
                print('Sampled [{}/{}] words and save to {}'.format(i + 1, num_samples, 'language_model/sample.txt'))

