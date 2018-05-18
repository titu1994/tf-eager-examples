import tensorflow as tf


class BasicLSTM(tf.keras.Model):
    def __init__(self, units, return_sequence=False, return_states=False, **kwargs):
        super(BasicLSTM, self).__init__(**kwargs)
        self.units = units
        self.return_sequence = return_sequence
        self.return_states = return_states

        def bias_initializer(_, *args, **kwargs):
            # Unit forget bias from the paper
            # - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
            return tf.keras.backend.concatenate([
                tf.keras.initializers.Zeros()((self.units,), *args, **kwargs),  # input gate
                tf.keras.initializers.Ones()((self.units,), *args, **kwargs),  # forget gate
                tf.keras.initializers.Zeros()((self.units * 2,), *args, **kwargs),  # context and output gates
            ])

        self.kernel = tf.keras.layers.Dense(4 * units, use_bias=False)
        self.recurrent_kernel = tf.keras.layers.Dense(4 * units, kernel_initializer='glorot_uniform', bias_initializer=bias_initializer)

    def call(self, inputs, training=None, mask=None, initial_states=None):
        # LSTM Cell in pure TF Eager code
        # reset the states initially if not provided, else use those
        if initial_states is None:
            h_state = tf.zeros((inputs.shape[0], self.units))
            c_state = tf.zeros((inputs.shape[0], self.units))
        else:
            assert len(initial_states) == 2, "Must pass a list of 2 states when passing 'initial_states'"
            h_state, c_state = initial_states

        h_list = []
        c_list = []

        for t in range(inputs.shape[1]):
            # LSTM gate steps
            ip = inputs[:, t, :]
            z = self.kernel(ip)
            z += self.recurrent_kernel(h_state)

            z0 = z[:, :self.units]
            z1 = z[:, self.units: 2 * self.units]
            z2 = z[:, 2 * self.units: 3 * self.units]
            z3 = z[:, 3 * self.units:]

            # gate updates
            i = tf.keras.activations.sigmoid(z0)
            f = tf.keras.activations.sigmoid(z1)
            c = f * c_state + i * tf.nn.tanh(z2)

            # state updates
            o = tf.keras.activations.sigmoid(z3)
            h = o * tf.nn.tanh(c)

            h_state = h
            c_state = c

            h_list.append(h_state)
            c_list.append(c_state)

        hidden_outputs = tf.stack(h_list, axis=1)
        hidden_states = tf.stack(c_list, axis=1)

        if self.return_states and self.return_sequence:
            return hidden_outputs, [hidden_outputs, hidden_states]
        elif self.return_states and not self.return_sequence:
            return hidden_outputs[:, -1, :], [h_state, c_state]
        elif self.return_sequence and not self.return_states:
            return hidden_outputs
        else:
            return hidden_outputs[:, -1, :]