import sonnet as snt
import tensorflow as tf
import collections

LSTMState = collections.namedtuple('LSTMState', ('state', 'output'))


class SntRNN(snt.RNNCore):
    def __init__(self, hidden_size, name="Snt_RNN"):
        """Constructor of the module.

        Args:
          hidden_size: an int, size of the outputs of the module (without batch
              size).
          name: the name of the module.
        """
        super(SntRNN, self).__init__(name=name)
        self._hidden_size = hidden_size

    def _build(self, inputs, state):
        """Builds a TF subgraph that performs one timestep of computation."""
        prev_state, prev_output = state

        x = tf.concat([prev_output, inputs], 1)

        x_to_f = snt.Linear(self._hidden_size, name="x_to_f")
        x_to_i = snt.Linear(self._hidden_size, name="x_to_i")
        x_to_c = snt.Linear(self._hidden_size, name="x_to_c")
        x_to_o = snt.Linear(self._hidden_size, name="x_to_o")

        f = tf.sigmoid(x_to_f(x), name="f")
        i = tf.sigmoid(x_to_i(x), name="i")
        o = tf.sigmoid(x_to_o(x), name="o")
        c_head = tf.tanh(x_to_c(x), name="c_head")

        c_state = tf.multiply(f, prev_state) + tf.multiply(i, c_head)
        outputs = tf.multiply(o, c_state)

        return outputs, (c_state, outputs)

    @property
    def state_size(self):
        """Returns a description of the state size, without batch dimension."""
        return (tf.TensorShape([self._hidden_size]),
                tf.TensorShape([self._hidden_size]))

    @property
    def output_size(self):
        """Returns a description of the output size, without batch dimension."""
        return tf.TensorShape([self._hidden_size])

        # def initial_state(self, batch_size, dtype):
        #     """Returns an initial state with zeros, for a batch size and data type.
        #
        #     NOTE: This method is here only for illustrative purposes, the corresponding
        #     method in its superclass should be already doing this.
        #     """
        #     return tf.concat([self.zero_state(batch_size, dtype),
        #                       self.zero_state(batch_size, dtype)], 1)
