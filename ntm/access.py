import sonnet as snt
import tensorflow as tf


def _erase_and_write(memory, address, reset_weights, values):
    with tf.name_scope('erase_memory', values=[memory, address, reset_weights]):
        expand_address = tf.expand_dims(address, 3)
        reset_weights = tf.expand_dims(reset_weights, 2)
        weighted_resets = expand_address * reset_weights
        reset_gate = tf.reduce_prod(1 - weighted_resets, [1])
        memory *= reset_gate

    with tf.name_scope('additive_write', values=[memory, address, values]):
        add_matrix = tf.matmul(address, values, adjoint_a=True)
        memory += add_matrix

    return memory


class MemoryAccess(snt.RNNCore):
    def __init__(self,
                 memory_size=128,
                 word_size=20,
                 num_reads=1,
                 num_writes=1,
                 shift_range=1,
                 name='memory_access'):
        super(MemoryAccess, self).__init__(name=name)
        self._memory_size = memory_size
        self._word_size = word_size
        self._num_reads = num_reads
        self._num_writes = num_writes
        self._shift_range = shift_range

    def _build(self, inputs, prev_state):
        parse_inputs = self._parse_inputs(inputs)

        prev_memory, prev_read_weight, prev_write_weight = prev_state

        write_weights = self._get_weights(parse_inputs, prev_memory, prev_write_weight)
        memory = _erase_and_write(
            prev_memory,
            address=write_weights,
            reset_weights=parse_inputs['erase_vectors'],
            values=parse_inputs['write_vectors'])

        read_weights = self._get_weights(
            parse_inputs, memory, prev_read_weight)

        read_words = tf.matmul(memory, read_weights)
        return read_words, (memory, read_weights, write_weights)

    def _parse_inputs(self, inputs):
        def _linear(first_dim, second_dim, name, activation=None):
            """Returns a linear transformation of `inputs`, followed by a reshape."""
            linear = snt.Linear(first_dim * second_dim, name=name)(inputs)
            if activation is not None:
                linear = activation(linear, name=name + '_activation')
            return tf.reshape(linear, [-1, first_dim, second_dim])

        write_vectors = _linear(self._word_size, self._num_writes, 'write_vectors')

        erase_vectors = _linear(self._word_size, self._num_writes, 'erase_vectors',
                                tf.sigmoid)

        results = {}
        x_to_k = _linear(self._word_size, self._num_reads, 'x_to_k')
        x_to_b = snt.Linear(self._num_reads)
        x_to_g = snt.Linear(self._num_reads)
        x_to_s = _linear(2 * self._shift_range + 1, self._num_reads, name="x_to_s")
        x_to_y = snt.Linear(self._num_reads)

        results["keys"] = snt.BatchApply(tf.nn.relu, name="keys")(x_to_k)
        results["beta"] = tf.nn.relu(x_to_b(inputs), name="beta")
        results["gamma"] = tf.nn.sigmoid(x_to_g(inputs), name="gamma")
        results["shift"] = snt.BatchApply(tf.nn.softmax, name="shift")(x_to_s)
        results["sharp"] = tf.nn.relu(x_to_y(inputs), name="sharp") + 1
        results["write_vectors"] = write_vectors
        results["erase_vectors"] = erase_vectors
        return results

    def _get_weights(self, parse_inputs, prev_memory, prev_weight):
        keys = parse_inputs["keys"]
        beta = parse_inputs["beta"]
        gamma = parse_inputs["gamma"]
        shift = parse_inputs["shift"]
        sharp = parse_inputs["sharp"]

        content_addressed_w = self._get_content_addressing(prev_memory,
                                                           keys,
                                                           beta)
        gated_weightings = self.apply_interpolation(content_addressed_w, prev_weight, gamma)
        after_conv_shift = self.apply_conv_shift(gated_weightings, shift)
        new_weightings = self.sharp_weights(after_conv_shift, sharp)

        return new_weightings

    def _get_content_addressing(self, memory, keys, strengths):
        normalized_memory = tf.nn.l2_normalize(memory, 2)
        normalized_keys = tf.nn.l2_normalize(keys, 1)
        similarity = tf.matmul(normalized_memory, normalized_keys)
        strengths = tf.expand_dims(strengths, 1)

        return tf.nn.softmax(similarity * strengths, 1)

    def apply_interpolation(self, content_weights, prev_weights, interpolation_gate):
        interpolation_gate = tf.expand_dims(interpolation_gate, 1)
        gated_weighting = interpolation_gate * content_weights + (1.0 - interpolation_gate) * prev_weights

        return gated_weighting

    def apply_conv_shift(self, gated_weighting, shift_weighting):
        gated_weighting = tf.concat([tf.expand_dims(gated_weighting[:, -1, :], axis=-1),
                                     gated_weighting,
                                     tf.expand_dims(gated_weighting[:, 0, :], axis=-1)], 1)

        gated_weighting = tf.expand_dims(gated_weighting, 0)
        shift_weighting = tf.expand_dims(shift_weighting, -1)

        conv = tf.nn.conv2d(
            gated_weighting,
            shift_weighting,
            strides=[1, 1, 1, 1],
            padding="VALID")

        return tf.squeeze(conv, axis=0)

    def sharp_weights(self, after_conv_shift, sharp_gamma):
        sharp_gamma = tf.expand_dims(sharp_gamma, 1)
        powed_conv_w = tf.pow(after_conv_shift, sharp_gamma)
        return powed_conv_w / tf.expand_dims(tf.reduce_sum(powed_conv_w, 1), 1)

    def update_memory(self, memory_matrix, write_weighting, add_vector, erase_vector):
        add_vector = tf.expand_dims(add_vector, 1)
        erase_vector = tf.expand_dims(erase_vector, 1)

        erasing = memory_matrix * (1 - tf.matmul(write_weighting, erase_vector))
        writing = tf.matmul(write_weighting, add_vector)
        updated_memory = erasing + writing

        return updated_memory

    @property
    def state_size(self):
        """Returns a tuple of the shape of the state tensors."""
        return (
            tf.TensorShape([self._memory_size, self._word_size]),
            tf.TensorShape([self._memory_size, self._num_reads]),
            tf.TensorShape([self._memory_size, self._num_writes]),
        )

    @property
    def output_size(self):
        """Returns the output shape."""
        return tf.TensorShape([self._word_size, self._num_reads])
