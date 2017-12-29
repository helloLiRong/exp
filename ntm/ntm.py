import sonnet as snt
import tensorflow as tf

from ntm import access


class NeuralTuringMachineCell(snt.RNNCore):
    def __init__(self,
                 access_config,
                 controller_config,
                 output_size,
                 name="ntm"):
        super(NeuralTuringMachineCell, self).__init__(name=name)
        with self._enter_variable_scope():
            self._controller = snt.LSTM(**controller_config)
            self._access = access.MemoryAccess(**access_config)

        self._access_output_size = self._access.output_size
        self._output_size = output_size

        self._output_size = tf.TensorShape([output_size])
        self._state_size = (
            self._access_output_size,
            self._access.state_size,
            self._controller.state_size)

    def _build(self, inputs, prev_state):
        prev_access_output, prev_access_state, prev_controller_state = prev_state

        batch_flatten = snt.BatchFlatten()
        controller_input = tf.concat(
            [batch_flatten(inputs), batch_flatten(prev_access_output)], 1)

        controller_output, controller_state = self._controller(
            controller_input, prev_controller_state)

        access_output, access_state = self._access(controller_output,
                                                   prev_access_state)
        output = tf.concat([controller_output, batch_flatten(access_output)], 1)
        output = snt.Linear(
            output_size=self._output_size.as_list()[0],
            name='output_linear')(output)
        return output, (access_output, access_state, controller_state)

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size
