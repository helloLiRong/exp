import functools
import tensorflow as tf
import sonnet as snt

# import dnc
from SntRNN import SntRNN
from ntm.ntm import NeuralTuringMachineCell
from dnc.dnc import DNC


def doublewrap(func):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """

    @functools.wraps(func)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return func(args[0])
        else:
            return lambda wrapee: func(wrapee, *args, **kwargs)

    return decorator


@doublewrap
def define_scope(func, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + func.__name__
    name = scope or func.__name__

    @property
    @functools.wraps(func)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, func(self))
        return getattr(self, attribute)

    return decorator


class MyRNNModel:
    def __init__(self, x, label,
                 hidden_units, max_step, input_size, output_size,
                 access_config, lr=1e-4, ):
        self.input = x
        self.label = label
        self.hidden_units = hidden_units
        self.max_step = max_step
        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.access_config = access_config
        # 对 weights biases 初始值的定义
        # self.weights = {
        #     # shape (28, 128)
        #     'in': tf.get_variable("weight_in", [self.input_size, self.hidden_units],
        #                           initializer=tf.truncated_normal_initializer(mean=0, stddev=1)),
        #     # shape (128, 10)
        #     'out': tf.get_variable("weight_out", [self.hidden_units, self.output_size],
        #                            initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
        # }
        #
        # self.biases = {
        #     # shape (128, )
        #     'in': tf.get_variable("biase_in", shape=[self.hidden_units, ], initializer=tf.constant_initializer(0.5)),
        #     # shape (10, )
        #     'out': tf.get_variable("biase_out", shape=[self.output_size, ], initializer=tf.constant_initializer(0.5))
        # }

        self.prediction
        self.optimize
        self.cost
        self.accuracy

    # @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    @define_scope
    def prediction(self):
        # lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_units, forget_bias=1.0, state_is_tuple=True)
        # cell = SntRNN(self.hidden_units)
        # initial_state = lstm_cell.initial_state(batch_size, tf.float32)
        cell = DNC(self.access_config,
                   {"hidden_size": self.hidden_units},
                   self.output_size)
        initial_state = cell.initial_state(128, tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(cell, self.input,
                                                 initial_state=initial_state,
                                                 dtype=tf.float32,
                                                 time_major=False)

        # print(outputs)
        # outputs = tf.reshape(outputs, [-1, self.hidden_units])
        # lin = snt.Linear(self.output_size)
        # results = lin(outputs)
        # results = tf.matmul(outputs, self.weights['out']) + self.biases['out']
        # results = tf.reshape(results, [-1, self.max_step, self.output_size])

        return outputs

    @define_scope
    def optimize(self):
        # optimizer = tf.train.AdamOptimizer(self.lr)
        optimizer = tf.train.AdamOptimizer()
        return optimizer.minimize(self.cost)

    @define_scope
    def cost(self):
        pred = tf.reshape(self.prediction, [-1, self.output_size])
        labels = tf.reshape(self.label, [-1, self.output_size])
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=labels))

    @define_scope
    def accuracy(self):
        predicts = tf.argmax(self.prediction, 2)
        labels = tf.argmax(self.label, 2)
        correct_num = tf.cast(tf.reduce_all(tf.equal(predicts, labels), 1), tf.float32)

        return tf.reduce_mean(correct_num)
