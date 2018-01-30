import numpy as np
import random as rand
import tensorflow as tf
from my_lstm import MyRNNModel


class CopyTask(object):
    def __init__(self, input_size,
                 min_step,
                 max_step,
                 real_max_step,
                 batch_size):
        self.input_size = input_size
        self.extra_bit = 3
        self.real_input_size = self.input_size + self.extra_bit
        self.output_size = 2 ** input_size + 1
        self.max_step = max_step
        self.min_step = min_step
        self.real_max_step = real_max_step
        self.batch_size = batch_size
        self.start_symbol = self.get_start_symbol()
        self.end_symbol = self.get_end_symbol()
        self.blank_symbol = self.get_blank_symbol()
        self.output_blank_symbol = self.get_output_blank_symbol()
        self.output_start_symbol = self.get_output_start_symbol()
        self.output_end_symbol = self.get_output_end_symbol()

        self.symbol_dict = ((self.start_symbol, '.'),
                            (self.end_symbol, '.'),
                            (self.output_start_symbol, "."),
                            (self.output_end_symbol, "."),
                            )

    def get_start_symbol(self):
        start_symbol = np.zeros(self.real_input_size)
        start_symbol[self.input_size] = 1
        return start_symbol

    def get_end_symbol(self):
        end_symbol = np.zeros(self.real_input_size)
        end_symbol[self.input_size + 1] = 1
        return end_symbol

    def get_blank_symbol(self):
        blank_symbol = np.zeros(self.real_input_size)
        blank_symbol[self.input_size + 2] = 1
        return blank_symbol

    def get_output_blank_symbol(self):
        blank_symbol = np.zeros(self.output_size)
        # blank_symbol[self.output_size - 1] = 1
        return blank_symbol

    def get_output_start_symbol(self):
        symbol = np.zeros(self.output_size)
        symbol[self.output_size - 2] = 1
        return symbol

    def get_output_end_symbol(self):
        symbol = np.zeros(self.output_size)
        symbol[self.output_size - 3] = 1
        return symbol

    def _create_datas(self, batch_size, min_step, max_step):
        res_x = []
        res_y = []

        for _ in range(batch_size):
            step = rand.randint(min_step, max_step)
            valid_data = np.random.randint(0, 2, [step, self.input_size])
            valid_data_x = np.concatenate([valid_data, np.zeros([step, self.extra_bit])], 1)
            valid_data_x = np.concatenate([valid_data_x, [self.get_end_symbol()]], 0)
            res_x.append(np.concatenate(
                [valid_data_x, [self.blank_symbol for _ in range(step + 1, self.real_max_step)]], 0)
            )

            valid_data_y = np.zeros([step, self.output_size])

            for i in range(step):
                index = self._bit2num(valid_data[i])
                valid_data_y[i][index] = 1

            # valid_data_y = np.concatenate(
            #     [[self.get_output_start_symbol()], valid_data_y, [self.get_output_end_symbol()]], 0)
            output_blanks = np.array([self.output_blank_symbol for _ in range(step, self.real_max_step)])

            valid_data_y = np.concatenate([output_blanks[:step + 1, :], valid_data_y, output_blanks[step + 1:, :]], 0)
            res_y.append(valid_data_y)
        return np.array(res_x), np.array(res_y)

    def _bit2num(self, bits):
        return int(sum(
            map(lambda m, n: m * (2 ** n), bits, range(self.input_size))
        ))

    def next_batch(self, batch_size):
        return self._create_datas(batch_size, self.min_step, self.max_step)

    def next_batch_for_test(self, batch_size, min_step, max_step):
        return self._create_datas(batch_size, min_step, max_step)

    def next_batch_curr(self, batch_size, min_step, max_step):
        normal_batch_size = batch_size // 10 * 8
        extra_size = batch_size - normal_batch_size
        res_x, res_y = self._create_datas(normal_batch_size, min_step, max_step)

        step = rand.randint(1, max_step)
        extra_data_x, extra_data_y = self._create_datas(extra_size, 1, step)
        x = np.concatenate([res_x, extra_data_x], 0)
        y = np.concatenate([res_y, extra_data_y], 0)
        return x, y

    def get_model(self, hidden_units, access_config, lr, clip_value):
        x = tf.placeholder(tf.float32,[None,])
        y = tf.placeholder(tf.float32)

        model = MyRNNModel(x, y,
                           hidden_units,
                           self.real_max_step,
                           self.input_size,
                           self.output_size,
                           access_config,
                           self.batch_size, lr, clip_value
                           )
        return model, x, y

    def show_result(self, predicts, inputs, labels, show_all=True):
        res = []
        for i in range(len(predicts)):
            predicts_strs = ''.join(
                [str(x).rjust(3, " ") for x in np.argmax(predicts[i], 1)])
            # if x != self.output_size - 1 and x != self.output_size - 2 and x != self.output_size - 3])
            labels_strs = ''.join([str(x).rjust(3, " ") for x in np.argmax(labels[i], 1)])
            # if x != self.output_size - 1 and x != self.output_size - 2 and x != self.output_size - 3])

            strs = ' predicts: %s \n labels:   %s\n\n' % (predicts_strs, labels_strs)
            res.append(strs)

        return res

# task = CopyTask(4, 1, 5, 10)
# x, y = task.next_batch(10)
# print(y)
