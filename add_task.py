import numpy as np
import random as rand


class AddTask(object):
    def __init__(self, input_size,
                 min_step,
                 max_step,
                 real_max_step):
        self.input_size = input_size
        self.max_step = max_step
        self.min_step = min_step
        self.real_max_step = real_max_step

        self.plus_symbol = self.get_plus_symbol()
        self.start_symbol = self.get_start_symbol()
        self.end_symbol = self.get_end_symbol()
        self.blank_symbol = self.get_blank_symbol()
        self.splite_symbol = [[2] for _ in range(self.real_max_step)]

        self.symbol_dict = ((self.plus_symbol, '+'),
                            (self.start_symbol, '>'),
                            (self.end_symbol, '.'),
                            (self.blank_symbol, ''),
                            )

    def get_plus_symbol(self):
        plus_symbol = [0] * self.input_size
        plus_symbol[10] = 1
        return plus_symbol

    def get_start_symbol(self):
        start_symbol = np.zeros(self.input_size)
        start_symbol[11] = 1
        return start_symbol

    def get_end_symbol(self):
        end_symbol = np.zeros(self.input_size)
        end_symbol[12] = 1
        return end_symbol

    def get_blank_symbol(self):
        blank_symbol = np.zeros(self.input_size)
        blank_symbol[13] = 1
        return blank_symbol

    def _create_datas(self, batch_size, min_step, max_step):
        res_x = []
        res_y = []

        for i in range(batch_size):
            first_num_length = rand.randint(min_step, max_step)
            second_num_length = rand.randint(min_step, max_step)
            first_num_array = self.create_data_by_length(first_num_length)
            second_num_array = self.create_data_by_length(second_num_length)

            add_sum = int(self.array2num([first_num_array])) + int(self.array2num([second_num_array]))

            data_y = self.num2array(add_sum)

            first_num_array.append(self.get_plus_symbol())
            first_num_array.extend(second_num_array)
            first_num_array.append(self.get_end_symbol())
            length_x = len(first_num_array)
            for _ in range(length_x, self.real_max_step):
                first_num_array.append(self.get_blank_symbol())
            res_x.append(first_num_array)

            for _ in range(length_x):
                data_y.insert(0, self.get_blank_symbol())

            length_y = len(data_y)
            for _ in range(length_y, self.real_max_step):
                data_y.append(self.get_blank_symbol())
            res_y.append(data_y)
        return np.array(res_x), np.array(res_y)

    # def _create_datas(self, batch_size):
    #     res_x = []
    #     res_y = []
    #
    #     for i in range(batch_size):
    #         first_num_length = rand.randint(self.min_step, self.max_step)
    #         second_num_length = rand.randint(self.min_step, self.max_step)
    #         first_num_array = self.create_data_by_length(first_num_length)
    #         second_num_array = self.create_data_by_length(second_num_length)
    #
    #         add_sum = int(self.array2num([first_num_array])) + int(self.array2num([second_num_array]))
    #
    #         data_y = self.num2array(add_sum)
    #
    #         first_num_array.append(self.get_end_symbol())
    #         second_num_array.append(self.get_end_symbol())
    #         for _ in range(first_num_length + 1, self.real_max_step):
    #             first_num_array.append(self.get_blank_symbol())
    #
    #         for _ in range(second_num_length + 1, self.real_max_step):
    #             second_num_array.append(self.get_blank_symbol())
    #
    #         data_y.append(self.get_end_symbol())
    #         length = len(data_y)
    #         for _ in range(length, self.real_max_step):
    #             data_y.append(self.get_blank_symbol())
    #
    #         res_x.append(np.concatenate([first_num_array, self.splite_symbol, second_num_array], 1))
    #         res_y.append(data_y)
    #
    #     return np.array(res_x), np.array(res_y)
    def next_batch(self, batch_size):
        return self._create_datas(batch_size, self.min_step, self.max_step)

    def next_batch_curr(self, batch_size, min_step, curr_step):
        normal_batch_size = batch_size // 10 * 8
        extra_size = batch_size - normal_batch_size
        res_x, res_y = self._create_datas(normal_batch_size, curr_step, curr_step)

        step = rand.randint(1, self.max_step)
        extra_data_x, extra_data_y = self._create_datas(extra_size, 1, step)
        x = np.concatenate([res_x, extra_data_x], 0)
        y = np.concatenate([res_y, extra_data_y], 0)
        return x, y

    def create_data_by_length(self, length):
        assert length > 0

        res = []

        for i in range(length):
            step_data = [0 for _ in range(self.input_size)]
            if i == length - 1:
                index = rand.randint(1, 9)
            else:
                index = rand.randint(0, 9)
            step_data[index] = 1
            res.append(step_data)

        return res

    def array2num(self, arrays):
        strs = ''
        for data in arrays:
            for step in data:
                for key, val in self.symbol_dict:
                    if np.all(step == key):
                        strs += val
                        break
                else:
                    strs += str(np.argmax(step))
        return strs[::-1]

    def num2array(self, num):
        '''
        将数字转成数组
        :param num:
        :return:

        例如：58 =》 [[0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,1,0,0,0,0,0],]
        '''
        res = []
        while num > 0:
            a_data = [0 for _ in range(self.input_size)]
            a_data[num % 10] = 1
            res.append(a_data)
            num //= 10

        return res

    def show_result(self, predicts, inputs, labels, show_all=True):

        pass
