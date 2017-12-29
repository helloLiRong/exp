import tensorflow as tf
from my_lstm import MyRNNModel
from add_task import AddTask
from copy_task import CopyTask
import time
import numpy as np

FLAGS = tf.flags.FLAGS

# Model parameters
# tf.flags.DEFINE_string("task", "add", "task name")
tf.flags.DEFINE_string("task", "copy", "task name")
tf.flags.DEFINE_string("model", "dnc", "model name")
tf.flags.DEFINE_integer("lr", 1e-3, "learning rate.")
tf.flags.DEFINE_integer("training_iters", 5000, "The number of training steps.")
tf.flags.DEFINE_integer("batch_size", 128, "The number of a batch data.")
tf.flags.DEFINE_integer("input_size", 4, "The number of input vector size in a step.")
tf.flags.DEFINE_integer("max_steps", 50, "The max steps.")
tf.flags.DEFINE_integer("hidden_units", 200, "The number of hidden units")
# tf.flags.DEFINE_integer("output_size", 14, "The number of output classes")
tf.flags.DEFINE_integer("output_size", 2 ** FLAGS.input_size + 1, "The number of output classes")
tf.flags.DEFINE_integer("output_keep_prob", 1, "dropout percentage")
tf.flags.DEFINE_integer("train_data_num", 10000, "The number of train data")
tf.flags.DEFINE_integer("test_data_num", 128, "The number of test data")
tf.flags.DEFINE_integer("min_train_time_step", 1, "The valid min step when training")
tf.flags.DEFINE_integer("max_train_time_step", 10, "The valid max step when training")
tf.flags.DEFINE_integer("test_min_train_time_step", 1, "The valid min step when testing")
tf.flags.DEFINE_integer("test_max_train_time_step", 2, "The valid max step when testing")

tf.flags.DEFINE_boolean("is_train", True, "we will train this model")
tf.flags.DEFINE_boolean("need_load_parameter", True, "need to load parameter or not")
tf.flags.DEFINE_string("save_path", "checkpoint_%s_%s/net.ckpt" % (FLAGS.task, FLAGS.model), "we will train this model")

tf.flags.DEFINE_integer("memory_size", 64, "The number of memory slots.")
tf.flags.DEFINE_integer("word_size", 20, "The width of each memory slot.")
tf.flags.DEFINE_integer("num_write_heads", 1, "Number of memory write heads.")
tf.flags.DEFINE_integer("num_read_heads", 1, "Number of memory read heads.")


def train():
    data_config = {
        "input_size": FLAGS.input_size,
        "min_step": FLAGS.min_train_time_step,
        "max_step": FLAGS.max_train_time_step,
        "real_max_step": FLAGS.max_steps
    }

    access_config = {
        "memory_size": FLAGS.memory_size,
        "word_size": FLAGS.word_size,
        "num_reads": FLAGS.num_read_heads,
        "num_writes": FLAGS.num_write_heads,
    }

    task = get_task(**data_config)
    model, x, y = task.get_model(FLAGS.hidden_units,
                                 access_config,
                                 FLAGS.lr)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver()
    if FLAGS.need_load_parameter:
        saver.restore(sess, FLAGS.save_path)

    step = 0
    input_length = 1
    t_accuracy = 0
    while input_length <= FLAGS.max_train_time_step and t_accuracy < 0.999:

        batch_xs, batch_ys = task.next_batch_curr(FLAGS.batch_size, 1, 10)

        sess.run(model.optimize, {x: batch_xs, y: batch_ys})

        if step % 100 == 0:
            saver.save(sess, FLAGS.save_path)

            cost_value, t_accuracy = 0, 0
            test_runs = 10
            for _ in range(test_runs):
                test_x, test_y = task.next_batch_curr(FLAGS.test_data_num, 1, 10)
                temp_predicts, temp_cost, temp_test = sess.run(
                    [model.prediction, model.cost, model.accuracy],
                    feed_dict={x: test_x, y: test_y})
                cost_value += temp_cost
                t_accuracy += temp_test

            train_accuracy = sess.run(model.accuracy, feed_dict={x: batch_xs, y: batch_ys})
            cost_value /= test_runs
            t_accuracy /= test_runs
            print("time: %s, step: %d, cost: %.5f, train: %.5f, test: %.5f" %
                  (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                   , step, cost_value, train_accuracy, t_accuracy))

            if t_accuracy > 0.9 and input_length != FLAGS.max_train_time_step:
                input_length += 1
        step += 1


def test():
    data_config = {
        "input_size": FLAGS.input_size,
        "min_step": 10,
        "max_step": 10,
        "real_max_step": FLAGS.max_steps
    }

    access_config = {
        "memory_size": FLAGS.memory_size,
        "word_size": FLAGS.word_size,
        "num_reads": FLAGS.num_read_heads,
        "num_writes": FLAGS.num_write_heads,
    }

    task = get_task(**data_config)
    model, x, y = task.get_model(FLAGS.hidden_units,
                                 access_config,
                                 FLAGS.lr)

    test_x, test_y = task.next_batch(128)

    sess = tf.Session()
    saver = tf.train.Saver()

    saver.restore(sess, FLAGS.save_path)

    test_predicts, cost_value, t_accuracy = sess.run(
        [model.prediction, model.cost, model.accuracy],
        feed_dict={x: test_x, y: test_y})

    for s in task.show_result(test_predicts, test_x, test_y):
        print(s)
    print("test: %.5f" % t_accuracy)


def get_task(**data_config):
    if FLAGS.task == "add":
        return AddTask(**data_config)
    if FLAGS.task == 'copy':
        return CopyTask(**data_config)


def main(unused_argv):
    tf.logging.set_verbosity(3)
    if FLAGS.is_train:
        train()
    else:
        test()


if __name__ == '__main__':
    tf.app.run()
