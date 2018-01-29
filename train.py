import time

import tensorflow as tf
from add_task import AddTask

from task.copy_task import CopyTask

FLAGS = tf.flags.FLAGS

# Model parameters
# tf.flags.DEFINE_string("task", "add", "task name")
tf.flags.DEFINE_string("task", "copy", "task name")
tf.flags.DEFINE_string("model", "dnc", "model name")
tf.flags.DEFINE_integer("lr", 1e-4, "learning rate.")
tf.flags.DEFINE_integer("training_iters", 5000, "The number of training steps.")
tf.flags.DEFINE_integer("batch_size", 64, "The number of a batch data.")
tf.flags.DEFINE_integer("input_size", 8, "The number of input vector size in a step.")
tf.flags.DEFINE_integer("max_steps", 50, "The max steps.")
tf.flags.DEFINE_integer("hidden_units", 100, "The number of hidden units")
# tf.flags.DEFINE_integer("output_size", 14, "The number of output classes")
tf.flags.DEFINE_integer("output_size", 2 ** FLAGS.input_size + 1, "The number of output classes")
tf.flags.DEFINE_integer("output_keep_prob", 1, "dropout percentage")
tf.flags.DEFINE_integer("train_data_num", 10000, "The number of train data")
tf.flags.DEFINE_integer("test_data_num", 128, "The number of test data")

tf.flags.DEFINE_integer("clip_value", 0,
                        "Maximum absolute value of controller and dnc outputs.")
tf.flags.DEFINE_integer("min_train_time_step", 1, "The valid min step when training")
tf.flags.DEFINE_integer("max_train_time_step", 3, "The valid max step when training")
tf.flags.DEFINE_integer("test_min_train_time_step", 1, "The valid min step when testing")
tf.flags.DEFINE_integer("test_max_train_time_step", 2, "The valid max step when testing")

tf.flags.DEFINE_boolean("is_train", True, "we will train this model")
tf.flags.DEFINE_boolean("need_load_parameter", False, "need to load parameter or not")
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
        "real_max_step": FLAGS.max_steps,
        "batch_size": FLAGS.batch_size
    }

    access_config = {
        "memory_size": FLAGS.memory_size,
        "word_size": FLAGS.word_size,
        "num_reads": FLAGS.num_read_heads,
        "num_writes": FLAGS.num_write_heads,
    }

    clip_value = FLAGS.clip_value
    task = get_task(**data_config)
    model, x, y = task.get_model(FLAGS.hidden_units,
                                 access_config,
                                 FLAGS.lr, clip_value)

    sess = tf.Session()

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.save_path + '/train' + str(time.time()),
                                         sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.save_path + '/test' + str(time.time()))

    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver()
    if FLAGS.need_load_parameter:
        saver.restore(sess, FLAGS.save_path)

    step = 0
    input_length = 1
    t_accuracy = 0
    while True:
        if input_length == FLAGS.max_train_time_step and t_accuracy > 0.999:
            break

        batch_xs, batch_ys = task.next_batch_curr(FLAGS.batch_size, input_length, input_length)

        sess.run(model.optimize, {x: batch_xs, y: batch_ys})
        if step % 100 == 0:
            saver.save(sess, FLAGS.save_path, step)

            # cost_value, t_accuracy = 0, 0
            # test_runs = 10
            test_x, test_y = task.next_batch_for_test(FLAGS.batch_size,
                                                      input_length,
                                                      input_length)
            [summary, temp_predicts, cost_value, t_accuracy] = sess.run(
                [merged, model.prediction, model.cost, model.accuracy],
                feed_dict={x: test_x, y: test_y})
            test_writer.add_summary(summary, step)

            [summary, train_accuracy] = sess.run([merged, model.accuracy], feed_dict={x: batch_xs, y: batch_ys})
            train_writer.add_summary(summary, step)
            print("time: %s, step: %d, cost: %.5f, train: %.5f, test: %.5f" %
                  (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                   , step, cost_value, train_accuracy, t_accuracy))

            if t_accuracy > 0.9 and input_length != FLAGS.max_train_time_step:
                input_length += 1
                test_x, test_y = task.next_batch_for_test(FLAGS.batch_size, input_length, input_length)
                [temp_test] = sess.run([model.accuracy], feed_dict={x: test_x, y: test_y})
                print('---------' * 3)
                print("time: %s, input_length: %d, test: %.5f" %
                      (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                       , input_length, temp_test))
                print('---------' * 3)
        step += 1


def test():
    data_config = {
        "input_size": FLAGS.input_size,
        "min_step": 5,
        "max_step": 5,
        "real_max_step": FLAGS.max_steps,
        "batch_size": FLAGS.batch_size
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
                                 FLAGS.lr, FLAGS.clip_value)

    test_x, test_y = task.next_batch(FLAGS.batch_size)

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
