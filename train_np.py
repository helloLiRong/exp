from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time

import dnc.dnc as dnc
from task.copy_task_np import CopyTask
import task.copy_task_np as copy_task

FLAGS = tf.flags.FLAGS

# Model parameters
tf.flags.DEFINE_integer("hidden_size", 128, "Size of LSTM hidden layer.")
tf.flags.DEFINE_integer("memory_size", 128, "The number of memory slots.")
tf.flags.DEFINE_integer("word_size", 4, "The width of each memory slot.")
tf.flags.DEFINE_integer("num_write_heads", 1, "Number of memory write heads.")
tf.flags.DEFINE_integer("num_read_heads", 1, "Number of memory read heads.")
tf.flags.DEFINE_integer("clip_value", 20,
                        "Maximum absolute value of controller and dnc outputs.")

# Optimizer parameters.
tf.flags.DEFINE_float("max_grad_norm", 50, "Gradient clipping norm limit.")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Optimizer learning rate.")
tf.flags.DEFINE_float("optimizer_epsilon", 1e-10,
                      "Epsilon used for RMSProp optimizer.")

# Task parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch size for training.")
tf.flags.DEFINE_integer("num_bits", 8, "Dimensionality of each vector to copy")
tf.flags.DEFINE_integer(
    "min_length", 1,
    "Lower limit on number of vectors in the observation pattern to copy")
tf.flags.DEFINE_integer(
    "max_length", 5,
    "Upper limit on number of vectors in the observation pattern to copy")

# Training options.
tf.flags.DEFINE_integer("num_training_iterations", 100000,
                        "Number of iterations to train for.")
tf.flags.DEFINE_integer("report_interval", 100,
                        "Iterations between reports (samples, valid loss).")
tf.flags.DEFINE_string("checkpoint_dir", "exp_result/dnc",
                       "Checkpointing directory.")
tf.flags.DEFINE_integer("checkpoint_interval", 1000,
                        "Checkpointing step interval.")
tf.flags.DEFINE_bool("is_training", True, "is training")
tf.flags.DEFINE_float("curriculum_learning_epsilon", 0.9, "epsilon of curriculum learning")
tf.flags.DEFINE_string("CLS", "Combining", "epsilon of curriculum learning")
tf.flags.DEFINE_integer("dataset_size", 1000, "size of dataset")
tf.flags.DEFINE_integer("current_length", 2, "current_length")


def run_model(input_sequence, output_size):
    """Runs model on input sequence."""

    access_config = {
        "memory_size": FLAGS.memory_size,
        "word_size": FLAGS.word_size,
        "num_reads": FLAGS.num_read_heads,
        "num_writes": FLAGS.num_write_heads,
    }
    controller_config = {
        "hidden_size": FLAGS.hidden_size,
    }
    clip_value = FLAGS.clip_value

    dnc_core = dnc.DNC(access_config, controller_config, output_size, clip_value)
    initial_state = dnc_core.initial_state(FLAGS.batch_size)
    output_sequence, _ = tf.nn.dynamic_rnn(
        cell=dnc_core,
        inputs=input_sequence,
        time_major=True,
        initial_state=initial_state)

    return output_sequence


def train(num_training_iterations, report_interval):
    """Trains the DNC and periodically reports the loss."""

    dataset = CopyTask(FLAGS.num_bits, FLAGS.batch_size,
                       FLAGS.min_length, FLAGS.max_length,
                       curriculum_learning_strategy=FLAGS.CLS)

    x = tf.placeholder(tf.float32, [None, None, dataset.target_size])
    y = tf.placeholder(tf.float32, [None, None, dataset.target_size])
    m = tf.placeholder(tf.float32)
    output_logits = run_model(x, dataset.target_size)
    # Used for visualization.
    output = tf.round(
        tf.expand_dims(m, -1) * tf.sigmoid(output_logits))
    train_accuracy = dataset.accuracy(output, y)

    train_loss = dataset.cost(output_logits, y,
                              m)

    # Set up optimizer with global norm clipping.
    trainable_variables = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(
        tf.gradients(train_loss, trainable_variables), FLAGS.max_grad_norm)

    global_step = tf.get_variable(
        name="global_step",
        shape=[],
        dtype=tf.int64,
        initializer=tf.zeros_initializer(),
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

    optimizer = tf.train.RMSPropOptimizer(
        FLAGS.learning_rate, epsilon=FLAGS.optimizer_epsilon)
    train_step = optimizer.apply_gradients(
        zip(grads, trainable_variables), global_step=global_step)

    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.checkpoint_dir + '/train')
    test_writer = tf.summary.FileWriter(FLAGS.checkpoint_dir + '/test')

    if FLAGS.checkpoint_interval > 0:
        hooks = [
            tf.train.CheckpointSaverHook(
                checkpoint_dir=FLAGS.checkpoint_dir,
                save_steps=FLAGS.checkpoint_interval,
                saver=saver)
        ]
    else:
        hooks = []

    # Train.
    with tf.train.SingularMonitoredSession(
            hooks=hooks, checkpoint_dir=FLAGS.checkpoint_dir) as sess:
        start_iteration = sess.run(global_step)
        total_loss = 0
        current_length = FLAGS.current_length
        need_create_data = True
        test_train_data = dataset.create_data(batch_size=FLAGS.batch_size, min_length=FLAGS.min_length,
                                              max_length=FLAGS.max_length)

        for train_iteration in range(start_iteration, num_training_iterations):

            if train_iteration % (1 * FLAGS.dataset_size) == 0 \
                    or need_create_data:
                all_train_data = dataset(
                    batch_size=FLAGS.batch_size * FLAGS.dataset_size * 1, max_length=current_length)
                start_idx = 0
                end_idx = start_idx + FLAGS.batch_size
                need_create_data = False

            if train_iteration % report_interval == 1:
                summary, train_accuracy_np = sess.run(
                    [merged, train_accuracy], feed_dict={
                        x: observations,
                        y: target,
                        m: mask
                    })

                train_writer.add_summary(summary, train_iteration)

                summary, output_np, test_accuracy_np = sess.run(
                    [merged, output, train_accuracy], feed_dict={
                        x: test_train_data.observations,
                        y: test_train_data.target,
                        m: test_train_data.mask
                    })

                test_writer.add_summary(summary, train_iteration)

                dataset_string = dataset.to_human_readable(test_train_data,
                                                           output_np)
                tf.logging.info("%s, "
                                "step %d, length: %d: Avg training loss %f. training_accuracy %f. "
                                "test_accuracy %f.\n%s\n",
                                (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                                train_iteration,
                                current_length,
                                total_loss / report_interval,
                                train_accuracy_np,
                                test_accuracy_np,
                                dataset_string)
                if current_length != FLAGS.max_length and train_accuracy_np > FLAGS.curriculum_learning_epsilon:
                    tf.logging.info("length: %d curriculum completed\n" % current_length)
                    current_length += 1
                    need_create_data = True
                total_loss = 0

            observations = all_train_data.observations[:, start_idx:end_idx, :]
            target = all_train_data.target[:, start_idx:end_idx, :]
            mask = all_train_data.mask[:, start_idx:end_idx]

            _, loss = sess.run([train_step, train_loss],
                               feed_dict={x: observations, y: target, m: mask})
            total_loss += loss
            start_idx = end_idx
            end_idx = start_idx + FLAGS.batch_size


def test():
    dataset = CopyTask(FLAGS.num_bits, FLAGS.batch_size,
                       FLAGS.min_length, FLAGS.max_length)
    dataset_tensors = dataset()

    output_logits = run_model(dataset_tensors.observations, dataset.target_size)
    # Used for visualization.
    output = tf.round(
        tf.expand_dims(dataset_tensors.mask, -1) * tf.sigmoid(output_logits))
    train_accuracy = dataset.accuracy(output, dataset_tensors.target)
    saver = tf.train.Saver()
    # test.
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        dataset_tensors_np, output_np, train_accuracy = sess.run([dataset_tensors, output, train_accuracy])
        dataset_string = dataset.to_human_readable(dataset_tensors_np,
                                                   output_np, True)
        tf.logging.info("%s\ntraining_accuracy %f.\n",
                        dataset_string, train_accuracy)


def main(unused_argv):
    tf.logging.set_verbosity(3)  # Print INFO log messages.
    if FLAGS.is_training:
        train(FLAGS.num_training_iterations, FLAGS.report_interval)
    else:
        test()


if __name__ == "__main__":
    tf.app.run()
