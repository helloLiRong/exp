import tensorflow as tf
import numpy as np
from task import new_copy_task as copy_task
from task.copy_task_np import CopyTask
from enum import Enum

# 将sess封装成具有tfdbg功能的Session
# dataset = repeat_copy.RepeatCopy()
# dataset = copy_task.CopyTask(batch_size=2)
#
# dataset_tensors = dataset()
# sess = tf.Session()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# init = tf.glorot_normal_initializer()
# sess.run(init)
# a = sess.run(dataset_tensors)
# b = sess.run(dataset.target_size)

# print(dataset.to_human_readable(a))
# new_mask = tf.expand_dims(a.mask, -1)
# obs = new_mask * tf.random_uniform([4, 1, 7], minval=0, maxval=2)

# print(sess.run(tf.equal(obs, a.target)))
# print(sess.run(tf.reduce_all(tf.equal(obs, a.target), axis=2)))
# print(sess.run(tf.reduce_all(tf.reduce_all(tf.equal(obs, a.target), axis=2), axis=0)))
# print(sess.run(tf.argmax(obs, axis=2)))
# print(sess.run(dataset.accuracy(obs, a.target)))
# print(a.observations)
# print(np.concatenate([a.observations[:, :, :-1], a.observations[:, :, -1:] * 10], axis=2))

CLS = Enum("curriculum_strategy", ("No", "Naive", "Combining"))

print(CLS.No.name == "No")
