import tensorflow as tf
import numpy as np
from task import new_copy_task as copy_task
from task.copy_task_np import CopyTask

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
dataset = CopyTask()
v = dataset(1, 2, 128)

observations = v.observations[:, 1:2, :]
target = v.target[:, 1:2, :]
mask = v.mask[:, :1]
print(dataset.to_human_readable(v, v.target))
print(mask)
