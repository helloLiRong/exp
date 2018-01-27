import repeat_copy
import tensorflow as tf

dataset = repeat_copy.RepeatCopy(8, 1,
                                 1, 1,
                                 1, 1)

dataset_tensors = dataset()
sess = tf.Session()
# init = tf.glorot_normal_initializer()
# sess.run(init)
a = sess.run(dataset_tensors)
# b = sess.run(dataset.target_size)

print(a.observations)
print(a.target)
print(a.mask)
