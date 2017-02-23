import tensorflow as tf

# g = tf.Graph()
#
# with g.as_default():
#     v1 = tf.placeholder('float', (1,2), name='v1')
#
#     print(v1.graph is g)
#
#
# with g.as_default():
#     v1 = tf.constant(30.0, name='v2')
#     print(v1)
#     print(v1.graph is g)
#
#     init = tf.global_variables_initializer()
#
# sess = tf.Session(graph=g)
# sess.run(init)
#
# for op in g.get_operations():
#     print(op.name)


for i in range(1, 10):
    print(i)
