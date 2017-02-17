import tensorflow as tf
import random
import math
from load import get_input

# parameters
training_epochs = 10
learning_rate = 0.01
batch_size = 100
display_step = 10

n_input = 1500
n_classes = 5
dropout = 0.75


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def conv_net(x, weights, biases, dropout):
    # input
    x = tf.reshape(x, shape=[-1, 50, 30, 1])

    # Convolution Layer 1
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=5)

    # Convolution Layer 2
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    fc1 = tf.nn.dropout(fc1, dropout)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([5 * 3 * 64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# input
train_x, train_y = get_input('./data/eng.train')
test_x, test_y = get_input('./data/eng.testa')

pred = conv_net(x, weights, biases, dropout)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

if __name__ == "__main__":
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = math.ceil((len(train_x) / batch_size))

            batch_list = [index for index in range(total_batch)]
            random.shuffle(batch_list)
            for batch_i in batch_list:
                _, c = sess.run([optimizer, cost],
                                feed_dict={x: train_x[batch_i * batch_size:(batch_i + 1) * batch_size],
                                           y: train_y[batch_i * batch_size:(batch_i + 1) * batch_size],
                                           keep_prob: dropout})
                avg_cost += c / total_batch

            print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                  "{:.9f}".format(avg_cost))

        print(accuracy.eval({x: test_x, y: test_y}))