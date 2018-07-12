import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from opt_meta_function import opt_meta_function

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


def init_weights(shape):
	weights = tf.random_normal(shape, stddev=0.1)
	return(tf.Variable(weights))


def forwardprop(X, w_1, w_2, w_3):
	h = tf.nn.sigmoid(tf.matmul(X, w_1))
	h2 = tf.nn.sigmoid(tf.matmul(h, w_2))
	yhat = tf.matmul(h2, w_3)
	return(yhat)

def get_data(filepath):
	data = pd.read_csv(filepath)

	X = data.iloc[:,1:].values
	Y = data.iloc[:,0].values

	N, M = data.shape
	all_X = np.ones((N, M))
	all_X[:,1:] = X

	unique = np.unique(Y)
	num_labels = len(unique)

	if Y.dtype not in [int, float]: 
		Y_dic = dict(zip(unique, list(range(len(unique)))))
		Y = [Y_dic[x] for x in Y]

	all_Y = np.eye(num_labels)[Y]

	return(train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED))


def train_save_neural_network(filepath):
	train_X, test_X, train_y, test_y = get_data(filepath)

	x_size = train_X.shape[1]
	h_size = 256
	h2_size = 128
	y_size = train_y.shape[1]

	X = tf.placeholder("float", shape=[None, x_size])
	Y = tf.placeholder("float", shape=[None, y_size])

	w_1 = init_weights((x_size, h_size))
	w_2 = init_weights((h_size, h2_size))
	w_3 = init_weights((h2_size, y_size))

	yhat = forwardprop(X, w_1, w_2, w_3)
	predict = tf.argmax(yhat, axis=1)

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=yhat))
	update = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

	saver = tf.train.Saver()

	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)

	for epoch in range(100):
		for i in range(len(train_X)):
			sess.run(update, feed_dict={X: train_X[i:i+1], Y: train_y[i:i+1]})

		train_accuracy = np.mean(np.argmax(train_y, axis=1) == sess.run(predict, feed_dict={X: train_X, Y: train_y}))

		test_accuracy = np.mean(np.argmax(test_y, axis=1) == sess.run(predict, feed_dict={X: test_X, Y: test_y}))

		print("Epoch = %d, train accuracy = %.2f%%, test accuracy %.2f%%" % (epoch+1, 100.*train_accuracy, 100.*test_accuracy))

	save_path = saver.save(sess, filepath + "model.ckpt")
	print("model saved in path: %s" % save_path)

	sess.close()

if __name__ == '__main__':
	opt_meta_function(train_save_neural_network)