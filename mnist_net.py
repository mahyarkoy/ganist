import numpy as np
import tensorflow as tf
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
#os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0, 1" for multiple

#np.random.seed(13)
#tf.set_random_seed(13)

tf_dtype = tf.float32
np_dtype = 'float32'

### Operations
def lrelu(x, leak=0.2, name="lrelu"):
	with tf.variable_scope(name):
		f1 = 0.5 * (1 + leak)
		f2 = 0.5 * (1 - leak)
		return f1 * x + f2 * abs(x)

def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=1, d_w=1, stddev=0.02,
           scope="conv2d", reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        conv = tf.nn.bias_add(conv, biases)

        return conv

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
	shape = input_.get_shape().as_list()
	with tf.variable_scope(scope or "Linear"):
		matrix = tf.get_variable("Matrix", [shape[1], output_size], tf_dtype,
								 tf.contrib.layers.xavier_initializer())
		bias = tf.get_variable("bias", [output_size], tf_dtype,
			initializer=tf.constant_initializer(bias_start))
		if with_w:
			return tf.matmul(input_, matrix) + bias, matrix, bias
		else:
			return tf.matmul(input_, matrix) + bias

def dense_batch(x, h_size, scope, phase, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        h1 = tf.contrib.layers.fully_connected(x, h_size, activation_fn=None, scope='dense')
    with tf.variable_scope(scope):
        h2 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, is_training=phase, scope='bn_'+str(reuse))
    return h2

def dense(x, h_size, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        h1 = tf.contrib.layers.fully_connected(x, h_size, activation_fn=None, scope='dense')
        #h1 = tf.contrib.layers.fully_connected(x, h_size, activation_fn=None, scope='dense', weights_initializer=tf.truncated_normal_initializer(stddev=0.02))
        #h1 = tf.contrib.layers.fully_connected(x, h_size, activation_fn=None, scope='dense', weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True))
    return h1

### Mnist Classifier Class definition
class MnistNet:
	def __init__(self, sess, log_dir='logs'):
		### run parameters
		self.log_dir = log_dir
		self.sess = sess

		### optimization parameters
		self.lr = 1e-4
		self.beta1 = 0.5
		self.beta2 = 0.5

		### network parameters
		self.data_dim = [28, 28, 1]
		self.num_class = 10
		self.c_act = lrelu

		### init graph and session
		self.build_graph()
		self.start_session()

	def build_graph(self):
		with tf.name_scope('mnist_net'):
			### define placeholders for image and label inputs
			self.im_input = tf.placeholder(tf_dtype, [None]+self.data_dim, name='im_input')
			self.labels = tf.placeholder(tf_dtype, [None, self.num_class], name='labs_input')
			self.train_phase = tf.placeholder(tf.bool, name='phase')

			### build classifier
			self.logits = self.build_classifier(self.im_input, self.c_act, self.train_phase)

			### build losses, preds and acc
			self.c_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels), axis=None)
			self.preds = tf.nn.softmax(self.logits)
			self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.preds, axis=1), tf.argmax(self.labels, axis=1)), tf_dtype))

			### collect params
			self.c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "c_net")

			### build optimizers
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				self.c_opt = tf.train.AdamOptimizer(self.lr, beta1=self.beta1, beta2=self.beta2).minimize(self.c_loss, var_list=self.c_vars)

			### summaries
			train_loss_sum = tf.summary.scalar("train_loss", self.c_loss)
			train_acc = tf.summary.scalar("train_acc", self.acc)
			eval_loss_sum = tf.summary.scalar("eval_loss", self.c_loss)
			eval_acc = tf.summary.scalar("eval_acc", self.acc)
			self.train_summary = tf.summary.merge([train_loss_sum, train_acc])
			self.eval_summary = tf.summary.merge([eval_loss_sum, eval_acc])

	def build_classifier(self, data_layer, act, train_phase, reuse=False):
		with tf.variable_scope('c_net'):
			### encoding the 28*28*1 image with conv into 4*4*256
			h1 = act(conv2d(data_layer, 64, d_h=2, d_w=2, scope='conv1', reuse=reuse))
			h2 = act(conv2d(h1, 128, d_h=2, d_w=2, scope='conv2', reuse=reuse))
			h3 = act(conv2d(h2, 256, d_h=2, d_w=2, scope='conv3', reuse=reuse))

			### fully connected classifier
			flat_h3 = tf.contrib.layers.flatten(h3)
			o = dense(flat_h3, self.num_class, scope='fco', reuse=reuse)
			return o

	def start_session(self):
		self.saver = tf.train.Saver(self.c_vars, keep_checkpoint_every_n_hours=1, max_to_keep=10)
		self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

	def save(self, fname):
		self.saver.save(self.sess, fname)

	def load(self, fname):
		self.saver.restore(self.sess, fname)

	def write_sum(self, sum_str, counter):
		self.writer.add_summary(sum_str, counter)

	def step(self, batch_data, labels=None, train=True, pred_only=False):
		batch_size = batch_data.shape[0]
		batch_data = batch_data.astype(np_dtype)

		### only forward classifier on batch_data and return preds
		if pred_only:
			feed_dict = {self.im_input: batch_data, self.train_phase: False}
			preds = self.sess.run(self.preds, feed_dict=feed_dict)
			return preds

		### labels setup to matrix
		mat_labels = 0.1 * np.ones((batch_size, self.num_class)) / (self.num_class-1)
		mat_labels[np.arange(batch_size), labels] = 0.9

		### only forward classifier on batch_data and return preds, acc and loss
		if not train:
			feed_dict = {self.im_input: batch_data, self.labels: mat_labels, self.train_phase: False}
			res_list = [self.preds, self.acc, self.c_loss]
			res_list = self.sess.run(res_list, feed_dict=feed_dict)
			return res_list[0], res_list[1], res_list[2]

		### run one training step on classifier and return preds, acc and sum
		feed_dict = {self.im_input:batch_data, self.labels: mat_labels, self.train_phase: True}
		res_list = [self.preds, self.acc, self.train_summary, self.c_opt]
		res_list = self.sess.run(res_list, feed_dict=feed_dict)

		### return preds, acc and sum
		return res_list[0], res_list[1], res_list[2]
