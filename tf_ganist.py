import numpy as np
import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0, 1" for multiple

np.random.seed(13)
tf.set_random_seed(13)

tf_dtype = tf.float32
np_dtype = 'float32'

### Operations
def lrelu(x, leak=0.2, name="lrelu"):
	with tf.variable_scope(name):
		f1 = 0.5 * (1 + leak)
		f2 = 0.5 * (1 - leak)
		return f1 * x + f2 * abs(x)

def conv2d(input_, output_dim,
           k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02,
           scope="conv2d", reuse=False, padding='SAME'):
    with tf.variable_scope(scope, reuse=reuse):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
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

### GAN Class definition
class Ganist:
	def __init__(self, log_dir='logs'):
		### run parameters
		self.log_dir = log_dir

		### optimization parameters
		self.g_lr = 2e-4
		self.g_beta1 = 0.5
		self.g_beta2 = 0.5
		self.d_lr = 2e-4
		self.d_beta1 = 0.5
		self.d_beta2 = 0.5

		### network parameters
		self.z_dim = [100] #256
		self.z_range = 1.0
		self.data_dim = [28, 28, 3]
		self.mm_loss_weight = 0.0
		self.gp_loss_weight = 5.0
		self.d_loss_type = 'was'
		self.g_loss_type = 'was'
		#self.d_act = tf.tanh
		#self.g_act = tf.tanh
		self.d_act = lrelu
		self.g_act = lrelu

		### init graph and session
		self.build_graph()
		self.start_session()

	def __del__(self):
		self.end_session()

	def build_graph(self):
		### define placeholders for image and label inputs
		self.im_input = tf.placeholder(tf_dtype, [None]+self.data_dim, name='im_input')
		self.z_input = tf.placeholder(tf_dtype, [None]+self.z_dim, name='z_input')
		self.e_input = tf.placeholder(tf_dtype, [None, 1, 1, 1], name='e_input')
		self.train_phase = tf.placeholder(tf.bool, name='phase')

		### build generator
		self.g_layer = self.build_gen(self.z_input, self.g_act, self.train_phase)

		### build discriminator
		self.r_logits = self.build_dis(self.im_input, self.d_act, self.train_phase)
		self.g_logits = self.build_dis(self.g_layer, self.d_act, self.train_phase, reuse=True)

		### build d losses
		if self.d_loss_type == 'log':
			self.d_r_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.r_logits, labels=tf.ones_like(self.r_logits, tf_dtype)), axis=None)
			self.d_g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.g_logits, labels=tf.zeros_like(self.g_logits, tf_dtype)), axis=None)
			self.d_loss = self.d_r_loss + self.d_g_loss
		elif self.d_loss_type == 'was':
			self.d_r_loss = -tf.reduce_mean(self.r_logits, axis=None)
			self.d_g_loss = tf.reduce_mean(self.g_logits, axis=None)
			rg_layer = (1.0 - self.e_input) * self.g_layer + self.e_input * self.im_input
			rg_logits = self.build_dis(rg_layer, self.d_act, self.train_phase, reuse=True)
			gp_loss = tf.reduce_mean(tf.square(tf.sqrt(tf.reduce_sum(tf.square(tf.gradients(rg_logits, rg_layer)), axis=1)) - 1.0), axis=None)
			self.d_loss = self.d_r_loss + self.d_g_loss + self.gp_loss_weight * gp_loss
		else:
			raise ValueError('>>> d_loss_type: %s is not defined!' % self.d_loss_type)

		### build g loss
		if self.g_loss_type == 'log':
			self.g_loss = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.g_logits, labels=tf.zeros_like(self.g_logits, tf_dtype)), axis=None)
		elif self.g_loss_type == 'mod':
			self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.g_logits, labels=tf.ones_like(self.g_logits, tf_dtype)), axis=None)
		elif self.g_loss_type == 'was':
			self.g_loss = -tf.reduce_mean(self.g_logits, axis=None)
		else:
			raise ValueError('>>> g_loss_type: %s is not defined!' % self.g_loss_type)

		### mean matching
		mm_loss = tf.reduce_mean(tf.square(tf.reduce_mean(self.g_layer, axis=0) - tf.reduce_mean(self.im_input, axis=0)), axis=None)
		self.g_loss = self.g_loss + self.mm_loss_weight * mm_loss

		### collect params
		self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "g_net")
		self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "d_net")

		### build optimizers
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			self.g_opt = tf.train.AdamOptimizer(self.g_lr, beta1=self.g_beta1, beta2=self.g_beta2).minimize(self.g_loss, var_list=self.g_vars)
			self.d_opt = tf.train.AdamOptimizer(self.d_lr, beta1=self.d_beta1, beta2=self.d_beta2).minimize(self.d_loss, var_list=self.d_vars)

		### summaries
		g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
		d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
		self.summary = tf.summary.merge([g_loss_sum, d_loss_sum])

	def build_gen(self, z, act, train_phase):
		with tf.variable_scope('g_net'):
			### fully connected from hidden z to image shape
			z_fc = act(dense(z, 4*4*64, scope='fcz'))
			h1 = tf.reshape(z_fc, [-1, 4, 4, 64])

			### decoding 4*4*256 code with upsampling and conv hidden layers into 32*32*3
			h1_us = tf.image.resize_nearest_neighbor(h1, [7, 7], name='us1')
			h2 = act(conv2d(h1_us, 32, scope='conv1'))

			h2_us = tf.image.resize_nearest_neighbor(h2, [14, 14], name='us2')
			h3 = act(conv2d(h2_us, 16, scope='conv2'))

			h3_us = tf.image.resize_nearest_neighbor(h3, [28, 28], name='us3')
			h4 = conv2d(h3_us, 3, scope='conv3')
			
			### output activation to bring data values in (-1,1)
			o = tf.nn.tanh(h4)

			return o

	def build_dis(self, data_layer, act, train_phase, reuse=False):
		with tf.variable_scope('d_net'):
			### encoding the 28*28*3 image with conv into 3*3*256
			h1 = act(conv2d(data_layer, 16, d_h=2, d_w=2, scope='conv1', reuse=reuse))
			h2 = act(conv2d(h1, 32, d_h=2, d_w=2, scope='conv2', reuse=reuse))
			#h3 = act(conv2d(h2, 64, d_h=2, d_w=2, scope='conv3', reuse=reuse))
			#h4 = conv2d(h2, 1, d_h=1, d_w=1, k_h=1, k_w=1, padding='VALID', scope='conv4', reuse=reuse)

			### fully connected discriminator
			#flat_h3 = tf.contrib.layers.flatten(h3)
			flat_h3 = tf.reshape(h2, [-1, 32*7*7])
			o = dense(flat_h3, 1, scope='fco', reuse=reuse)
			return o

	def start_session(self):
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
		config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
		self.saver = tf.train.Saver(tf.global_variables(), keep_checkpoint_every_n_hours=1, max_to_keep=10)
		self.sess = tf.Session(config=config)
		self.sess.run(tf.global_variables_initializer())
		self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

	def end_session(self):
		self.sess.close()

	#def clean(self):
#		self.sess.close()
		#tf.reset_default_graph()

	def save(self, fname):
		self.saver.save(self.sess, fname)

	def load(self, fname):
		self.saver.restore(self.sess, fname)

	def write_sum(self, sum_str, counter):
		self.writer.add_summary(sum_str, counter)

	def step(self, batch_data, batch_size, gen_update=False, dis_only=False, gen_only=False, z_data=None):
		batch_size = batch_data.shape[0] if batch_data is not None else batch_size		
		batch_data = batch_data.astype(np_dtype) if batch_data is not None else None

		### sample e from uniform (-1,1): for gp penalty in WGAN
		e_data = np.random.uniform(low=0.0, high=1.0, size=(batch_size, 1, 1, 1))
		e_data = e_data.astype(np_dtype)

		### only forward discriminator on batch_data
		if dis_only:
			feed_dict = {self.im_input: batch_data, self.e_input: e_data, self.train_phase: False}
			u_logits = self.sess.run(self.r_logits, feed_dict=feed_dict)
			return u_logits.flatten()

		### sample z from uniform (-1,1)
		if z_data is None:
			z_data = np.random.uniform(low=-self.z_range, high=self.z_range, size=[batch_size]+self.z_dim)
			#z_data = np.random.normal(loc=0.0, scale=1.0, size=(batch_size, self.z_dim))
		z_data = z_data.astype(np_dtype)

		### only forward generator on z
		if gen_only:
			feed_dict = {self.z_input: z_data, self.train_phase: False}
			g_layer = self.sess.run(self.g_layer, feed_dict=feed_dict)
			return g_layer

		### run one training step on discriminator, otherwise on generator, and log
		feed_dict = {self.im_input:batch_data, self.z_input: z_data, self.e_input: e_data, self.train_phase: True}
		if not gen_update:
			res_list = [self.g_layer, self.summary, self.d_opt]
			res_list = self.sess.run(res_list, feed_dict=feed_dict)
			#print '>>> r_logits shape:', res_list[3].shape
		else:
			res_list = [self.g_layer, self.summary, self.g_opt]
			res_list = self.sess.run(res_list, feed_dict=feed_dict)

		#print '>>> r_logits shape:', res_list[3].shape
		### return summary and g_layer
		return res_list[1], res_list[0]
