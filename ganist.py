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
	def __init__(self, data_dim):
		# s_data must have zero pads at the end to represent gen class, columns are features
		self.g_lr = 1e-4
		self.g_beta1 = 0.5
		self.g_beta2 = 0.5
		self.d_lr = 1e-4
		self.d_beta1 = 0.5
		self.d_beta2 = 0.5

		### network parameters
		self.z_dim = 1 #256
		self.z_range = 1.0
		self.data_dim = data_dim
		self.mm_loss_weight = 0.0
		self.gp_loss_weight = 10.0
		self.d_loss_type = 'log'
		self.g_loss_type = 'mod'
		self.d_act = tf.tanh
		self.g_act = tf.tanh
		#self.d_act = tf.nn.relu
		#self.g_act = lrelu

		### init graph and session
		self.build_graph()
		self.start_session()

	def __del__(self):
		self.end_session()

	def build_graph(self):
		### define placeholders for image and label inputs
		self.im_input = tf.placeholder(tf_dtype, [None, self.data_dim], name='im_input')
		self.z_input = tf.placeholder(tf_dtype, [None, self.z_dim], name='z_input')
		self.e_input = tf.placeholder(tf_dtype, [None, 1], name='e_input')
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

		### logs
		r_logits_mean = tf.reduce_mean(self.r_logits, axis=None)
		g_logits_mean = tf.reduce_mean(self.g_logits, axis=None)
		d_r_logits_diff = tf.sqrt(tf.reduce_mean(tf.square(tf.gradients(self.d_r_loss, self.r_logits)), axis=None))
		d_g_logits_diff = tf.sqrt(tf.reduce_mean(tf.square(tf.gradients(self.d_g_loss, self.g_logits)), axis=None))
		g_logits_diff = tf.sqrt(tf.reduce_mean(tf.square(tf.gradients(self.g_loss, self.g_logits)), axis=None))
		g_out_diff = tf.sqrt(tf.reduce_mean(tf.square(tf.gradients(self.g_loss, self.g_layer)), axis=None))
		
		diff = tf.zeros((1,), tf_dtype)
		for v in self.d_vars:
			if 'bn_' in v.name:
				continue
			diff = diff + tf.sqrt(tf.reduce_mean(tf.square(tf.gradients(self.d_r_loss, v)), axis=None))
		d_r_param_diff = 1.0 * diff / len(self.d_vars)

		diff = tf.zeros((1,), tf_dtype)
		for v in self.d_vars:
			if 'bn_' in v.name:
				continue
			diff = diff + tf.sqrt(tf.reduce_mean(tf.square(tf.gradients(self.d_g_loss, v)), axis=None))
		d_g_param_diff = 1.0 * diff / len(self.d_vars)

		diff = tf.zeros((1,), tf_dtype)
		for v in self.d_vars:
			if 'bn_' in v.name:
				continue
			diff = diff + tf.sqrt(tf.reduce_mean(tf.square(tf.gradients(self.d_loss, v)), axis=None))
		d_param_diff = 1.0 * diff / len(self.d_vars)

		diff = tf.zeros((1,), tf_dtype)
		for v in self.g_vars:
			if 'bn_' in v.name:
				continue
			diff = diff + tf.sqrt(tf.reduce_mean(tf.square(tf.gradients(self.g_loss, v)), axis=None))
		g_param_diff = 1.0 * diff / len(self.g_vars)

		### build optimizers
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			self.g_opt = tf.train.AdamOptimizer(self.g_lr, beta1=self.g_beta1, beta2=self.g_beta2).minimize(self.g_loss, var_list=self.g_vars)
			self.d_opt = tf.train.AdamOptimizer(self.d_lr, beta1=self.d_beta1, beta2=self.d_beta2).minimize(self.d_loss, var_list=self.d_vars)

		### summaries
		self.d_r_logs = [self.d_loss, d_param_diff, self.d_r_loss, r_logits_mean, d_r_logits_diff, d_r_param_diff]
		self.d_g_logs = [self.d_g_loss, g_logits_mean, d_g_logits_diff, d_g_param_diff]
		self.g_logs = [self.g_loss, g_logits_diff, g_out_diff, g_param_diff]
		

	def build_gen(self, z, act, train_phase):
		h1_size = 128
		h2_size = 128
		h3_size = 128
		h4_size = 128
		with tf.variable_scope('g_net'):
			#h1 = linear(z, h1_size, scope='fc1')
			h1 = dense(z, h1_size, scope='fc1')
			h1 = act(h1)

			h2 = dense(h1, h2_size, scope='fc2')
			h2 = act(h2)

			#h3 = dense(h2, h3_size, scope='fc3')
			#h3 = act(h3)

			#h4 = dense(h3, h4_size, scope='fc4')
			#h4 = act(h4)

			o = dense(h2, self.data_dim, scope='fco')
			return o

	def build_dis(self, data_layer, act, train_phase, reuse=False):
		h1_size = 128
		h2_size = 128
		h3_size = 128
		h4_size = 128
		with tf.variable_scope('d_net'):
			h1 = dense(data_layer, h1_size, scope='fc1', reuse=reuse)
			h1 = act(h1)

			#h2 = dense_batch(h1, h2_size, scope='fc2', reuse=reuse, phase=train_phase)
			h2 = dense(h1, h2_size, scope='fc2', reuse=reuse)
			h2 = act(h2)
			
			#h3 = dense(h2, h3_size, scope='fc3', reuse=reuse)
			#h3 = act(h3)

			#h4 = dense(h3, h4_size, scope='fc4', reuse=reuse)
			#h4 = act(h4)

			o = dense(h2, 1, scope='fco', reuse=reuse)
			return o

	def start_session(self):
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
		config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
		self.saver = tf.train.Saver(tf.global_variables(), keep_checkpoint_every_n_hours=1)
		self.sess = tf.Session(config=config)
		self.sess.run(tf.global_variables_initializer())

	def end_session(self):
		self.sess.close()

	def save(self, fname):
		self.saver.save(self.sess, fname)

	def load(self, fname):
		self.saver.restore(self.sess, fname)

	def step(self, batch_data, batch_size, gen_update=False, dis_only=False, gen_only=False, z_data=None):
		batch_data = batch_data.astype(np_dtype) if batch_data is not None else None
		
		### sample e from uniform (-1,1): for gp penalty in WGAN
		e_data = np.random.uniform(low=0.0, high=1.0, size=(batch_size, 1))
		e_data = e_data.astype(np_dtype)

		### only forward discriminator on batch_data
		if dis_only:
			feed_dict = {self.im_input: batch_data, self.e_input: e_data, self.train_phase: False}
			u_logits = self.sess.run(self.r_logits, feed_dict=feed_dict)
			return u_logits.flatten()

		### sample z from uniform (-1,1)
		if z_data is None:
			z_data = np.random.uniform(low=-self.z_range, high=self.z_range, size=(batch_size, self.z_dim))
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
			res_list = [self.g_layer, self.g_logs, self.d_r_logs, self.d_g_logs, self.d_opt]
			res_list = self.sess.run(res_list, feed_dict=feed_dict)
		else:
			res_list = [self.g_layer, self.g_logs, self.d_r_logs, self.d_g_logs, self.g_opt]
			res_list = self.sess.run(res_list, feed_dict=feed_dict)

		return tuple(res_list[1:]), res_list[0]
