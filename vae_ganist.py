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
           scope="conv2d", reuse=False, padding='SAME'):
    with tf.variable_scope(scope, reuse=reuse):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        #w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
        #                    initializer=tf.truncated_normal_initializer(stddev=stddev))
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

### VAE GAN Class definition
class VAEGanist:
	def __init__(self, sess, log_dir='logs'):
		### run parameters
		self.log_dir = log_dir
		self.sess = sess

		### optimization parameters
		self.g_lr = 2e-4
		self.g_beta1 = 0.5
		self.g_beta2 = 0.5
		self.d_lr = 2e-4
		self.d_beta1 = 0.5
		self.d_beta2 = 0.5

		### network parameters **g_num** **mt**
		### >>> dataset sensitive: data_dim
		self.z_dim = 16 #256
		self.z_range = 1.0
		self.data_dim = [28, 28, 1]
		self.mm_loss_weight = 0.0
		self.gp_loss_weight = 10.0
		self.d_loss_type = 'was'
		self.g_loss_type = 'was'
		self.encoder_act = lrelu
		self.decoder_act = lrelu
		self.transform_act = lrelu
		self.d_act = lrelu

		### init graph and session
		self.build_graph()
		self.start_session()

	def build_graph(self):
		with tf.name_scope('ganist'):
			### define placeholders for image and label inputs
			self.im_input = tf.placeholder(tf_dtype, [None]+self.data_dim, name='im_input')
			self.z_input = tf.placeholder(tf_dtype, [None, self.z_dim], name='z_input')
			self.e_input = tf.placeholder(tf_dtype, [None, 1, 1, 1], name='e_input')
			self.train_phase = tf.placeholder(tf.bool, name='phase')

			### build encoder and compute z_codes
			z_mean, z_std = self.build_encoder(self.im_input, self.encoder_act, self.train_phase)
			z_samples = tf.random_normal([tf.shape(z_mean)[0], self.z_dim], 0., 1., dtype=tf_dtype)
			self.z_codes = z_mean + z_std * z_samples

			### build decoder
			self.im_decode = self.build_decoder(self.z_codes, self.decoder_act, self.train_phase)
			self.z_decode = self.build_decoder(self.z_input, self.decoder_act, self.train_phase, reuse=True)

			### build vae loss
			self.rec_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.im_decode - self.im_input), [1, 2, 3]))
			self.kl_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_std) - tf.log(1e-8+tf.square(z_std)) ,1))
			self.vae_loss = self.rec_loss + self.kl_loss

			### build transform to get g_layer
			self.g_layer = self.build_tranform(self.z_decode, self.transform_act, self.train_phase)

			### build discriminator
			self.r_logits = self.build_dis(self.im_input, self.d_act, self.train_phase)
			self.g_logits = self.build_dis(self.g_layer, self.d_act, self.train_phase, reuse=True)

			### real gen manifold interpolation
			rg_layer = (1.0 - self.e_input) * self.g_layer + self.e_input * self.im_input
			self.rg_logits = self.build_dis(rg_layer, self.d_act, self.train_phase, reuse=True)

			### build d losses
			if self.d_loss_type == 'log':
				self.d_r_loss = tf.reduce_mean(
					tf.nn.sigmoid_cross_entropy_with_logits(
						logits=self.r_logits, labels=tf.ones_like(self.r_logits, tf_dtype)), axis=None)
				self.d_g_loss = tf.reduce_mean(
					tf.nn.sigmoid_cross_entropy_with_logits(
						logits=self.g_logits, labels=tf.zeros_like(self.g_logits, tf_dtype)), axis=None)
				self.d_rg_loss = tf.reduce_mean(
					tf.nn.sigmoid_cross_entropy_with_logits(
						logits=self.rg_logits, labels=tf.ones_like(self.rg_logits, tf_dtype)), axis=None)
			elif self.d_loss_type == 'was':
				self.d_r_loss = -tf.reduce_mean(self.r_logits, axis=None)
				self.d_g_loss = tf.reduce_mean(self.g_logits, axis=None)
			else:
				raise ValueError('>>> d_loss_type: %s is not defined!' % self.d_loss_type)

			### gradient penalty
			### NaN free norm gradient
			rg_grad = tf.gradients(self.rg_logits, rg_layer)
			rg_grad_flat = tf.reshape(rg_grad, [-1, np.prod(self.data_dim)])
			rg_grad_ok = tf.reduce_sum(tf.square(rg_grad_flat), axis=1) > 0.
			rg_grad_safe = tf.where(rg_grad_ok, rg_grad_flat, tf.ones_like(rg_grad_flat))
			rg_grad_abs = tf.where(rg_grad_flat >= 0., rg_grad_flat, -rg_grad_flat)
			rg_grad_norm = tf.where(rg_grad_ok, 
				tf.norm(rg_grad_safe, axis=1), tf.reduce_sum(rg_grad_abs, axis=1))
			gp_loss = tf.reduce_mean(tf.square(rg_grad_norm - 1.0))
			
			### d loss combination
			self.d_loss = self.d_r_loss + self.d_g_loss + self.gp_loss_weight * gp_loss

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

			### g loss combination
			self.g_loss = self.g_loss + self.mm_loss_weight * mm_loss

			### collect params
			self.encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "encoder_net")
			self.decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "decoder_net")
			self.trans_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "transform_net")
			self.disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "disc_net")

			### compute stat of weights
			self.nan_vars = 0.
			self.inf_vars = 0.
			self.zero_vars = 0.
			self.big_vars = 0.
			self.count_vars = 0
			for v in self.encoder_vars + self.decoder_vars + self.trans_vars + self.disc_vars:
				self.nan_vars += tf.reduce_sum(tf.cast(tf.is_nan(v), tf_dtype))
				self.inf_vars += tf.reduce_sum(tf.cast(tf.is_inf(v), tf_dtype))
				self.zero_vars += tf.reduce_sum(tf.cast(tf.square(v) < 1e-6, tf_dtype))
				self.big_vars += tf.reduce_sum(tf.cast(tf.square(v) > 1., tf_dtype))
				self.count_vars += tf.reduce_prod(v.get_shape())
			self.count_vars = tf.cast(self.count_vars, tf_dtype)
			self.nan_vars /= self.count_vars 
			self.inf_vars /= self.count_vars
			self.zero_vars /= self.count_vars
			self.big_vars /= self.count_vars

			### build optimizers
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			print '>>> update_ops list: ', update_ops
			with tf.control_dependencies(update_ops):
				self.vae_opt = tf.train.AdamOptimizer(2e-4, beta1=0.9, beta2=0.999).minimize(self.vae_loss, var_list=self.encoder_vars+self.decoder_vars)
				self.g_opt = tf.train.AdamOptimizer(2e-4, beta1=0.5, beta2=0.5).minimize(self.g_loss, var_list=self.trans_vars)
				self.d_opt = tf.train.AdamOptimizer(2e-4, beta1=0.5, beta2=0.5).minimize(self.d_loss, var_list=self.disc_vars)

			### summaries
			self.vae_loss_sum = tf.summary.scalar("g_loss", self.vae_loss)
			g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
			d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
			self.summary = tf.summary.merge([g_loss_sum, d_loss_sum])

	def build_encoder(self, im_data, act, train_phase):
		with tf.variable_scope('encoder_net'):
			bn = tf.contrib.layers.batch_norm
			### encoding 28*28 image to 7*7 image
			h1 = act(conv2d(im_data, 32, d_h=2, d_w=2, scope='conv1'))
			h2 = act(conv2d(h1, 32, d_h=2, d_w=2, scope='conv2'))
			### flatten and dense to get hidden mean and std
			flat_h2 = tf.contrib.layers.flatten(h2)
			z_mean = dense(flat_h2, self.z_dim, scope='z_mean')
			z_std = tf.sigmoid(dense(flat_h2, self.z_dim, scope='z_std'))
			return z_mean, z_std
	
	def build_decoder(self, z, act, train_phase, reuse=False):
		with tf.variable_scope('decoder_net'):
			zi = z
			bn = tf.contrib.layers.batch_norm
	
			### fully connected from hidden z to image shape
			z_fc = act(dense(zi, 7*7*32, scope='fcz', reuse=reuse))
			h1 = tf.reshape(z_fc, [-1, 7, 7, 32])

			### decoding 7*7*32 code with upsampling and conv hidden layers into 28*28
			h1_us = tf.image.resize_nearest_neighbor(h1, [14, 14], name='us1')
			h2 = act(conv2d(h1_us, 32, scope='conv1', reuse=reuse))

			h2_us = tf.image.resize_nearest_neighbor(h2, [28, 28], name='us2')
			h3 = conv2d(h2_us, self.data_dim[-1], scope='conv2', reuse=reuse)
			
			### output activation to bring data values in (-1,1)
			o = tf.tanh(h3)
			return o

	def build_tranform(self, im_data, act, train_phase):
		with tf.variable_scope('transform_net'):
			h1 = act(conv2d(im_data, 32, scope='conv1'))
			h2 = conv2d(h1, self.data_dim[-1], scope='conv2')
			o = tf.tanh(h2)
			return o

	def build_dis(self, im_data, act, train_phase, reuse=False):
		with tf.variable_scope('disc_net'):
			### encoding 28*28 image to 7*7 image
			h1 = act(conv2d(im_data, 32, d_h=2, d_w=2, scope='conv1', reuse=reuse))
			h2 = act(conv2d(h1, 32, d_h=2, d_w=2, scope='conv2', reuse=reuse))
			### flatten and dense to get logits
			flat_h2 = tf.contrib.layers.flatten(h2)
			o = dense(flat_h2, 1, scope='fco', reuse=reuse)
			return o

	def start_session(self):
		self.saver = tf.train.Saver(self.encoder_vars+self.decoder_vars+self.trans_vars+self.disc_vars, 
			keep_checkpoint_every_n_hours=5, max_to_keep=5)
		self.saver_vae = tf.train.Saver(self.encoder_vars+self.decoder_vars)
		self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

	def save(self, fname):
		self.saver.save(self.sess, fname)

	def load(self, fname):
		self.saver.restore(self.sess, fname)

	def load_vae(self, fname):
		self.saver_vae.restore(self.sess, fname)

	def write_sum(self, sum_str, counter):
		self.writer.add_summary(sum_str, counter)

	def step_vae(self, batch_data, batch_size, update=False, gen_only=False, stats_only=False, z_data=None):
		batch_size = batch_data.shape[0] if batch_data is not None else batch_size		
		batch_data = batch_data.astype(np_dtype) if batch_data is not None else None
		
		if stats_only:
			res_list = [self.nan_vars, self.inf_vars, self.zero_vars, 
						self.big_vars, self.count_vars]
			res_list = self.sess.run(res_list, feed_dict={})
			return res_list

		if z_data is None:
			z_data = np.random.normal(loc=0.0, scale=1.0, size=(batch_size, self.z_dim))

		### only forward decoder on z
		if gen_only:
			feed_dict = {self.z_input: z_data, self.train_phase: False}
			z_decode = self.sess.run(self.z_decode, feed_dict=feed_dict)
			return z_decode

		feed_dict = {self.im_input:batch_data, self.train_phase: True}
		### run one training step on vae
		if update:
			res_list = [self.im_decode, self.vae_loss_sum, self.vae_opt]
			res_list = self.sess.run(res_list, feed_dict=feed_dict)
			### return vae_loss summary and g_layer
			return res_list[1], res_list[0]
		### run image reconstruction, return reconstructed images and their z codes
		else:
			res_list = [self.im_decode, self.z_codes]
			res_list = self.sess.run(res_list, feed_dict=feed_dict)
			return res_list

	def step(self, batch_data, batch_size, gen_update=False, 
		dis_only=False, gen_only=False, stats_only=False, z_data=None):
		batch_size = batch_data.shape[0] if batch_data is not None else batch_size		
		batch_data = batch_data.astype(np_dtype) if batch_data is not None else None

		if stats_only:
			res_list = [self.nan_vars, self.inf_vars, self.zero_vars, 
						self.big_vars, self.count_vars]
			res_list = self.sess.run(res_list, feed_dict={})
			return res_list

		### sample e from uniform (0,1): for gp penalty in WGAN
		e_data = np.random.uniform(low=0.0, high=1.0, size=(batch_size, 1, 1, 1))
		e_data = e_data.astype(np_dtype)

		### only forward discriminator on batch_data
		if dis_only:
			feed_dict = {self.im_input: batch_data, self.train_phase: False}
			u_logits = self.sess.run(self.r_logits, feed_dict=feed_dict)
			return u_logits.flatten()

		### sample z from normal (-1,1)
		if z_data is None:
			z_data = np.random.normal(loc=0.0, scale=1.0, size=(batch_size, self.z_dim))

		### only forward generator on z
		if gen_only:
			feed_dict = {self.im_input:batch_data, self.z_input: z_data, self.train_phase: False}
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
