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
def lrelu(x, leak=0.1, name="lrelu"):
	with tf.variable_scope(name):
		f1 = 0.5 * (1 + leak)
		f2 = 0.5 * (1 - leak)
		return f1 * x + f2 * abs(x)

def conv2d(input_, output_dim,
		   k_h=5, k_w=5, d_h=1, d_w=1, k_init=tf.contrib.layers.xavier_initializer(),
		   scope=None, reuse=False, 
		   padding='same', use_bias=True, trainable=True):
	
	conv = tf.layers.conv2d(
		input_, output_dim, [k_h, k_w], strides=[d_h, d_w], 
		padding=padding, use_bias=use_bias, 
		kernel_initializer=k_init, name=scope, reuse=reuse, trainable=trainable)

	return conv

def conv2d_tr(input_, output_dim,
		   k_h=5, k_w=5, d_h=1, d_w=1, k_init=tf.contrib.layers.xavier_initializer(),
		   scope=None, reuse=False, 
		   padding='same', use_bias=True, trainable=True):
    
    conv_tr = tf.layers.conv2d_transpose(
            input_, output_dim, [k_h, k_w], strides=[d_h, d_w], 
            padding=padding, use_bias=use_bias, 
            kernel_initializer=k_init, name=scope, reuse=reuse, trainable=trainable)
    
    return conv_tr

'''
Group ResNext.
filter_dim: internal bottle neck filter dim.
group_dim: each group conv output dim (must be a factor of filter size).
input_: shape (b, h, w, c)
'''
def resnext(input_, output_dim, filter_dim, scope, train_phase,
			op_type='same', bn=True, act=lrelu, group_dim=4, project_input=False, reuse=False):
	with tf.variable_scope(scope, reuse=reuse):
		
		def bn_id(x, training=True):
			return x
		bn = bn_id if bn is False else tf.layers.batch_normalization
		#bn = tf.contrib.layers.batch_norm
		imh = tf.shape(input_)[1]
		imw = tf.shape(input_)[2]

		if input_.get_shape().as_list()[3] != output_dim or project_input is True:
			shortcut = conv2d(input_, output_dim, k_h=1, k_w=1)
		else:
			shortcut = input_

		if op_type == 'down':
			conv_op = conv2d
			conv_dh = conv_dw = 2
			shortcut = tf.nn.pool(shortcut, [5, 5], "AVG", "SAME", strides=[conv_dh, conv_dw])
		elif op_type == 'up':
			conv_op = conv2d_tr
			conv_dh = conv_dw = 2
			shortcut = tf.image.resize_nearest_neighbor(shortcut, [imh*2, imw*2])
		elif op_type == 'same':
			conv_op = conv2d
			conv_dh = conv_dw = 1
			shortcut = shortcut

		### reduce to bottleneck size
		hd = act(bn(conv2d(input_, filter_dim, k_h=1, k_w=1), training=train_phase))
		
		### apply group conv
		hg = list()
		for i in range(0, filter_dim, group_dim):
			hg.append(act(bn(conv_op(hd[:, :, :, i:i+group_dim], group_dim, d_h=conv_dh, d_w=conv_dw), training=train_phase)))

		### concat and bring to output_dim size
		hg_concat = tf.concat(hg, axis=3)
		output = conv2d(hg_concat, output_dim, k_h=1, k_w=1)
		return act(bn(output + shortcut, training=train_phase))


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
		self.e_lr = 2e-4
		self.e_beta1 = 0.9
		self.e_beta2 = 0.999
		self.pg_lr = 1e-3
		self.pg_beta1 = 0.5
		self.pg_beta2 = 0.5

		### network parameters **g_num** **mt**
		### >>> dataset sensitive: data_dim
		self.z_dim = 100 #256
		self.man_dim = 0
		self.g_num = 1
		self.z_range = 1.0
		self.data_dim = [64, 64, 3]
		self.mm_loss_weight = 0.0
		self.gp_loss_weight = 10.0
		self.rg_loss_weight = 0.0
		self.rec_penalty_weight = 0.0
		self.en_loss_weight = 0.0
		self.rl_lr = 0.99
		self.pg_q_lr = 0.01
		self.rl_bias = 0.0
		self.pg_temp = 1.0
		self.g_rl_vals = 0. * np.ones(self.g_num, dtype=np_dtype)
		self.g_rl_pvals = 0. * np.ones(self.g_num, dtype=np_dtype)
		self.d_loss_type = 'was'
		self.g_loss_type = 'was'
		#self.d_act = tf.tanh
		#self.g_act = tf.tanh
		self.d_act = lrelu
		self.g_act = tf.nn.relu

		### init graph and session
		self.build_graph()
		self.start_session()

	def build_graph(self):
		with tf.name_scope('ganist'):
			### define placeholders for image and label inputs **g_num** **mt**
			self.im_input = tf.placeholder(tf_dtype, [None]+self.data_dim, name='im_input')
			#self.z_input = tf.placeholder(tf_dtype, [None, self.z_dim], name='z_input')
			#self.z_input = tf.placeholder(tf_dtype, [None, 1, 1, 1], name='z_input')
			self.z_input = tf.placeholder(tf.int32, [None], name='z_input')
			self.zi_input = tf.placeholder(tf_dtype, [None, self.z_dim], name='zi_input')
			self.e_input = tf.placeholder(tf_dtype, [None, 1, 1, 1], name='e_input')
			self.train_phase = tf.placeholder(tf.bool, name='phase')

			### build generator **mt**
			self.g_layer = self.build_gen(self.z_input, self.zi_input, self.g_act, self.train_phase)
			#self.g_layer = self.build_gen_mt(self.im_input, self.z_input, self.g_act, self.train_phase)

			### build discriminator
			self.r_logits, self.r_hidden = self.build_dis(self.im_input, self.d_act, self.train_phase)
			self.g_logits, self.g_hidden = self.build_dis(self.g_layer, self.d_act, self.train_phase, reuse=True)
			self.r_en_logits = self.build_encoder(self.r_hidden, self.d_act, self.train_phase)
			self.g_en_logits = self.build_encoder(self.g_hidden, self.d_act, self.train_phase, reuse=True)

			### real gen manifold interpolation
			rg_layer = (1.0 - self.e_input) * self.g_layer + self.e_input * self.im_input
			self.rg_logits, _ = self.build_dis(rg_layer, self.d_act, self.train_phase, reuse=True)

			### build d losses
			if self.d_loss_type == 'log':
				self.d_r_loss = tf.nn.sigmoid_cross_entropy_with_logits(
						logits=self.r_logits, labels=tf.ones_like(self.r_logits, tf_dtype))
				self.d_g_loss = tf.nn.sigmoid_cross_entropy_with_logits(
						logits=self.g_logits, labels=tf.zeros_like(self.g_logits, tf_dtype))
				self.d_rg_loss = tf.nn.sigmoid_cross_entropy_with_logits(
						logits=self.rg_logits, labels=tf.ones_like(self.rg_logits, tf_dtype))
			elif self.d_loss_type == 'was':
				self.d_r_loss = -self.r_logits 
				self.d_g_loss = self.g_logits
				self.d_rg_loss = -self.rg_logits
			else:
				raise ValueError('>>> d_loss_type: %s is not defined!' % self.d_loss_type)

			### gradient penalty
			### NaN free norm gradient
			rg_grad = tf.gradients(self.rg_logits, rg_layer)
			rg_grad_flat = tf.reshape(rg_grad, [-1, np.prod(self.data_dim)])
			rg_grad_ok = tf.reduce_sum(tf.square(rg_grad_flat), axis=1) > 1.
			rg_grad_safe = tf.where(rg_grad_ok, rg_grad_flat, tf.ones_like(rg_grad_flat))
			#rg_grad_abs = tf.where(rg_grad_flat >= 0., rg_grad_flat, -rg_grad_flat)
			rg_grad_abs =  0. * rg_grad_flat
			rg_grad_norm = tf.where(rg_grad_ok, 
				tf.norm(rg_grad_safe, axis=1), tf.reduce_sum(rg_grad_abs, axis=1))
			gp_loss = tf.square(rg_grad_norm - 1.0)
			### for logging
			self.rg_grad_norm_output = tf.norm(rg_grad_flat, axis=1)
			
			### d loss combination **g_num**
			self.d_loss_mean = tf.reduce_mean(self.d_r_loss + self.d_g_loss)
			self.d_loss_total = self.d_loss_mean + self.gp_loss_weight * tf.reduce_mean(gp_loss)

			### build g loss
			if self.g_loss_type == 'log':
				self.g_loss = -tf.nn.sigmoid_cross_entropy_with_logits(
					logits=self.g_logits, labels=tf.zeros_like(self.g_logits, tf_dtype))
			elif self.g_loss_type == 'mod':
				self.g_loss = tf.nn.sigmoid_cross_entropy_with_logits(
					logits=self.g_logits, labels=tf.ones_like(self.g_logits, tf_dtype))
			elif self.g_loss_type == 'was':
				self.g_loss = -self.g_logits
			else:
				raise ValueError('>>> g_loss_type: %s is not defined!' % self.g_loss_type)

			self.g_loss_mean = tf.reduce_mean(self.g_loss, axis=None)
			self.g_grad_norm = tf.norm(tf.reshape(
				tf.gradients(self.g_loss, self.g_layer), [-1, np.prod(self.data_dim)]), axis=1)

			### mean matching
			mm_loss = tf.reduce_mean(tf.square(tf.reduce_mean(self.g_layer, axis=0) - tf.reduce_mean(self.im_input, axis=0)), axis=None)

			### reconstruction penalty
			rec_penalty = tf.reduce_mean(tf.minimum(tf.log(tf.reduce_sum(
				tf.square(self.g_layer - self.im_input), axis=[1, 2, 3])+1e-6), 6.)) \
				+ tf.reduce_mean(tf.minimum(tf.log(tf.reduce_sum(
				tf.square(self.g_layer - tf.reverse(self.im_input, axis=[0])), axis=[1, 2, 3])+1e-6), 6.))

			### generated encoder loss: lower bound on mutual_info(z_input, generator id) **g_num**
			self.g_en_loss = tf.nn.softmax_cross_entropy_with_logits(
				labels=tf.one_hot(tf.reshape(self.z_input, [-1]), self.g_num, dtype=tf_dtype), 
				logits=self.g_en_logits)

			### real encoder entropy: entropy of g_id given real image, marginal entropy of g_id **g_num**
			self.r_en_h = -tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(self.r_en_logits) * tf.nn.log_softmax(self.r_en_logits), axis=1))
			r_en_marg_pr = tf.reduce_mean(tf.nn.softmax(self.r_en_logits), axis=0)
			self.r_en_marg_hlb = -tf.reduce_sum(r_en_marg_pr * tf.log(r_en_marg_pr + 1e-8))
			print 'r_en_logits_shape: ', self.r_en_logits.shape

			### discounter
			self.rl_counter = tf.get_variable('rl_counter', dtype=tf_dtype,
				initializer=1.0)

			### g loss combination **g_num**
			#self.g_loss_mean += self.mm_loss_weight * mm_loss - self.rec_penalty_weight * rec_penalty
			self.g_loss_total = self.g_loss_mean + self.en_loss_weight * tf.reduce_mean(self.g_en_loss)

			### e loss combination
			self.en_loss_total = tf.reduce_mean(self.g_en_loss) + \
				0. * self.r_en_h + 0.* -self.r_en_marg_hlb

			### collect params
			self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "g_net")
			self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "d_net")
			self.e_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "e_net")

			### compute stat of weights
			self.nan_vars = 0.
			self.inf_vars = 0.
			self.zero_vars = 0.
			self.big_vars = 0.
			self.count_vars = 0
			for v in self.g_vars + self.d_vars:
				self.nan_vars += tf.reduce_sum(tf.cast(tf.is_nan(v), tf_dtype))
				self.inf_vars += tf.reduce_sum(tf.cast(tf.is_inf(v), tf_dtype))
				self.zero_vars += tf.reduce_sum(tf.cast(tf.square(v) < 1e-6, tf_dtype))
				self.big_vars += tf.reduce_sum(tf.cast(tf.square(v) > 1., tf_dtype))
				self.count_vars += tf.reduce_prod(v.get_shape())
			self.count_vars = tf.cast(self.count_vars, tf_dtype)
			#self.nan_vars /= self.count_vars 
			#self.inf_vars /= self.count_vars
			self.zero_vars /= self.count_vars
			self.big_vars /= self.count_vars

			self.g_vars_count = 0
			self.d_vars_count = 0
			self.e_vars_count = 0
			for v in self.g_vars:
				self.g_vars_count += int(np.prod(v.get_shape()))
			for v in self.d_vars:
				self.d_vars_count += int(np.prod(v.get_shape()))
			for v in self.e_vars:
				self.e_vars_count += int(np.prod(v.get_shape()))

			### build optimizers
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			print '>>> update_ops list: ', update_ops
			with tf.control_dependencies(update_ops):
				self.g_opt = tf.train.AdamOptimizer(
					self.g_lr, beta1=self.g_beta1, beta2=self.g_beta2).minimize(
					self.g_loss_total, var_list=self.g_vars)
				self.d_opt = tf.train.AdamOptimizer(
					self.d_lr, beta1=self.d_beta1, beta2=self.d_beta2).minimize(
					self.d_loss_total, var_list=self.d_vars)
				self.e_opt = tf.train.AdamOptimizer(
					self.e_lr, beta1=self.e_beta1, beta2=self.e_beta2).minimize(
					self.en_loss_total, var_list=self.e_vars)

			### summaries **g_num**
			g_loss_sum = tf.summary.scalar("g_loss", self.g_loss_mean)
			d_loss_sum = tf.summary.scalar("d_loss", self.d_loss_mean)
			e_loss_sum = tf.summary.scalar("e_loss", self.en_loss_total)
			self.summary = tf.summary.merge([g_loss_sum, d_loss_sum, e_loss_sum])

			### Policy gradient updates **g_num**
			self.pg_var = tf.get_variable('pg_var', dtype=tf_dtype,
				initializer=self.g_rl_vals)
			self.pg_q = tf.get_variable('pg_q', dtype=tf_dtype,
				initializer=self.g_rl_vals)
			self.pg_base = tf.get_variable('pg_base', dtype=tf_dtype,
				initializer=0.0)
			self.pg_var_flat = self.pg_temp * tf.reshape(self.pg_var, [1, -1])
			
			### log p(x) for the selected policy at each batch location
			log_soft_policy = -tf.nn.softmax_cross_entropy_with_logits(
				labels=tf.one_hot(tf.reshape(self.z_input, [-1]), self.g_num, dtype=tf_dtype), 
				logits=tf.tile(self.pg_var_flat, tf.shape(tf.reshape(self.z_input, [-1, 1]))))
			
			self.gi_h = -tf.reduce_sum(tf.nn.softmax(self.pg_var) * tf.nn.log_softmax(self.pg_var))
			
			### policy gradient reward
			#pg_reward = tf.reshape(self.d_g_loss, [-1]) - 0.*self.en_loss_weight * tf.reshape(self.g_en_loss, [-1])
			pg_reward = tf.reduce_mean(self.r_en_logits, axis=0)
			
			### critic update (q values update)
			#pg_q_z = tf.gather(self.pg_q, tf.reshape(self.z_input, [-1]))
			#pg_q_opt = tf.scatter_update(self.pg_q, tf.reshape(self.z_input, [-1]), 
			#		self.pg_q_lr*pg_q_z + (1-self.pg_q_lr) * pg_reward)
			rl_counter_opt = tf.assign(self.rl_counter, self.rl_counter * 0.999)

			### r_en_logits as q values
			pg_q_opt = tf.assign(self.pg_q, (1-self.pg_q_lr)*self.pg_q + \
				self.pg_q_lr * pg_reward)

			### cross entropy E_x H(p(c|x)||q(c))
			with tf.control_dependencies([pg_q_opt, rl_counter_opt]):
				en_pr = tf.nn.softmax(self.r_en_logits)
				pg_loss_total = -tf.reduce_mean(en_pr * tf.nn.log_softmax(self.pg_var)) \
					- 1000. * self.rl_counter * self.gi_h	

			### actor update (p values update)
			#with tf.control_dependencies([pg_q_opt, rl_counter_opt]):
			#	pg_q_zu = tf.gather(self.pg_q, tf.reshape(self.z_input, [-1]))
			#	pg_loss_total = -tf.reduce_mean(log_soft_policy * pg_q_zu) + \
			#		1000. * self.rl_counter * -self.gi_h

			self.pg_opt = tf.train.AdamOptimizer(
					self.pg_lr, beta1=self.pg_beta1, beta2=self.pg_beta2).minimize(
					pg_loss_total, var_list=[self.pg_var])
			#self.pg_opt = tf.train.GradientDescentOptimizer(self.pg_lr).minimize(
			#	pg_loss_total, var_list=[self.pg_var])


	def build_gen_mt(self, im_data, z, act, train_phase):
		with tf.variable_scope('g_net'):
			### interpolate im_data with its reverse, odd batch_size gives true image
			zi = (1.0 - z) * im_data + z * tf.reverse(im_data, [0])
			h1 = act(conv2d(zi, 32, scope='conv1'))
			h2 = act(conv2d(h1, 32, scope='conv2'))
			h3 = act(conv2d(h2, 32, scope='conv3'))
			h4 = conv2d(h3, self.data_dim[-1], scope='conv4')
			o = tf.tanh(h4)
			#o = h4 + zi
			return o

	def build_gen(self, z, zi, act, train_phase):
		ol = list()
		with tf.variable_scope('g_net'):
			for gi in range(self.g_num):
				with tf.variable_scope('gnum_%d' % gi):
					im_size = self.data_dim[0]
					batch_size = tf.shape(z)[0]

					### **g_num**
					zi = tf.random_uniform([batch_size, self.z_dim], 
						minval=-self.z_range, maxval=self.z_range, dtype=tf_dtype)
					bn = tf.contrib.layers.batch_norm
			
					### fully connected from hidden z 44128 to image shape
					z_fc = act(bn(dense(zi, 8*8*128*2, scope='fcz')))
					h1 = tf.reshape(z_fc, [-1, 8, 8, 128*2])

					### deconv version
					'''
					h2 = act(bn(conv2_tr(h1, 64*4, d_h=2, d_w=2, scope='conv1')))
					h3 = act(bn(conv2_tr(h2, 32*4, d_h=2, d_w=2, scope='conv2')))
					h4 = conv2_tr(h3, self.data_dim[-1], d_h=2, d_w=2, scope='conv3')
					'''

					### us version: decoding 4*4*256 code with upsampling and conv hidden layers into 32*32*3
					'''
					h1_us = tf.image.resize_nearest_neighbor(h1, [im_size//4, im_size//4], name='us1')
					h2 = act(conv2d(h1_us, 64*4, scope='conv1'))

					h2_us = tf.image.resize_nearest_neighbor(h2, [im_size//2, im_size//2], name='us2')
					h3 = act(conv2d(h2_us, 32*4, scope='conv2'))
				
					h3_us = tf.image.resize_nearest_neighbor(h3, [im_size, im_size], name='us3')
					h4 = conv2d(h3_us, self.data_dim[-1], scope='conv3')
					'''

					### resnext version
					btnk_dim = 128
					h2 = resnext(h1, 128*2, btnk_dim, 'res1', train_phase, 
								op_type='up', bn=False, act=act)
					h3 = resnext(h2, 128*2, btnk_dim, 'res2', train_phase,
								op_type='up', bn=False, act=act)
					h4 = resnext(h3, 128*2, btnk_dim, 'res3', train_phase, 
								op_type='up', bn=False, act=act)
					h5 = conv2d(h4, self.data_dim[-1], scope='convo')
					ol.append(tf.tanh(h5))

			z_1_hot = tf.reshape(tf.one_hot(z, self.g_num, dtype=tf_dtype), [-1, self.g_num, 1, 1, 1])
			z_map = tf.tile(z_1_hot, [1, 1]+self.data_dim)
			os = tf.stack(ol, axis=1)
			o = tf.reduce_sum(os * z_map, axis=1)
			#o = ol[0]
			return o

	def build_dis(self, data_layer, act, train_phase, reuse=False):
		with tf.variable_scope('d_net'):
			bn = tf.contrib.layers.batch_norm
			### encoding the 28*28*3 image with conv into 3*3*256
			'''
			h1 = act(conv2d(data_layer, 32*4, d_h=2, d_w=2, scope='conv1', reuse=reuse))
			h2 = act(conv2d(h1, 64*4, d_h=2, d_w=2, scope='conv2', reuse=reuse))
			h3 = act(conv2d(h2, 128*4, d_h=2, d_w=2, scope='conv3', reuse=reuse))
			#h4 = conv2d(h2, 1, d_h=1, d_w=1, k_h=1, k_w=1, padding='VALID', scope='conv4', reuse=reuse)
			'''
			### resnext version
			btnk_dim = 128
			h1 = resnext(data_layer, 128*2, btnk_dim, 'res1', train_phase, 
						op_type='down', bn=False, act=act, reuse=reuse)
			h2 = resnext(h1, 128*2, btnk_dim, 'res2', train_phase, 
						op_type='down', bn=False, act=act, reuse=reuse)
			h3 = resnext(h2, 128*2, btnk_dim, 'res3', train_phase, 
						op_type='down', bn=False, act=act, reuse=reuse)
			### fully connected discriminator
			flat = tf.contrib.layers.flatten(h3)
			o = dense(flat, 1, scope='fco', reuse=reuse)
			return o, flat

	def build_encoder(self, hidden_layer, act, train_phase, reuse=False):
		bn = tf.contrib.layers.batch_norm
		with tf.variable_scope('e_net'):
			with tf.variable_scope('encoder'):
				### encoding the data_layer into number of generators
				flat = hidden_layer
				flat = act(bn(dense(flat, 128*4, scope='fc', reuse=reuse), 
					reuse=reuse, scope='bf1', is_training=train_phase))
				o = dense(flat, self.g_num, scope='fco', reuse=reuse)
				return o

	def start_session(self):
		self.saver = tf.train.Saver(tf.global_variables(), 
			keep_checkpoint_every_n_hours=1, max_to_keep=5)
		self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

	def save(self, fname):
		self.saver.save(self.sess, fname)

	def load(self, fname):
		self.saver.restore(self.sess, fname)

	def write_sum(self, sum_str, counter):
		self.writer.add_summary(sum_str, counter)

	def step(self, batch_data, batch_size, gen_update=False, 
		dis_only=False, gen_only=False, stats_only=False, 
		en_only=False, z_data=None, zi_data=None):
		batch_size = batch_data.shape[0] if batch_data is not None else batch_size		
		batch_data = batch_data.astype(np_dtype) if batch_data is not None else None

		if stats_only:
			res_list = [self.nan_vars, self.inf_vars, self.zero_vars, self.big_vars]
			res_list = self.sess.run(res_list, feed_dict={})
			return res_list

		### sample e from uniform (0,1): for gp penalty in WGAN
		e_data = np.random.uniform(low=0.0, high=1.0, size=(batch_size, 1, 1, 1))
		e_data = e_data.astype(np_dtype)

		### sample z from uniform (-1,1)
		if zi_data is None:
			zi_data = np.random.uniform(low=-self.z_range, high=self.z_range, 
				size=[batch_size, self.z_dim])
			#z_data = np.random.uniform(low=-self.z_range, high=self.z_range, 
			#	size=[batch_size, self.z_dim-self.man_dim])
			#z_data = np.random.normal(loc=0.0, scale=1.0, size=(batch_size, self.z_dim))
			#if self.man_dim > 0:
				### select manifold of each random point (1 hot)
			#	man_id = np.random.choice(self.man_dim, batch_size)
			#	z_man = np.zeros((batch_size, self.man_dim))
			#	z_man[range(batch_size), man_id] = 1.
			#	z_data = np.concatenate([z_data, z_man], axis=1)
		zi_data = zi_data.astype(np_dtype)
		
		### multiple generator uses z_data to select gen **g_num**
		self.g_rl_vals, self.g_rl_pvals = self.sess.run((self.pg_q, self.pg_var), feed_dict={})
		if z_data is None:
			#g_th = min(1 + self.rl_counter // 1000, self.g_num)
			#g_th = self.g_num
			#z_pr = np.exp(self.pg_temp * self.g_rl_pvals[:g_th])
			#z_pr = z_pr / np.sum(z_pr)
			#z_data = np.random.choice(g_th, size=batch_size, p=z_pr)
			z_data = np.random.randint(low=0, high=self.g_num, size=batch_size)
		
		### z_data for manifold transform **mt**
		'''
		if z_data is None:
			### sample e from uniform (0,1): for interpolation in mt
			z_data = np.random.uniform(low=0.0, high=1.0, size=(batch_size, 1, 1, 1))
		z_data = z_data.astype(np_dtype)
		'''
		### only forward discriminator on batch_data
		if dis_only:
			feed_dict = {self.im_input: batch_data, self.z_input: z_data, self.zi_input: zi_data,
						self.e_input: e_data, self.train_phase: False}
			res_list = [self.r_logits, self.rg_grad_norm_output]
			res_list = self.sess.run(res_list, feed_dict=feed_dict)
			return res_list[0].flatten(), res_list[1].flatten()

		### only forward encoder on batch_data
		if en_only:
			feed_dict = {self.im_input: batch_data, self.train_phase: False}
			en_logits = self.sess.run(self.r_en_logits, feed_dict=feed_dict)
			return en_logits

		### only forward generator on z
		if gen_only:
			feed_dict = {self.z_input: z_data, self.zi_input: zi_data, self.train_phase: False}
			g_layer = self.sess.run(self.g_layer, feed_dict=feed_dict)
			return g_layer

		### run one training step on discriminator, otherwise on generator, and log **g_num**
		feed_dict = {self.im_input:batch_data, self.z_input: z_data, self.zi_input: zi_data,
					self.e_input: e_data, self.train_phase: True}
		if not gen_update:
			res_list = [self.g_layer, self.summary, self.d_opt]
			res_list = self.sess.run(res_list, feed_dict=feed_dict)
		else:
			res_list = [self.g_layer, self.summary, 
						self.g_opt, self.e_opt, self.pg_opt]
						#self.r_en_h, self.r_en_marg_hlb, self.gi_h, self.g_en_loss]
			res_list = self.sess.run(res_list, feed_dict=feed_dict)
			#self.g_rl_vals, self.g_rl_pvals = self.sess.run((self.pg_q, self.pg_var), feed_dict={})
			#print '>>> gi_h: ', res_list[7]
			### RL value updates
			#self.g_rl_vals[z_data] += (1-self.rl_lr) * \
			#	(-res_list[3][:,0] - self.g_rl_vals[z_data])
			#self.g_rl_vals += 1e-3
			#self.rl_counter += 1
		#print '>>> r_logits shape:', res_list[3].shape
		### return summary and g_layer
		return res_list[1], res_list[0]
