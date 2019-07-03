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
	
	k_init = tf.truncated_normal_initializer(stddev=0.02)
	conv = tf.layers.conv2d(
		input_, output_dim, [k_h, k_w], strides=[d_h, d_w], 
		padding=padding, use_bias=use_bias, 
		kernel_initializer=k_init, name=scope, reuse=reuse, trainable=trainable)

	return conv

def conv2d_tr(input_, output_dim,
		   k_h=5, k_w=5, d_h=1, d_w=1, k_init=tf.contrib.layers.xavier_initializer(),
		   scope=None, reuse=False, 
		   padding='same', use_bias=True, trainable=True):
    
    k_init = tf.truncated_normal_initializer(stddev=0.02)
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
		k_init = tf.truncated_normal_initializer(stddev=0.02)
		h1 = tf.layers.dense(x, h_size, kernel_initializer=k_init, name=scope, reuse=reuse)
		#h1 = tf.contrib.layers.fully_connected(x, h_size, activation_fn=None, scope='dense')
		#h1 = tf.contrib.layers.fully_connected(x, h_size, activation_fn=None, scope='dense', weights_initializer=tf.truncated_normal_initializer(stddev=0.02))
		#h1 = tf.contrib.layers.fully_connected(x, h_size, activation_fn=None, scope='dense', weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True))
	return h1

'''
computes cos similarity between flattened x and y (except last axis).
'''
def cos_sim(x, y):
	ch = x.shape[-1]
	x_flat = tf.reshape(x, [-1, ch])
	y_flat = tf.reshape(y, [-1, ch])
	cos_term = tf.reduce_sum(x_flat * y_flat, axis=0)
	x_norm = tf.sqrt(tf.reduce_sum(tf.square(x_flat), axis=0))
	y_norm = tf.sqrt(tf.reduce_sum(tf.square(y_flat), axis=0))
	return cos_term / (x_norm * y_norm)

'''
computes ratio of non zero weights in the given var x
'''
def non_zero_ratio(x, th=0.001):
	ch = x.shape[-1]
	x_flat = tf.reshape(x, [-1, ch])
	non_zero_count = tf.reduce_sum(tf.cast(tf.square(x_flat) > th**2, tf_dtype), axis=0)
	count = tf.cast(tf.reduce_prod(x.shape[:-1]), tf_dtype)
	return non_zero_count / count

'''
creates ops and vars for computing cos similarity between current and previous net_vars.
layer_list: list of strings containing layer names to be included in computation.
net_vars: list of tf variables containing the layer list variables.
returns layer_sim: dict of 1d tensors containing sim value for each channel of each layer.
return backup_op: op for updating the previous var values (storing new current values in them).
'''
def compute_layer_sim(layer_list, net_vars):
	layer_sim = dict()
	layer_non_zero = dict()
	vars_pre = dict()
	vars_pre_ops = list()
	with tf.variable_scope('sim_layer'):
		for li, ln in enumerate(layer_list):
			sim_list = list()
			non_zero_list = list()
			for v in net_vars:
				if ln in v.name and 'kernel' in v.name:
					print '>>> VAR: ' + ln
					print v.get_shape().as_list()
					vars_pre[v.name] = tf.get_variable(v.name.split(':')[0], shape=v.shape, trainable=False)
					vars_pre_ops.append(tf.assign(vars_pre[v.name], v))
					sim_list.append(tf.reshape(cos_sim(vars_pre[v.name], v), [-1]))
					non_zero_list.append(tf.reshape(non_zero_ratio(v), [-1]))
			if len(sim_list) > 1:
				layer_sim[ln] = tf.concat(sim_list, axis=0)
				layer_non_zero[ln] = tf.concat(non_zero_list, axis=0)
			elif len(sim_list) == 1:
				layer_sim[ln] = sim_list[0]
				layer_non_zero[ln] = non_zero_list[0]
				
	backup_op = tf.group(*vars_pre_ops)
	return backup_op, layer_sim, layer_non_zero

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
		self.z_dim = 128 #256
		self.man_dim = 0
		self.g_num = 1
		self.z_range = 1.0
		self.data_dim = [128, 128, 3]
		self.hp_loss_weight = 1.
		self.gp_loss_weight = 10.0
		self.rg_loss_weight = 0.0
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

	def apply_filter(self, im):
		### np kernel
		sigma = 1.
		krange = 10
		ksize = 2*krange + 1
		t = np.linspace(-krange, krange, ksize)
		bump = np.exp(0.5 * -t**2/sigma**2)
		bump /= np.trapz(bump) # normalize the integral to 1
		kernel = bump[:, np.newaxis] * bump[np.newaxis, :]
		### tf conv2d
		kernel = kernel.reshape((ksize, ksize, 1, 1)).astype(np_dtype)
		kernel = np.repeat(kernel, self.data_dim[-1], axis=2)
		return tf.nn.depthwise_conv2d(im, kernel, [1, 1, 1, 1], padding='SAME')

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

			### filter images
			#self.r_layer_lp = self.apply_filter(self.im_input)
			#self.r_layer_hp = self.im_input - self.r_layer_lp
			#self.g_layer_lp = self.apply_filter(self.g_layer)
			#self.g_layer_hp = self.g_layer - self.g_layer_lp

			### build discriminator
			self.r_logits, self.g_logits, self.rg_logits, self.rg_layer = \
				self.build_dis_logits(self.im_input, self.g_layer, self.train_phase, sub_scope='or')

			### build discriminator down scaled (x1/4)
			self.im_data_ds = tf.nn.avg_pool(self.im_input, 
				ksize=[1, 5, 5, 1], strides=[1, 4, 4, 1], padding='SAME')
			self.g_data_ds = tf.nn.avg_pool(self.g_layer, 
				ksize=[1, 5, 5, 1], strides=[1, 4, 4, 1], padding='SAME')
			self.r_logits_ds, self.g_logits_ds, self.rg_logits_ds, self.rg_layer_ds = \
				self.build_dis_logits(self.im_data_ds, self.g_data_ds, self.train_phase, sub_scope='ds')
			print '>>> im_ds shape:', self.im_data_ds.get_shape().as_list()

			### build discriminator up scaled (x4)
			self.im_data_us = tf.image.resize_nearest_neighbor(self.im_input, 
				[self.data_dim[0]*4, self.data_dim[1]*4])
			self.g_data_us = tf.image.resize_nearest_neighbor(self.g_layer, 
				[self.data_dim[0]*4, self.data_dim[1]*4])
			self.r_logits_us, self.g_logits_us, self.rg_logits_us, self.rg_layer_us = \
				self.build_dis_logits(self.im_data_us, self.g_data_us, self.train_phase, sub_scope='us')
			print '>>> im_ds shape:', self.im_data_us.get_shape().as_list()

			### build discriminator for low pass
			#self.r_logits, self.r_hidden = self.build_dis(self.im_input, self.train_phase)
			#self.g_logits, self.g_hidden = self.build_dis(self.g_layer, self.train_phase, reuse=True)
			#self.r_en_logits = self.build_encoder(self.r_hidden, self.d_act, self.train_phase)
			#self.g_en_logits = self.build_encoder(self.g_hidden, self.d_act, self.train_phase, reuse=True)

			### real gen manifold interpolation
			#rg_layer = (1.0 - self.e_input) * self.g_layer + self.e_input * self.im_input
			#self.rg_logits, _ = self.build_dis(rg_layer, self.train_phase, reuse=True)

			### build discriminator for high pass
			#self.r_logits_hp, self.r_hidden_hp = self.build_dis(self.r_layer_hp, self.train_phase, reuse=True)
			#self.r_logits_hp = self.build_fc(self.r_hidden_hp)
			#self.g_logits_hp, self.g_hidden_hp = self.build_dis(self.g_layer_hp, self.train_phase, reuse=True)
			#self.g_logits_hp = self.build_fc(self.g_hidden_hp, reuse=True)

			### real gen manifold interpolation for high pass
			#rg_layer_hp = (1.0 - self.e_input) * self.g_layer_hp + self.e_input * self.r_layer_hp
			#self.rg_logits_hp, rg_hidden_hp = self.build_dis(rg_layer_hp, self.train_phase, reuse=True)
			#self.rg_logits_hp = self.build_fc(rg_hidden_hp, reuse=True)

			### build d losses
			d_loss, rg_grad_norm_output = \
				self.build_dis_loss(self.r_logits, self.g_logits, self.rg_logits, self.rg_layer)
			d_loss_ds, rg_grad_norm_output_ds = \
				self.build_dis_loss(self.r_logits_ds, self.g_logits_ds, self.rg_logits_ds, self.rg_layer_ds)
			d_loss_us, rg_grad_norm_output_us = \
				self.build_dis_loss(self.r_logits_us, self.g_logits_us, self.rg_logits_us, self.rg_layer_us)
			#d_loss_hp, rg_grad_norm_output_hp = \
			#	self.build_dis_loss(self.r_logits_hp, self.g_logits_hp, self.rg_logits_hp, rg_layer_hp)
			#self.rg_grad_norm_output = (rg_grad_norm_output_hp + rg_grad_norm_output ) / 2.0
			self.rg_grad_norm_output = (rg_grad_norm_output + rg_grad_norm_output_ds + rg_grad_norm_output_us) / 3.
			self.d_loss_total = d_loss #+ \
				#d_loss_ds + d_loss_us
				#self.hp_loss_weight * d_loss_hp

			### generated encoder loss: lower bound on mutual_info(z_input, generator id) **g_num**
			#self.g_en_loss = tf.nn.softmax_cross_entropy_with_logits(
			#	labels=tf.one_hot(tf.reshape(self.z_input, [-1]), self.g_num, dtype=tf_dtype), 
			#	logits=self.g_en_logits)

			### real encoder entropy: entropy of g_id given real image, marginal entropy of g_id **g_num**
			#self.r_en_h = -tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(self.r_en_logits) * tf.nn.log_softmax(self.r_en_logits), axis=1))
			#r_en_marg_pr = tf.reduce_mean(tf.nn.softmax(self.r_en_logits), axis=0)
			#self.r_en_marg_hlb = -tf.reduce_sum(r_en_marg_pr * tf.log(r_en_marg_pr + 1e-8))
			#print 'r_en_logits_shape: ', self.r_en_logits.shape

			### discounter
			#self.rl_counter = tf.get_variable('rl_counter', dtype=tf_dtype,
			#	initializer=1.0)

			### build g loss
			g_loss = self.build_gen_loss(self.g_logits)
			g_loss_ds = self.build_gen_loss(self.g_logits_ds)
			g_loss_us = self.build_gen_loss(self.g_logits_us)
			#self.g_loss_hp = self.build_gen_loss(self.g_logits_hp)
			self.g_loss_total = g_loss #+ \
				#g_loss_ds + g_loss_us
				#self.hp_loss_weight * self.g_loss_hp
				#self.en_loss_weight * tf.reduce_mean(self.g_en_loss)

			### e loss combination
			#self.en_loss_total = tf.reduce_mean(self.g_en_loss) + \
			#	0. * self.r_en_h + 0.* -self.r_en_marg_hlb

			### collect params
			self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "g_net")
			self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "d_net")
			#self.e_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "e_net")
			self.g_vars_sub = [v for v in self.g_vars if 'fcz' in v.name]
			self.d_vars_sub = [v for v in self.d_vars if 'conv3' in v.name or 'conv2' in v.name or 'conv1' in v.name]
			print '>>> d_vars_sub:'
			print self.d_vars_sub

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
			#self.e_vars_count = 0
			for v in self.g_vars:
				self.g_vars_count += int(np.prod(v.get_shape()))
			for v in self.d_vars:
				self.d_vars_count += int(np.prod(v.get_shape()))
			#for v in self.e_vars:
			#	self.e_vars_count += int(np.prod(v.get_shape()))

			### compute conv layer learning variation
			self.g_sim_layer_list = ['conv0', 'conv1', 'conv2', 'conv3', 'fcz']
			self.g_backup, self.g_layer_sim, self.g_layer_non_zero = compute_layer_sim(self.g_sim_layer_list, self.g_vars)

			self.d_sim_layer_list = ['conv0', 'conv1', 'conv2', 'conv3', 'fco']
			self.d_backup, self.d_layer_sim, self.d_layer_non_zero  = compute_layer_sim(self.d_sim_layer_list, self.d_vars)

			### build optimizers
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			print '>>> update_ops list: ', update_ops
			with tf.control_dependencies(update_ops):
				self.g_opt = tf.train.AdamOptimizer(
					self.g_lr, beta1=self.g_beta1, beta2=self.g_beta2).minimize(
					self.g_loss_total, var_list=self.g_vars)
				self.g_sub_opt = tf.train.AdamOptimizer(
					self.g_lr, beta1=self.g_beta1, beta2=self.g_beta2).minimize(
					self.g_loss_total, var_list=self.g_vars_sub)
				self.d_opt = tf.train.AdamOptimizer(
					self.d_lr, beta1=self.d_beta1, beta2=self.d_beta2).minimize(
					self.d_loss_total, var_list=self.d_vars)
				self.d_sub_opt = tf.train.AdamOptimizer(
					self.d_lr, beta1=self.d_beta1, beta2=self.d_beta2).minimize(
					self.d_loss_total, var_list=self.d_vars_sub)
				#self.e_opt = tf.train.AdamOptimizer(
				#	self.e_lr, beta1=self.e_beta1, beta2=self.e_beta2).minimize(
				#	self.en_loss_total, var_list=self.e_vars)

			### summaries **g_num**
			g_loss_sum = tf.summary.scalar("g_loss", self.g_loss_total)
			d_loss_sum = tf.summary.scalar("d_loss", self.d_loss_total)
			#e_loss_sum = tf.summary.scalar("e_loss", self.en_loss_total)
			self.summary = tf.summary.merge([g_loss_sum, d_loss_sum])

			### Policy gradient updates **g_num**
			#self.pg_var = tf.get_variable('pg_var', dtype=tf_dtype,
			#	initializer=self.g_rl_vals)
			#self.pg_q = tf.get_variable('pg_q', dtype=tf_dtype,
			#	initializer=self.g_rl_vals)
			#self.pg_base = tf.get_variable('pg_base', dtype=tf_dtype,
			#	initializer=0.0)
			#self.pg_var_flat = self.pg_temp * tf.reshape(self.pg_var, [1, -1])
			
			### log p(x) for the selected policy at each batch location
			#log_soft_policy = -tf.nn.softmax_cross_entropy_with_logits(
			#	labels=tf.one_hot(tf.reshape(self.z_input, [-1]), self.g_num, dtype=tf_dtype), 
			#	logits=tf.tile(self.pg_var_flat, tf.shape(tf.reshape(self.z_input, [-1, 1]))))
			
			#self.gi_h = -tf.reduce_sum(tf.nn.softmax(self.pg_var) * tf.nn.log_softmax(self.pg_var))
			
			### policy gradient reward
			#pg_reward = tf.reduce_mean(self.r_en_logits, axis=0)
			
			### critic update (q values update)
			#rl_counter_opt = tf.assign(self.rl_counter, self.rl_counter * 0.999)

			### r_en_logits as q values
			#pg_q_opt = tf.assign(self.pg_q, (1-self.pg_q_lr)*self.pg_q + \
			#	self.pg_q_lr * pg_reward)

			### cross entropy E_x H(p(c|x)||q(c))
			#with tf.control_dependencies([pg_q_opt, rl_counter_opt]):
			#	en_pr = tf.nn.softmax(self.r_en_logits)
			#	pg_loss_total = -tf.reduce_mean(en_pr * tf.nn.log_softmax(self.pg_var)) \
			#		- 1000. * self.rl_counter * self.gi_h

			#self.pg_opt = tf.train.AdamOptimizer(
			#		self.pg_lr, beta1=self.pg_beta1, beta2=self.pg_beta2).minimize(
			#		pg_loss_total, var_list=[self.pg_var])

	def build_dis_logits(self, im_data, g_data, train_phase, sub_scope='or'):
		int_rand = tf.random_uniform([tf.shape(im_data)[0]], minval=0.0, maxval=1.0, dtype=tf_dtype)
		int_rand = tf.reshape(int_rand, [-1, 1, 1, 1])
		### build discriminator for low pass
		r_logits, r_hidden = self.build_dis(im_data, train_phase, sub_scope=sub_scope)
		g_logits, g_hidden = self.build_dis(g_data, train_phase, sub_scope=sub_scope, reuse=True)
		### real gen manifold interpolation
		rg_layer = (1.0 - int_rand) * g_data + int_rand * im_data
		rg_logits, _ = self.build_dis(rg_layer, train_phase, sub_scope=sub_scope, reuse=True)
		return r_logits, g_logits, rg_logits, rg_layer

	def build_dis_loss(self, r_logits, g_logits, rg_logits, rg_layer):
		### build d losses
		if self.d_loss_type == 'log':
			d_r_loss = tf.nn.sigmoid_cross_entropy_with_logits(
					logits=r_logits, labels=tf.ones_like(r_logits, tf_dtype))
			d_g_loss = tf.nn.sigmoid_cross_entropy_with_logits(
					logits=g_logits, labels=tf.zeros_like(g_logits, tf_dtype))
			d_rg_loss = tf.nn.sigmoid_cross_entropy_with_logits(
					logits=rg_logits, labels=tf.ones_like(rg_logits, tf_dtype))
		elif self.d_loss_type == 'was':
			d_r_loss = -r_logits 
			d_g_loss = g_logits
			d_rg_loss = -rg_logits
		else:
			raise ValueError('>>> d_loss_type: %s is not defined!' % self.d_loss_type)

		### gradient penalty
		### NaN free norm gradient
		rg_grad = tf.gradients(rg_logits, rg_layer)
		rg_grad_flat = tf.contrib.layers.flatten(rg_grad[0])
		rg_grad_ok = tf.reduce_sum(tf.square(rg_grad_flat), axis=1) > 1.
		rg_grad_safe = tf.where(rg_grad_ok, rg_grad_flat, tf.ones_like(rg_grad_flat))
		#rg_grad_abs = tf.where(rg_grad_flat >= 0., rg_grad_flat, -rg_grad_flat)
		rg_grad_abs =  0. * rg_grad_flat
		rg_grad_norm = tf.where(rg_grad_ok, 
			tf.norm(rg_grad_safe, axis=1), tf.reduce_sum(rg_grad_abs, axis=1))
		gp_loss = tf.square(rg_grad_norm - 1.0)
		### for logging
		rg_grad_norm_output = tf.norm(rg_grad_flat, axis=1)
		### combine losses
		d_loss_mean = tf.reduce_mean(d_r_loss + d_g_loss)
		d_loss_total = d_loss_mean + self.gp_loss_weight * tf.reduce_mean(gp_loss)
		return d_loss_total, rg_grad_norm_output

	def build_gen_loss(self, g_logits):
		if self.g_loss_type == 'log':
				g_loss = -tf.nn.sigmoid_cross_entropy_with_logits(
					logits=g_logits, labels=tf.zeros_like(g_logits, tf_dtype))
		elif self.g_loss_type == 'mod':
			g_loss = tf.nn.sigmoid_cross_entropy_with_logits(
				logits=g_logits, labels=tf.ones_like(g_logits, tf_dtype))
		elif self.g_loss_type == 'was':
			g_loss = -g_logits
		else:
			raise ValueError('>>> g_loss_type: %s is not defined!' % self.g_loss_type)

		return tf.reduce_mean(g_loss)

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
					z_fc = act(bn(dense(zi, 8*8*256*2, scope='fcz')))
					h0 = tf.reshape(z_fc, [-1, 8, 8, 256*2])

					### deconv version
					'''
					h2 = act(bn(conv2_tr(h1, 64*4, d_h=2, d_w=2, scope='conv1')))
					h3 = act(bn(conv2_tr(h2, 32*4, d_h=2, d_w=2, scope='conv2')))
					h4 = conv2_tr(h3, self.data_dim[-1], d_h=2, d_w=2, scope='conv3')
					'''

					### us version: decoding 4*4*256 code with upsampling and conv hidden layers into 32*32*3
					h0_us = tf.image.resize_nearest_neighbor(h0, [im_size//8, im_size//8], name='us0')
					h1 = act(bn(conv2d(h0_us, 128*2, scope='conv0')))

					h1_us = tf.image.resize_nearest_neighbor(h1, [im_size//4, im_size//4], name='us1')
					h2 = act(bn(conv2d(h1_us, 64*2, scope='conv1')))

					h2_us = tf.image.resize_nearest_neighbor(h2, [im_size//2, im_size//2], name='us2')
					h3 = act(bn(conv2d(h2_us, 32*2, scope='conv2')))
				
					h3_us = tf.image.resize_nearest_neighbor(h3, [im_size, im_size], name='us3')
					h4 = conv2d(h3_us, self.data_dim[-1], scope='conv3')

					### resnext version
					'''
					btnk_dim = 64
					h2 = resnext(h1, 128, btnk_dim, 'res1', train_phase, 
								op_type='up', bn=False, act=act)
					h3 = resnext(h2, 64, btnk_dim//2, 'res2', train_phase,
								op_type='up', bn=False, act=act)
					h4 = resnext(h3, 32, btnk_dim//4, 'res3', train_phase, 
								op_type='up', bn=False, act=act)
					h5 = conv2d(h4, self.data_dim[-1], scope='convo')
					'''
					ol.append(tf.tanh(h4))

			#z_1_hot = tf.reshape(tf.one_hot(z, self.g_num, dtype=tf_dtype), [-1, self.g_num, 1, 1, 1])
			#z_map = tf.tile(z_1_hot, [1, 1]+self.data_dim)
			#os = tf.stack(ol, axis=1)
			#o = tf.reduce_sum(os * z_map, axis=1)
			o = ol[0]
			return o

	def build_dis(self, data_layer, train_phase, reuse=False, sub_scope='or'):
		act = self.d_act
		with tf.variable_scope('d_net'):
			with tf.variable_scope(sub_scope):
				bn = tf.contrib.layers.batch_norm
				### encoding the 28*28*3 image with conv into 3*3*256
				
				h0 = act(conv2d(data_layer, 32*2, d_h=2, d_w=2, scope='conv0', reuse=reuse))
				h1 = act(conv2d(h0, 64*2, d_h=2, d_w=2, scope='conv1', reuse=reuse))
				h2 = act(conv2d(h1, 128*2, d_h=2, d_w=2, scope='conv2', reuse=reuse))
				h3 = act(conv2d(h2, 256*2, d_h=2, d_w=2, scope='conv3', reuse=reuse))
				#h4 = conv2d(h2, 1, d_h=1, d_w=1, k_h=1, k_w=1, padding='VALID', scope='conv4', reuse=reuse)
				'''
				### resnext version
				btnk_dim = 64
				h1 = resnext(data_layer, 32, btnk_dim//4, 'res1', train_phase, 
							op_type='down', bn=False, act=act, reuse=reuse)
				h2 = resnext(h1, 64, btnk_dim//2, 'res2', train_phase, 
							op_type='down', bn=False, act=act, reuse=reuse)
				h3 = resnext(h2, 128, btnk_dim, 'res3', train_phase, 
							op_type='down', bn=False, act=act, reuse=reuse)
				'''
				### fully connected discriminator
				flat = tf.contrib.layers.flatten(h3)
				o = dense(flat, 1, scope='fco', reuse=reuse)
				return o, flat

	def build_fc(self, hidden_layer, reuse=False):
		with tf.variable_scope('d_net'):
			return dense(hidden_layer, 1, scope='dfilter_fc', reuse=reuse)

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
		self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
		self.saver_var_only = tf.train.Saver(self.g_vars+self.d_vars)#+self.e_vars+[self.pg_var, self.pg_q])
		self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

	def save(self, fname):
		self.saver.save(self.sess, fname)

	def load(self, fname):
		self.saver_var_only.restore(self.sess, fname)

	def write_sum(self, sum_str, counter):
		self.writer.add_summary(sum_str, counter)

	def step(self, batch_data, batch_size, gen_update=False, 
		dis_only=False, gen_only=False, stats_only=False, 
		g_layer_stats=False, d_layer_stats=False,
		en_only=False, z_data=None, zi_data=None, run_count=0.0, 
		filter_only=False, g_output_type='or'):
		batch_size = batch_data.shape[0] if batch_data is not None else batch_size		
		batch_data = batch_data.astype(np_dtype) if batch_data is not None else None

		if stats_only:
			res_list = [self.nan_vars, self.inf_vars, self.zero_vars, self.big_vars]
			res_list = self.sess.run(res_list, feed_dict={})
			return res_list

		if g_layer_stats or d_layer_stats:
			### compute layer sims
			sim_dict = self.g_layer_sim if g_layer_stats else self.d_layer_sim
			sim_dict = self.sess.run(sim_dict, feed_dict={})
			### compute layer non zero ratio
			non_zero_dict = self.g_layer_non_zero if g_layer_stats else self.d_layer_non_zero
			non_zero_dict = self.sess.run(non_zero_dict, feed_dict={})
			### update previous backup vars
			backup_op = self.g_backup if g_layer_stats else self.d_backup
			self.sess.run(backup_op)

			return sim_dict, non_zero_dict

		### sample e from uniform (0,1): for gp penalty in WGAN
		e_data = np.random.uniform(low=0.0, high=1.0, size=(batch_size, 1, 1, 1))
		e_data = e_data.astype(np_dtype)

		### sample z from uniform (-1,1)
		if zi_data is None:
			zi_data = np.random.uniform(low=-self.z_range, high=self.z_range, 
				size=[batch_size, self.z_dim])
		zi_data = zi_data.astype(np_dtype)
		
		### multiple generator uses z_data to select gen **g_num**
		#self.g_rl_vals, self.g_rl_pvals = self.sess.run((self.pg_q, self.pg_var), feed_dict={})
		if z_data is None:
			#g_th = min(1 + self.rl_counter // 1000, self.g_num)
			#g_th = self.g_num
			#z_pr = np.exp(self.pg_temp * self.g_rl_pvals[:g_th])
			#z_pr = z_pr / np.sum(z_pr)
			#z_data = np.random.choice(g_th, size=batch_size, p=z_pr)
			z_data = np.random.randint(low=0, high=self.g_num, size=batch_size)
		
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
			#go_layer = self.g_layer if filter_only is False else self.g_layer_lp
			if g_output_type == 'ds':
				feed_dict = {self.im_input: batch_data, self.z_input: z_data, 
					self.zi_input: zi_data, self.train_phase: False}
				go_layer = self.im_data_ds
			elif g_output_type == 'us':
				feed_dict = {self.im_input: batch_data, self.z_input: z_data, 
					self.zi_input: zi_data, self.train_phase: False}
				go_layer = self.im_data_us
			else:
				go_layer = self.g_layer
			imo_layer = self.sess.run(go_layer, feed_dict=feed_dict)
			return imo_layer

		### only filter
		if filter_only:
			feed_dict = {self.im_input:batch_data, self.train_phase: False}
			imo_layer = self.sess.run(self.r_layer_lp, feed_dict=feed_dict)
			return imo_layer

		### run one training step on discriminator, otherwise on generator, and log **g_num**
		feed_dict = {self.im_input:batch_data, self.z_input: z_data, self.zi_input: zi_data,
					self.e_input: e_data, self.train_phase: True}
		if not gen_update:
			d_opt_ptr = self.d_opt #if run_count < 5e4 else self.d_sub_opt
			res_list = [self.g_layer, self.summary, d_opt_ptr]
			res_list = self.sess.run(res_list, feed_dict=feed_dict)
		else:
			g_opt_ptr = self.g_opt #if run_count < 5e4 else self.g_sub_opt
			res_list = [self.g_layer, self.summary, g_opt_ptr]
						#self.e_opt, self.pg_opt]
						#self.r_en_h, self.r_en_marg_hlb, self.gi_h, self.g_en_loss]
			res_list = self.sess.run(res_list, feed_dict=feed_dict)
			
		### return summary and g_layer
		return res_list[1], res_list[0]
