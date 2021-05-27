import numpy as np
import tensorflow as tf
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
#os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0, 1" for multiple

#np.random.seed(13)
#tf.set_random_seed(13)

tf_dtype = tf.float32
np_dtype = 'float32'

### Operations from pggan
#----------------------------------------------------------------------------
# Get/create weight tensor for a convolutional or fully-connected layer.

def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
	if fan_in is None: fan_in = np.prod(shape[:-1])
	std = gain / np.sqrt(fan_in) # He init
	if use_wscale:
		wscale = tf.constant(np.float32(std), name='wscale')
		return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
	else:
		return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))

#----------------------------------------------------------------------------
# Fully-connected layer.

def dense_ws(x, fmaps, gain=np.sqrt(2), use_wscale=False, scope='dense', reuse=False):
	with tf.variable_scope(scope, reuse=reuse):
		if len(x.shape) > 2:
			x = tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])
		w = get_weight([x.shape[-1].value, fmaps], gain=gain, use_wscale=use_wscale)
		w = tf.cast(w, x.dtype)
		return tf.matmul(x, w)

#----------------------------------------------------------------------------
# Convolutional layer.

def conv2d_ws(x, fmaps, kernel=5, gain=np.sqrt(2), use_wscale=False, scope='conv', reuse=False):
	with tf.variable_scope(scope, reuse=reuse):
		assert kernel >= 1 and kernel % 2 == 1
		w = get_weight([kernel, kernel, x.shape[-1].value, fmaps], gain=gain, use_wscale=use_wscale)
		w = tf.cast(w, x.dtype)
		return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

#----------------------------------------------------------------------------
# Apply bias to the given activation tensor.

def apply_bias(x, scope='bias', reuse=False):
	with tf.variable_scope(scope, reuse=reuse):
		b = tf.get_variable('bias', shape=[x.shape[-1]], initializer=tf.initializers.zeros())
		b = tf.cast(b, x.dtype)
		if len(x.shape) == 2:
			return x + b
		else:
			return x + tf.reshape(b, [1, 1, 1, -1])

#----------------------------------------------------------------------------
# Leaky ReLU activation. Same as tf.nn.leaky_relu, but supports FP16.

def lrelu(x, alpha=0.2):
	with tf.name_scope('LeakyRelu'):
		alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
		return tf.maximum(x * alpha, x)

#----------------------------------------------------------------------------
# Nearest-neighbor upscaling layer.

def upscale2d(x, factor=2):
	assert isinstance(factor, int) and factor >= 1
	if factor == 1: return x
	s = x.shape
	x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
	x = tf.tile(x, [1, 1, factor, 1, factor, 1])
	x = tf.reshape(x, [-1, s[1] * factor, s[2] * factor, s[3]])
	return x

#----------------------------------------------------------------------------
# Fused upscale2d + conv2d.
# Faster and uses less memory than performing the operations separately.

def upscale2d_conv2d(x, fmaps, kernel=5, gain=np.sqrt(2), use_wscale=False, scope='us_conv', reuse=False):
	with tf.variable_scope(scope, reuse=reuse):
		assert kernel >= 1 and kernel % 2 == 1
		w = get_weight([kernel, kernel, fmaps, x.shape[-1].value], gain=gain, use_wscale=use_wscale, fan_in=(kernel**2)*x.shape[-1].value)
		w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
		w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
		w = tf.cast(w, x.dtype)
		os = [tf.shape(x)[0], x.shape[1] * 2, x.shape[2] * 2, fmaps]
		return tf.nn.conv2d_transpose(x, w, os, strides=[1,2,2,1], padding='SAME')

#----------------------------------------------------------------------------
# Box filter downscaling layer.

def downscale2d(x, factor=2):
	assert isinstance(factor, int) and factor >= 1
	if factor == 1: return x
	ksize = [1, factor, factor, 1]
	return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID') # NOTE: requires tf_config['graph_options.place_pruned_graph'] = True

#----------------------------------------------------------------------------
# Fused conv2d + downscale2d.
# Faster and uses less memory than performing the operations separately.

def conv2d_downscale2d(x, fmaps, kernel=5, gain=np.sqrt(2), use_wscale=False, scope='conv_ds', reuse=False):
	with tf.variable_scope(scope, reuse=reuse):
		assert kernel >= 1 and kernel % 2 == 1
		w = get_weight([kernel, kernel, x.shape[-1].value, fmaps], gain=gain, use_wscale=use_wscale)
		w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
		w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
		w = tf.cast(w, x.dtype)
		return tf.nn.conv2d(x, w, strides=[1,2,2,1], padding='SAME')

#----------------------------------------------------------------------------
# Pixelwise feature vector normalization.

def pixel_norm(x, epsilon=1e-8):
	return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + epsilon)

		
### Operations
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
					#print '>>> VAR: ' + ln
					#print v.get_shape().as_list()
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

'''
tf image upsampling.
im: shape [b, h, w, c]
'''
def tf_upsample(im):
	im = tf.convert_to_tensor(im, dtype=tf_dtype)
	_, h, w, c = im.get_shape().as_list()
	#us_x = tf.tile(tf.reshape(im, [-1, h*w, 1, c]), [1, 1, 2, 1])
	#us_xy = tf.tile(tf.reshape(us_x, [-1, h, 1, 2*w, c]), [1, 1, 2, 1, 1])
	us_x = tf.pad(tf.reshape(im, [-1, h*w, 1, c]), [[0,0], [0,0], [0,1], [0,0]])
	us_xy = tf.pad(tf.reshape(us_x, [-1, h, 1, 2*w, c]), [[0,0], [0,0], [0,1], [0,0], [0,0]])
	im_us = tf.reshape(us_xy, [-1, 2*h, 2*w, c])
	if h%2 == 1:
		im_us = im_us[:, :-1, :, :]
	if w%2 == 1:
		im_us = im_us[:, :, :-1, :]
	#output = tf_binomial_blur(im_us, kernel=np.array([1., 2., 1.])/2.) if smooth == True else im_us
	#print '>>> tf upsample shape: ', output.get_shape().as_list()
	return im_us

'''
tf image downsampling.
im: shape [b, h, w, c] 
'''
def tf_downsample(im):
	im = tf.convert_to_tensor(im, dtype=tf_dtype)
	output = tf.nn.max_pool(im, ksize=[1, 1, 1, 1], strides=[1, 2, 2, 1], padding='SAME')
	#print '>>> tf downsample shape: ', output.get_shape().as_list()
	return output

'''
tf applies binomial blur to approximate gaussian blurring.
im: shape [b, h, w, c]
'''
def tf_binomial_blur(im, kernel):
	im = tf.convert_to_tensor(im, dtype=tf_dtype)
	kernel = tf.convert_to_tensor(kernel, dtype=tf_dtype)
	c = tf.shape(im)[3]
	ksize = kernel.shape[0]
	kernel_x = tf.tile(tf.reshape(kernel, [1, ksize, 1, 1]), [1, 1, c, 1])
	kernel_y = tf.tile(tf.reshape(kernel, [ksize, 1, 1, 1]), [1, 1, c, 1])
	output = tf.nn.depthwise_conv2d(
		tf.nn.depthwise_conv2d(im, kernel_x, [1, 1, 1, 1], padding='SAME'), 
		kernel_y, [1, 1, 1, 1], padding='SAME')
	return output

'''
tf applies gaussian blur.
im: shape [b, h, w, c]
'''
def tf_gauss_blur(im, sigma, krange=25):
	if sigma == 0:
		return im
	ksize = 2*krange + 1
	t = np.linspace(-krange, krange, ksize)
	kernel = np.exp(0.5 * -t**2/sigma**2)
	kernel = kernel / np.sum(kernel)
	kernel = tf.convert_to_tensor(kernel, dtype=tf_dtype)
	im = tf.convert_to_tensor(im, dtype=tf_dtype)
	c = tf.shape(im)[3]
	kernel_x = tf.tile(tf.reshape(kernel, [1, ksize, 1, 1]), [1, 1, c, 1])
	kernel_y = tf.tile(tf.reshape(kernel, [ksize, 1, 1, 1]), [1, 1, c, 1])
	output = tf.nn.depthwise_conv2d(
		tf.nn.depthwise_conv2d(im, kernel_x, [1, 1, 1, 1], padding='SAME'), 
		kernel_y, [1, 1, 1, 1], padding='SAME')
	return output

'''
Makes a gaussian filter with sigma and ksize+1.
'''
def make_gauss(sigma=1., ksize=40):
	ksize = ksize + 1
	if sigma < 0.001:
		kernel = 0. * np.zeros(ksize)
		kernel[ksize//2] = 1.0
		return kernel
	t = np.linspace(-ksize//2, ksize//2, ksize)
	kernel = np.exp(0.5 * -t**2/sigma**2)
	kernel = kernel / np.sum(kernel)
	return kernel

'''
Makes a windowed sinc filter using Blackman window.
fc: cutoff frequency
ksize: filter sample size
stack: number of times to convolve filter with itself (improves stop band)
'''
def make_winsinc_blackman(fc=1./4, ksize=40, stack=1):
	x = np.arange(ksize+1)
	x[ksize//2] = 0
	kernel = np.sin(2.* np.pi * fc * (x - ksize//2)) / (x - ksize//2)
	kernel = kernel * (0.42 - 0.5*np.cos(2. * np.pi * x / ksize) + 0.08*np.cos(4. * np.pi * x) / ksize)
	kernel[ksize//2] = 2. * np.pi * fc
	kernel = kernel / np.sum(kernel)
	kernel_stack = kernel
	for _ in range(stack):
		kernel_stack = np.convolve(kernel_stack, kernel, 'full')
	return kernel_stack

'''
Shifts im frequencies with fc_x, fc_y and returns cos and sin components.
im: shape [b, h, w, c]
fc: f_center/f_sample which must be in [0, 0.5]
'''
def tf_freq_shift(im, fc_x, fc_y):
	im = tf.convert_to_tensor(im, dtype=tf_dtype)
	b, h, w, c = im.get_shape().as_list()

	kernel_loc = 2.*np.pi*fc_x * np.arange(w).reshape((1, 1, w, 1)) + \
		2.*np.pi*fc_y * np.arange(h).reshape((1, h, 1, 1))
	kernel_cos = np.cos(kernel_loc)
	kernel_sin = np.sin(kernel_loc)
	kernel_cos = tf.convert_to_tensor(kernel_cos, dtype=tf_dtype)
	kernel_sin = tf.convert_to_tensor(kernel_sin, dtype=tf_dtype)
	return im * kernel_cos, im * kernel_sin

'''
Shifts a complex (real, imaginary) input image input using tf_freq_shift.
Returns (real, imaginary) complex image.
'''
def tf_freq_shift_complex(im_r, im_i, fc_x, fc_y):
	g_r_cos, g_r_sin = tf_freq_shift(im_r, fc_x, fc_y)
	g_i_cos, g_i_sin = tf_freq_shift(im_i, fc_x, fc_y)
	return g_r_cos - g_i_sin, g_r_sin + g_i_cos

'''
tf constructs a laplacian pyramid as a list [layer0, layer1, ...].
im: shape [b, h, w, c]
levels: number of layers of the pyramid
'''
def tf_make_lap_pyramid(im, levels=3, freq_shift=False, resize=False):
	im = tf.convert_to_tensor(im, dtype=tf_dtype)
	kernel = np.array([1., 4., 6., 4., 1.]) / 16 #make_winsinc_blackman(fc=1/5.)
	pyramid = list()
	for l in range(levels):
		if l == levels-1:
			pyramid.append(im)
		else:
			im_blur = tf_binomial_blur(im, kernel=kernel)
			im_ds = tf_downsample(im_blur)
			im_us = tf_binomial_blur(tf_upsample(im_ds), kernel=2. * kernel)
			pyramid.append(im - im_us)
			im = im_ds
	
	pyramid_re = pyramid[::-1]
	if freq_shift:
		fc_x = fc_y = 1. / 4
		pyramid_re = [tf_freq_shift(pi, fc_x, fc_y)[0] if i > 0 else pi for i, pi in enumerate(pyramid_re)]
	
	if resize:
		for i, pi in enumerate(pyramid_re):
			pi_ds = pi
			#for _ in range(i):
			#	pi_ds = tf_downsample(tf_binomial_blur(pi_ds))
			pyramid_re[i] = tf_binomial_blur(pi_ds, kernel) if i > 0 else pi_ds
			#pyramid_re[i] = pi_ds

	return pyramid_re

'''
tf reconstruct original images from the given laplacian pyramid.
pyramid: list of [layer0, layer1, ...] of the gaussian pyramid where each layer [b, h, w, c]
'''
def tf_reconst_lap_pyramid(pyramid, freq_shift=False, resize=False):
	pyramid_re = list(pyramid)
	kernel = np.array([1., 4., 6., 4., 1.]) / 16 #make_winsinc_blackman(fc=1./4)
	if resize:
		for i, pi in enumerate(pyramid_re):
			pi_us = pi
			for _ in range(i):
				pi_us  = tf_binomial_blur(tf_upsample(pi_us), kernel=2. * kernel)
			pyramid_re[i] = pi_us

	if freq_shift:
		fc_x = fc_y = 1. / 4
		pyramid_re = [tf_freq_shift(pi, fc_x, fc_y)[0] if i > 0 else pi for i, pi in enumerate(pyramid_re)]

	reconst = pyramid_re[0]
	for im in pyramid_re[1:]:
		im_us = tf_binomial_blur(tf_upsample(reconst), kernel=2. * kernel)
		reconst = im_us + im
	return reconst

'''
splits the im into 4 images.
'''
def tf_split(im):
	im = tf.convert_to_tensor(im, dtype=tf_dtype)
	_, h, w, c = im.get_shape().as_list()
	im_tl = im[:, :h//2, :w//2, :]
	im_tr = im[:, :h//2, w//2:w, :]
	im_bl = im[:, h//2:h, :w//2, :]
	im_br = im[:, h//2:h, w//2:w, :]
	return im_tl, im_tr, im_bl, im_br

'''
reconstructs the full size image by connecting the split.
split: a list of image tensors (b, h, w, c) with the order (tl, tr, bl, br)
'''
def tf_reconst_split(split):
	im_top = tf_join_rows(split[:2])
	im_bottom = tf_join_rows(split[2:4])
	return tf_join_cols([im_top, im_bottom])

def tf_join_rows(split):
	_, h, _, c = split[0].get_shape().as_list()
	w = sum(im.get_shape().as_list()[2] for im in split)
	return tf.reshape(tf.stack(split, axis=2), (-1, h, w, c))

def tf_join_cols(split):
	_, _, w, c = split[0].get_shape().as_list()
	h = sum(im.get_shape().as_list()[1] for im in split)
	return tf.reshape(tf.concat(split, axis=1), (-1, h, w, c))

### shifts layer and concats real and imaginary parts, then downsamples to out_size
def fs_layer(layer, fc_x, fc_y, out_size):
	blur_kernel = np.array([1., 4., 6., 4., 1.]) / 16.
	layer_size = layer.get_shape().as_list()[1]
	layer_cos, layer_sin = tf_freq_shift(layer, fc_x, fc_y)
	out = tf.concat([layer_cos, layer_sin], 3)
	for i in range(int(np.log2(layer_size//out_size))):
		out = tf_downsample(tf_binomial_blur(out, blur_kernel))
	#print('>>> FS_Layer shape: {}'.format(out.get_shape().as_list()))
	return out

def build_gen_v1_branch(data_dim, zi, act, train_phase, im_size, sub_scope='or'):
	train_phase = True
	ol = list()
	with tf.variable_scope('g_net'):
		with tf.variable_scope(sub_scope):
			bn = tf.contrib.layers.batch_norm
			h2_us = tf.image.resize_nearest_neighbor(zi, [im_size//2, im_size//2], name='us2')
			h3 = act(bn(conv2d(h2_us, 64, scope='conv2'), is_training=train_phase))
			h4 = conv2d(h3, 2*data_dim[-1], k_h=1, k_w=1, scope='conv3')
			o = tf.tanh(h4)
			o_us = tf.image.resize_nearest_neighbor(o, [im_size, im_size], name='us3')
	return h3, h4, o_us

def build_gen_v1(data_dim, zi, act, train_phase, im_size, sub_scope='or', reuse=None, layer_ins=None):
	#train_phase = True
	layer_ins = [None, None, None, None] if layer_ins is None else layer_ins
	ol = list()
	with tf.variable_scope('g_net', reuse=reuse):
		with tf.variable_scope(sub_scope):
			bn = tf.contrib.layers.batch_norm
			### setup based on size
			if im_size == 32:
				z_fc = act(bn(dense(zi, 4*4*256, scope='fcz'),
					is_training=train_phase))
				h0 = tf.reshape(z_fc, [-1, 4, 4, 256])
				h1 = h0
			elif im_size == 64:
				z_fc = act(bn(dense(zi, 4*4*512, scope='fcz'),
					is_training=train_phase))
				h0 = tf.reshape(z_fc, [-1, 4, 4, 512])
				#h1 = h0
				h0_us = tf.image.resize_nearest_neighbor(h0, [8, 8], name='us0')
				h1 = act(bn(conv2d(h0_us, 256, scope='conv0'), 
					is_training=train_phase))
			elif im_size == 128:
				z_fc = act(bn(dense(zi, 8*8*512, scope='fcz'),
					is_training=train_phase))
				h0 = tf.reshape(z_fc, [-1, 8, 8, 512])
				h0_us = tf.image.resize_nearest_neighbor(h0, [16, 16], name='us0')
				h0_us = h0_us if layer_ins[0] is None else layer_ins[0]
				h1 = act(bn(conv2d(h0_us, 256, scope='conv0'), 
					is_training=train_phase))
			else:
				raise ValueError('{} for generator im_size is not defined!'.format(im_size))

			### us version: decoding fc code with upsampling and conv hidden layers
			h1_us = tf.image.resize_nearest_neighbor(h1, [im_size//4, im_size//4], name='us1')
			h1_us = h1_us if layer_ins[1] is None else layer_ins[1]
			h2 = act(bn(conv2d(h1_us, 128, scope='conv1'),
				is_training=train_phase))

			h2_us = tf.image.resize_nearest_neighbor(h2, [im_size//2, im_size//2], name='us2')
			h2_us = h2_us if layer_ins[2] is None else layer_ins[2]
			h3 = act(bn(conv2d(h2_us, 64, scope='conv2'),
				is_training=train_phase))
		
			h3_us = tf.image.resize_nearest_neighbor(h3, [im_size, im_size], name='us3')
			h3_us = h3_us if layer_ins[3] is None else layer_ins[3]
			h4 = conv2d(h3_us, data_dim[-1], scope='conv3')

			### resnext version
			'''
			btnk_dim = 64
			h2 = resnext(h1, 128, btnk_dim, 'res1', train_phase, 
						op_type='up', bn=False, act=act)
			h3 = resnext(h2, 64, btnk_dim//2, 'res2', train_phase,
						op_type='up', bn=False, act=act)
			h4 = resnext(h3, 32, btnk_dim//4, 'res3', train_phase, 
						op_type='up', bn=False, act=act)
			h5 = conv2d(h4, data_dim[-1], scope='convo')
			'''
			o = tf.tanh(h4)
		return h0, h1, h2, h3, h4, o

def build_dis_v1(data_layer, train_phase, im_size, sub_scope='or', reuse=False):
	act = lrelu
	with tf.variable_scope('d_net'):
		with tf.variable_scope(sub_scope):
			bn = tf.contrib.layers.batch_norm

			### encoding the 28*28*3 image with conv into 3*3*256
			h0 = act(conv2d(data_layer, 64, d_h=2, d_w=2, scope='conv0', reuse=reuse))
			h1 = act(conv2d(h0, 128, d_h=2, d_w=2, scope='conv1', reuse=reuse))
			h2 = act(conv2d(h1, 256, d_h=2, d_w=2, scope='conv2', reuse=reuse))
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

			### im_size setup
			if im_size == 32:
				h3 = h2
			elif im_size == 64:
				h3 = act(conv2d(h2, 512, d_h=2, d_w=2, scope='conv3', reuse=reuse))
			elif im_size == 128:
				h3 = act(conv2d(h2, 512, d_h=2, d_w=2, scope='conv3', reuse=reuse))
			else:
				raise ValueError('{} for discriminator im_size is not defined!'.format(im_size))

			### fully connected discriminator
			flat = tf.contrib.layers.flatten(h3)
			o = dense(flat, 1, scope='fco', reuse=reuse)
			return o, flat

def build_gen_v2(data_dim, zi, act, train_phase, im_size, sub_scope='or'):
	act = lrelu
	train_phase = True
	use_pixelnorm = True
	use_batchnorm = False
	use_wscale = True
	with tf.variable_scope('g_net'):
		with tf.variable_scope(sub_scope):
			def pn(x): 
				return pixel_norm(x, epsilon=1e-8) if use_pixelnorm else x
			def bn(x, **kwargs): 
				return tf.contrib.layers.batch_norm(x, is_training=kwargs['is_training']) if use_batchnorm else x

			### setup based on size
			if im_size == 32:
				z_fc = pn(act(apply_bias(dense_ws(zi, 4*4*256, use_wscale=use_wscale, scope='fcz'), scope='fcz')))
				h0 = tf.reshape(z_fc, [-1, 4, 4, 256])
				h1 = h0
			elif im_size == 64:
				z_fc = pn(act(apply_bias(dense_ws(zi, 4*4*512, use_wscale=use_wscale, scope='fcz'), scope='fcz')))
				h0 = tf.reshape(z_fc, [-1, 4, 4, 512])
				#h1 = h0
				h1 = pn(act(apply_bias(upscale2d_conv2d(h0, 256, use_wscale=use_wscale, scope='conv0'), scope='conv0')))
			elif im_size == 128:
				z_fc = pn(act(apply_bias(dense_ws(zi, 8*8*512, use_wscale=use_wscale, scope='fcz'), scope='fcz')))
				h0 = tf.reshape(z_fc, [-1, 8, 8, 512])
				h1 = pn(act(apply_bias(upscale2d_conv2d(h0, 256, use_wscale=use_wscale, scope='conv0'), scope='conv0')))
			else:
				raise ValueError('{} for generator im_size is not defined!'.format(im_size))

			### us version: decoding fc code with upsampling and conv hidden layers
			h2 = pn(act(apply_bias(upscale2d_conv2d(h1, 128, use_wscale=use_wscale, scope='conv1'), scope='conv1')))
			h3 = pn(act(apply_bias(upscale2d_conv2d(h2, 64, use_wscale=use_wscale, scope='conv2'), scope='conv2')))
			h4 = apply_bias(upscale2d_conv2d(h3, data_dim[-1], use_wscale=use_wscale, scope='conv3'), scope='conv3')
			o = tf.tanh(h4)
		return h0, h1, h2, h3, o

def build_dis_v2(data_layer, train_phase, im_size, sub_scope='or', reuse=False):
	act = lrelu
	use_wscale = True
	with tf.variable_scope('d_net', reuse=reuse):
		with tf.variable_scope(sub_scope):

			### encoding the image with conv and downsampling
			h0 = act(apply_bias(conv2d_downscale2d(data_layer, 64, use_wscale=use_wscale, scope='conv0'), scope='conv0'))
			h1 = act(apply_bias(conv2d_downscale2d(h0, 128, use_wscale=use_wscale, scope='conv1'), scope='conv1'))
			h2 = act(apply_bias(conv2d_downscale2d(h1, 256, use_wscale=use_wscale, scope='conv2'), scope='conv2'))

			### im_size setup
			if im_size == 32:
				h3 = h2
			elif im_size == 64:
				h3 = act(apply_bias(conv2d_downscale2d(h2, 512, use_wscale=use_wscale, scope='conv3'), scope='conv3'))
			elif im_size == 128:
				h3 = act(apply_bias(conv2d_downscale2d(h2, 512, use_wscale=use_wscale, scope='conv3'), scope='conv3'))
			else:
				raise ValueError('{} for discriminator im_size is not defined!'.format(im_size))

			### fully connected discriminator
			flat = tf.reshape(h3, [-1, np.prod(h3.get_shape().as_list()[1:])])
			o = dense_ws(h3, 1, scope='fco')
			return o, flat

### GAN Class definition
class Ganist:
	def __init__(self, sess, log_dir='logs'):
		### run parameters
		self.log_dir = log_dir
		self.sess = sess
		#self.device_names = ['/device:GPU:0', '/device:GPU:0', '/device:GPU:0']

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
		self.build_gen = build_gen_v1
		self.build_dis = build_dis_v1

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

	def build_shift_gan(self, im_input, zi_input, im_size, gen_size, 
			scope='0', gen_collect=None, im_collect=None, comb_list=None, 
			d_loss_list=None, g_loss_list=None, rg_grad_norm_list=None):
		gen_collect = list() if gen_collect is None else gen_collect
		im_collect = list() if im_collect is None else im_collect
		comb_list = list() if comb_list is None else comb_list
		d_loss_list = list() if d_loss_list is None else d_loss_list
		g_loss_list = list() if g_loss_list is None else g_loss_list
		rg_grad_norm_list = list() if rg_grad_norm_list is None else rg_grad_norm_list
		#freq_list = [(0.25, 0.), (0., 0.25), (0.25, 0.25), (-0.25, 0.25)]
		freq_list = [(1/16., 0.), (0., 1/16.), (1/16., 1/16.), (-1/16., 1/16.)]
		#freq_list = [(1./8, 0.), (0., 1./8), (1./8, 1./8), (-1./8, 1./8),
		#			(1./16, 0.), (0., 1./16), (1./16, 1./16), (-1./16, 1./16)]
		#freq_list = [(0.25, 0.25)]
		blur_kernel = np.array([1., 4., 6., 4., 1.]) / 16. #make_winsinc_blackman(1./4, ksize=30)
		#blur_kernel = np.array([1., 6., 15., 20., 15., 6., 1.])
		#blur_kernel /= np.sum(blur_kernel)
		
		### shift the image
		im_fs_list = list()
		for i, fc in enumerate(freq_list):
			im_r, im_i = tf_freq_shift(im_input, -fc[0], -fc[1])
			im_fs_list += [im_r, im_i]

		### low pass the shifted image
		im_lp_list = [tf_downsample(tf_binomial_blur(
			tf_downsample(tf_binomial_blur(im, blur_kernel)), blur_kernel)) for im in im_fs_list]
		im_input_ds = tf_downsample(tf_binomial_blur(
			tf_downsample(tf_binomial_blur(im_input, blur_kernel)), blur_kernel))

		### reconstruct the shifted image
		im_rec_list = list()
		for i, fc in enumerate(freq_list):
			im_fs_r = tf_binomial_blur(tf_upsample(
				tf_binomial_blur(tf_upsample(im_lp_list[i*2]), 2.*blur_kernel)), 2.*blur_kernel)
			im_fs_i = tf_binomial_blur(tf_upsample(
				tf_binomial_blur(tf_upsample(im_lp_list[i*2+1]), 2.*blur_kernel)), 2.*blur_kernel)
			im_r, _ = tf_freq_shift_complex(im_fs_r, im_fs_i, fc[0], fc[1])
			im_rec_list.append(im_r)

		im_rec_list.append(tf_binomial_blur(tf_upsample(
				tf_binomial_blur(tf_upsample(im_input_ds), 2.*blur_kernel)), 2.*blur_kernel))
		im_collect += im_lp_list + [im_input_ds] + im_rec_list + [sum(im_rec_list)]

		### build generators recursively
		g_layer_list = list()
		#if gen_size < im_size:
		#	for i in range(2*len(freq_list)):
		#		build_shift_gan(tf_downsample(im_lp_list[i]), zi_input, im_size//2, gen_size,
		#			'{}_{}'.format(scope, i), gen_collect, comb_list, 
		#			d_loss_list, g_loss_list, rg_grad_norm_list)
		#		g_layer_list.append(tf_binomial_blur(tf_upsample(comb_list[-1]), 2.*blur_kernel))
		#else:
		for i in range(2*len(freq_list)):
			g_layer = self.build_gen(self.data_dim, zi_input, self.g_act, self.train_phase, 
				im_size=gen_size, sub_scope='level_{}_{}'.format(scope, i))[-1]
			g_layer_list.append(g_layer)
			gen_collect.append(g_layer)
			#gen_collect.append(g_layer[:, :, :, 0:3])
			#gen_collect.append(g_layer[:, :, :, 3:6])

		### apply upsample and low pass filter on g_layer to bring it to im_size
		g_lp_list = list()
		for gl in g_layer_list:
			g_us = gl
			for i in range(int(np.log2(im_size//gen_size))):
				g_us = tf_binomial_blur(tf_upsample(g_us), 2.*blur_kernel)
			g_lp_list.append(g_us)

		### build delta generator (must use im_size in recursive case)
		g_delta = self.build_gen(self.data_dim, zi_input, self.g_act, self.train_phase, 
			im_size=im_size, sub_scope='level_{}_delta'.format(scope))[-1]#[:, :, :, 0:3]
		gen_collect.append(g_delta)
		g_delta_lp = g_delta #tf_binomial_blur(tf_upsample(g_delta), 2.*blur_kernel)
		g_delta_ds = tf_downsample(tf_binomial_blur(
			tf_downsample(tf_binomial_blur(g_delta, blur_kernel)), blur_kernel))
		
		#g_delta_fill = self.build_gen(self.data_dim, zi_input, self.g_act, self.train_phase, 
		#	im_size=im_size, sub_scope='level_{}_delta_fill'.format(scope))[-1]#[:, :, :, 0:3]
		#gen_collect.append(g_delta_fill)
		#g_delta_hp, _ = tf_freq_shift(g_delta_fill, 0.5, 0.5)

		### build output combination by shifting
		g_layer_fs_list = g_lp_list #list()
		#for i, fc in enumerate(freq_list):
		#	g_fs_r, _ = tf_freq_shift_complex(g_lp_list[i*2], g_lp_list[i*2+1], fc[0], fc[1])
		#	#g_fs_r, _ = tf_freq_shift_complex(
		#	#	g_lp_list[i][:, :, :, 0:3], g_lp_list[i][:, :, :, 3:6], fc[0], fc[1])
		#	g_layer_fs_list.append(g_fs_r)

		g_layer_fs_list.append(g_delta_lp)
		#g_layer_fs_list.append(g_delta_hp)
		g_comb = sum(g_layer_fs_list)
		comb_list.append(g_comb)

		### build band pass discriminators
		#for i, gl in enumerate(g_layer_list):
		#	g_layer_ds = tf_downsample(tf_binomial_blur(gl, blur_kernel))
		#	d_loss, g_loss, rg_grad_norm_output = \
		#		self.build_gan_loss(im_lp_list[i], g_layer_ds, gen_size//2, scope='level_{}_{}'.format(scope, i))
		#	d_loss_list.append(d_loss)
		#	g_loss_list.append(g_loss)
		#	rg_grad_norm_list.append(rg_grad_norm_output)

		### build freq shifted discriminators
		#for i, fc in enumerate(freq_list):
		#	disc_size = gen_size
		#	im_fs = fs_layer(im_input, -fc[0], -fc[1], disc_size)
		#	g_fs = fs_layer(g_comb, -fc[0], -fc[1], disc_size)
		#	d_loss, g_loss, rg_grad_norm_output = \
		#		self.build_gan_loss(im_fs, g_fs, disc_size, scope='level_{}_{}'.format(scope, i))
		#	d_loss_list.append(d_loss)
		#	g_loss_list.append(g_loss)
		#	rg_grad_norm_list.append(rg_grad_norm_output)

		### build aligning discriminator
		d_loss, g_loss, rg_grad_norm_output = \
			self.build_gan_loss(im_input, g_comb, im_size, scope='level_{}_comb'.format(scope))
		d_loss_list.append(d_loss)
		g_loss_list.append(g_loss)
		rg_grad_norm_list.append(rg_grad_norm_output)

		### build low pass discriminator
		#d_loss, g_loss, rg_grad_norm_output = \
		#	self.build_gan_loss(im_input_ds, g_delta_ds, im_size//4, scope='level_{}_delta'.format(scope))
		#d_loss_list.append(d_loss)
		#g_loss_list.append(g_loss)
		#rg_grad_norm_list.append(rg_grad_norm_output)
		
		### return collected operators lists
		#gen_collect += g_layer_fs_list# + [g_delta_ds]
		gen_collect = g_layer_fs_list
		return gen_collect, im_collect, comb_list, d_loss_list, g_loss_list, rg_grad_norm_list

	def build_fsg_gan(self, im_input, zi_input, im_size, 
			scope='0', gen_collect=None, im_collect=None, comb_list=None, 
			d_loss_list=None, g_loss_list=None, rg_grad_norm_list=None):
		
		gen_collect = list() if gen_collect is None else gen_collect
		im_collect = list() if im_collect is None else im_collect
		comb_list = list() if comb_list is None else comb_list
		d_loss_list = list() if d_loss_list is None else d_loss_list
		g_loss_list = list() if g_loss_list is None else g_loss_list
		rg_grad_norm_list = list() if rg_grad_norm_list is None else rg_grad_norm_list
		freq_list = [(1/16., 0.), (0., 1/16.), (1/16., 1/16.), (-1/16., 1/16.)]
		im_collect = [im_input]

		### build delta generator (must use im_size in recursive case)
		g_feats = self.build_gen(self.data_dim, zi_input, self.g_act, self.train_phase, 
			im_size=im_size, sub_scope='level_{}_delta'.format(scope))
		g_delta = g_feats[-1][:, :, :, 0:3]

		### build generators recursively
		g_layer_list = list()
		for i in range(len(freq_list)):
			g_layer = build_gen_v1_branch(self.data_dim, g_feats[2], self.g_act, self.train_phase, 
				im_size=im_size, sub_scope='level_{}_{}'.format(scope, i))[-1]
			g_layer_list.append(g_layer)

		### build output combination by shifting
		g_layer_fs_list = list()
		for i, fc in enumerate(freq_list):
			g_fs_r, _ = tf_freq_shift_complex(g_layer_list[i][:, :, :, 0:3], g_layer_list[i][:, :, :, 3:6], fc[0], fc[1])
			g_layer_fs_list.append(g_fs_r)

		g_layer_fs_list.append(g_delta)
		g_comb = sum(g_layer_fs_list)
		comb_list.append(g_comb)

		### build aligning discriminator
		d_loss, g_loss, rg_grad_norm_output = \
			self.build_gan_loss(im_input, g_comb, im_size, scope='level_{}_comb'.format(scope))
		d_loss_list.append(d_loss)
		g_loss_list.append(g_loss)
		rg_grad_norm_list.append(rg_grad_norm_output)
		
		### return collected operators lists
		gen_collect = g_layer_fs_list
		return gen_collect, im_collect, comb_list, d_loss_list, g_loss_list, rg_grad_norm_list

	def build_gan_loss(self, im_layer, g_layer, im_size, scope):
		### build logits (discriminator)
		r_logits, g_logits, rg_logits, rg_layer = \
			self.build_dis_logits(im_layer, g_layer, 
				self.train_phase, im_size, sub_scope=scope)
		
		### d loss
		d_loss, rg_grad_norm_output = \
			self.build_dis_loss(r_logits, g_logits, rg_logits, rg_layer)
		### g loss
		g_loss = self.build_gen_loss(g_logits)

		return d_loss, g_loss, rg_grad_norm_output

	def build_wgan_fs(self, im_input, zi_input, im_size):
		### generators
		g_layer = self.build_gen(self.data_dim, zi_input, self.g_act, self.train_phase, 
				im_size=im_size, sub_scope='main')[-1]
		g_layer2 = self.build_gen(self.data_dim, zi_input, self.g_act, self.train_phase, 
				im_size=im_size, sub_scope='comp')[-1]
		g_layer2, _ = tf_freq_shift(g_layer2, 0.5, 0.5)

		gen_collect = [g_layer, g_layer2]
		im_collect = [im_input]
		comb_list = [g_layer+g_layer2]

		### discriminators
		d_loss, g_loss, rg_grad_norm_output = \
				self.build_gan_loss(im_input, comb_list[-1], im_size=im_size, scope='main')
		d_loss_list = [d_loss]
		g_loss_list = [g_loss]
		rg_grad_norm_list = [rg_grad_norm_output]
		
		im_fs = fs_layer(im_input, -0.5, -0.5, im_size)
		g_fs = fs_layer(comb_list[-1], -0.5, -0.5, im_size)
		d_loss, g_loss, rg_grad_norm_output = \
				self.build_gan_loss(im_fs, g_fs, im_size=im_size, scope='comp')
		d_loss_list.append(d_loss)
		g_loss_list.append(g_loss)
		rg_grad_norm_list.append(rg_grad_norm_output)

		return gen_collect, im_collect, comb_list, d_loss_list, g_loss_list, rg_grad_norm_list

	def build_wgan(self, im_input, zi_input, im_size):
		### generators
		g_layer = self.build_gen(self.data_dim, zi_input, self.g_act, self.train_phase, 
				im_size=im_size, sub_scope='l2')[-1] ### subscope 'l2' for older wganbn, 'main' for more recent
		gen_collect = [g_layer]
		im_collect = [im_input]
		comb_list = [g_layer]

		### discriminators
		d_loss, g_loss, rg_grad_norm_output = \
				self.build_gan_loss(im_input, g_layer, im_size=im_size, scope='l2') ### subscope 'l2' for older wganbn, 'main' for more recent
		d_loss_list = [d_loss]
		g_loss_list = [g_loss]
		rg_grad_norm_list = [rg_grad_norm_output]

		return gen_collect, im_collect, comb_list, d_loss_list, g_loss_list, rg_grad_norm_list

	def build_wgan_with_separate_layer_inputs(self, im_input, zi_input, im_size):
		### generators
		g_layer = self.build_gen(self.data_dim, zi_input, self.g_act, self.train_phase, 
				im_size=im_size, sub_scope='l2')[-1] ### subscope 'l2' for older wganbn, 'main' for more recent
		gen_collect = [g_layer]
		im_collect = [im_input]
		comb_list = [g_layer]
		
		temp_in = np.zeros((1, 16, 16, 512))
		temp_in[:, 8, 8, :] = 1
		temp_in = tf.convert_to_tensor(temp_in, dtype=tf_dtype)
		gen_collect.append(self.build_gen(self.data_dim, zi_input, self.g_act, self.train_phase, 
				im_size=im_size, sub_scope='l2', reuse=True, layer_ins=[temp_in, None, None, None])[-2])

		temp_in = np.zeros((1, 32, 32, 256))
		temp_in[:, 16, 16, :] = 1
		temp_in = tf.convert_to_tensor(temp_in, dtype=tf_dtype)
		gen_collect.append(self.build_gen(self.data_dim, zi_input, self.g_act, self.train_phase, 
				im_size=im_size, sub_scope='l2', reuse=True, layer_ins=[None, temp_in, None, None])[-2])

		temp_in = np.zeros((1, 64, 64, 128))
		temp_in[:, 32, 32, :] = 1
		temp_in = tf.convert_to_tensor(temp_in, dtype=tf_dtype)
		gen_collect.append(self.build_gen(self.data_dim, zi_input, self.g_act, self.train_phase, 
				im_size=im_size, sub_scope='l2', reuse=True, layer_ins=[None, None, temp_in, None])[-2])

		temp_in = np.zeros((1, 128, 128, 64))
		temp_in[:, 64, 64, :] = 1
		temp_in = tf.convert_to_tensor(temp_in, dtype=tf_dtype)
		gen_collect.append(self.build_gen(self.data_dim, zi_input, self.g_act, self.train_phase, 
				im_size=im_size, sub_scope='l2', reuse=True, layer_ins=[None, None, None, temp_in])[-2])


		### discriminators
		d_loss, g_loss, rg_grad_norm_output = \
				self.build_gan_loss(im_input, g_layer, im_size=im_size, scope='l2') ### subscope 'l2' for older wganbn, 'main' for more recent
		d_loss_list = [d_loss]
		g_loss_list = [g_loss]
		rg_grad_norm_list = [rg_grad_norm_output]

		return gen_collect, im_collect, comb_list, d_loss_list, g_loss_list, rg_grad_norm_list


	def build_wgan_gshift(self, im_input, zi_input, im_size):
		### generators
		g_layer = self.build_gen(self.data_dim, zi_input, self.g_act, self.train_phase, 
				im_size=im_size, sub_scope='l2')[-1] ### subscope 'l2' for older wganbn, 'main' for more recent
		g_layer, _ = tf_freq_shift(g_layer, 0.5, 0.5)
		gen_collect = [g_layer]
		im_collect = [im_input]
		comb_list = [g_layer]

		### discriminators
		d_loss, g_loss, rg_grad_norm_output = \
				self.build_gan_loss(im_input, g_layer, im_size=im_size, scope='l2') ### subscope 'l2' for older wganbn, 'main' for more recent
		d_loss_list = [d_loss]
		g_loss_list = [g_loss]
		rg_grad_norm_list = [rg_grad_norm_output]

		return gen_collect, im_collect, comb_list, d_loss_list, g_loss_list, rg_grad_norm_list

	def build_graph(self):
		with tf.name_scope('ganist'):
			### define placeholders for image and label inputs **g_num** **mt**
			self.im_input = tf.placeholder(tf_dtype, [None]+self.data_dim, name='im_input')
			self.zi_input = tf.placeholder(tf_dtype, [None, self.z_dim], name='zi_input')
			self.train_phase = tf.placeholder(tf.bool, name='phase')

			### apply regular wgan
			#self.gen_collect, self.im_collect, self.comb_list,\
			#self.d_loss_list, self.g_loss_list, self.rg_grad_norm_list = \
			#	self.build_wgan_with_separate_layer_inputs(self.im_input, self.zi_input, 
			#		im_size=self.data_dim[0])
			#self.im_input_rec = self.im_collect[-1]

			### apply freq shift gan
			self.gen_collect, self.im_collect, self.comb_list,\
			self.d_loss_list, self.g_loss_list, self.rg_grad_norm_list = \
				self.build_shift_gan(self.im_input, self.zi_input, 
					im_size=self.data_dim[0], gen_size=64)
			self.im_input_rec = self.im_collect[-1]

			### apply pyramid for real images
			#self.im_input_l0, self.im_input_l1, self.im_input_l2 = \
			#	tf_make_lap_pyramid(self.im_input, levels=3)
			#self.im_input_rec = tf_reconst_lap_pyramid(
			#	[self.im_input_l0, self.im_input_l1, self.im_input_l2])

			### apply split for real images
			#self.im_input_l0, self.im_input_l1, self.im_input_l2, self.im_input_l3 = \
			#	tf_split(self.im_input)
			#self.im_input_rec = tf_reconst_split(
			#	[self.im_input_l0, self.im_input_l1, self.im_input_l2, self.im_input_l3])			
			
			### reconstruct from misaligned pyramids
			#self.im_input_rec_mal1 = tf_reconst_lap_pyramid(
			#	[self.im_input_l0, tf.reverse(self.im_input_l1, [0]), self.im_input_l2])
			#self.im_input_rec_mal2 = tf_reconst_lap_pyramid(
			#	[self.im_input_l0, self.im_input_l1, tf.reverse(self.im_input_l2, [0])])

			### build generators at each pyramid level
			#self.g_layer_l0 = self.build_gen(self.zi_input, self.g_act, self.train_phase, 
			#	im_size=32, sub_scope='l0')[-1]
			#self.g_layer_l1 = self.build_gen(self.zi_input, self.g_act, self.train_phase, 
			#	im_size=64, sub_scope='l1')[-1]
			#self.g_layer_l2 = self.build_gen(self.zi_input, self.g_act, self.train_phase, 
			#	im_size=128, sub_scope='l2')[-1]
			##self.g_layer_l3 = self.build_gen(self.zi_input, self.g_act, self.train_phase, 
			##	im_size=64, sub_scope='l3')[-1]
			#self.g_vars_l0 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'g_net/l0')
			#self.g_vars_l1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'g_net/l1')
			#self.g_vars_l2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'g_net/l2')
			##self.g_vars_l3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'g_net/l3')

			### reconst from generated pyramid
			#self.g_layer_rec = tf_reconst_lap_pyramid(
			#	[self.g_layer_l0, self.g_layer_l1, self.g_layer_l2])

			### reconst from generated split
			#self.g_layer_rec = tf_reconst_split(
			#	[self.g_layer_l0, self.g_layer_l1, self.g_layer_l2, self.g_layer_l3])

			### collect output samples
			#self.gen_collect = [self.g_layer_l0, self.g_layer_l1, self.g_layer_l2]
			#self.im_collect = [self.im_input_l0, self.im_input_l1, self.im_input_l2]

			### collect g vars
			#self.g_vars = self.g_vars_l2 #self.g_vars_l0 + self.g_vars_l1 + self.g_vars_l2 + self.g_vars_l3
			self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'g_net')

			### build discriminator pyramid l0
			### build model
			#d_loss_l0, g_loss_l0, rg_grad_norm_output_l0 = \
			#	self.build_gan_loss(self.im_input_l0, self.g_layer_l0, im_size=32, scope='l0')
			#self.d_vars_l0 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'd_net/l0')

			### build discriminator pyramid l1
			#d_loss_l1, g_loss_l1, rg_grad_norm_output_l1 = \
			#	self.build_gan_loss(self.im_input_l1, self.g_layer_l1, im_size=64, scope='l1')
			#self.d_vars_l1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'd_net/l1')

			### build discriminator pyramid l2
			#d_loss_l2, g_loss_l2, rg_grad_norm_output_l2 = \
			#	self.build_gan_loss(self.im_input_l2, self.g_layer_l2, im_size=128, scope='l2')
			#self.d_vars_l2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'd_net/l2')

			### build discriminator pyramid l3
			#d_loss_l3, g_loss_l3, rg_grad_norm_output_l3 = \
			#	self.build_gan_loss(self.im_input_l3, self.g_layer_l3, im_size=128, scope='l3')
			#self.d_vars_l3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'd_net/l3')

			### build discriminator pyramid reconst
			#d_loss_rec, g_loss_rec, rg_grad_norm_output_rec = \
			#	self.build_gan_loss(self.im_input, self.g_layer_rec, im_size=128, scope='rec')
			#self.d_vars_rec = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'd_net/rec')

			### build discriminator pyramid reconst on misaligned l1 real images
			#_, self.g_logits_rec_mal1, self.rg_logits_rec_mal1, self.rg_layer_rec_mal1 = \
			#	self.build_dis_logits(self.im_input, self.im_input_rec_mal1,
			#		self.train_phase, im_size=128, sub_scope='rec', reuse=True)
			### d loss rec mal1
			#d_loss_rec_mal1, rg_grad_norm_output_rec_mal1 = \
			#	self.build_dis_loss(self.r_logits_rec, self.g_logits_rec_mal1, 
			#		self.rg_logits_rec_mal1, self.rg_layer_rec_mal1)

			### build discriminator pyramid reconst on misaligned l2 real images
			#_, self.g_logits_rec_mal2, self.rg_logits_rec_mal2, self.rg_layer_rec_mal2 = \
			#	self.build_dis_logits(self.im_input, self.im_input_rec_mal2,
			#		self.train_phase, im_size=128, sub_scope='rec', reuse=True)
			### d loss rec mal2
			#d_loss_rec_mal2, rg_grad_norm_output_rec_mal2 = \
			#	self.build_dis_loss(self.r_logits_rec, self.g_logits_rec_mal2, 
			#		self.rg_logits_rec_mal2, self.rg_layer_rec_mal2)

			### collect d vars
			#self.d_vars = self.d_vars_l2 #self.d_vars_l0 + self.d_vars_l1 + self.d_vars_l2 + self.d_vars_l3 + \
				#self.d_vars_rec
			self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'd_net')

			### collect d loss
			#self.d_loss_list = [d_loss_l0, d_loss_l1, d_loss_l2, d_loss_rec]
			#self.rg_grad_norm_list = [rg_grad_norm_output_l0, rg_grad_norm_output_l1, rg_grad_norm_output_l2, rg_grad_norm_output_rec]
			self.d_loss_total = sum(self.d_loss_list)
			self.rg_grad_norm_output = sum(self.rg_grad_norm_list) / len(self.rg_grad_norm_list)
			#self.rg_grad_norm_output = (rg_grad_norm_output_l0 + rg_grad_norm_output_l1 + \
			#	rg_grad_norm_output_l2 + rg_grad_norm_output_l2 + rg_grad_norm_output_rec) / 5.# + \
				#rg_grad_norm_output_rec_mal1 + rg_grad_norm_output_rec_mal2) / 6.

			### d opt total
			self.d_opt_handle = \
				tf.train.AdamOptimizer(self.d_lr, beta1=self.d_beta1, beta2=self.d_beta2)
			d_grads_vars = self.d_opt_handle.compute_gradients(self.d_loss_total, self.d_vars)
			self.d_opt_total = self.d_opt_handle.apply_gradients(d_grads_vars)
			
			### collect g loss *L2LOSS
			#self.g_loss_list = [g_loss_l0, g_loss_l1, g_loss_l2, g_loss_rec]
			self.g_loss_total = sum(self.g_loss_list)

			### L2 Loss for generator *L2LOSS
			#self.g_loss_total = tf.reduce_mean(
			#	tf.reduce_sum(tf.square(self.im_input - self.comb_list[-1]), axis=[1, 2, 3]))

			### g opt total
			self.g_opt_handle = tf.train.AdamOptimizer(self.g_lr, beta1=self.g_beta1, beta2=self.g_beta2)
			g_grads_vars = self.g_opt_handle.compute_gradients(self.g_loss_total, self.g_vars)
			self.g_opt_total = self.g_opt_handle.apply_gradients(g_grads_vars)

			### bn variables
			self.bn_moving_vars = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) \
				if 'BatchNorm' in v.name and \
				('moving_mean' in v.name or 'moving_variance' in  v.name)]
			print('>>> bn vars:', self.bn_moving_vars)

			### collect opt
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				self.d_opt = tf.group(self.d_opt_total)
				self.g_opt = tf.group(self.g_opt_total)

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
			for v in self.g_vars:
				self.g_vars_count += int(np.prod(v.get_shape()))
			for v in self.d_vars:
				self.d_vars_count += int(np.prod(v.get_shape()))

			### compute conv layer learning variation
			self.g_sim_layer_list = ['conv0', 'conv1', 'conv2', 'conv3', 'fcz']
			self.g_backup, self.g_layer_sim, self.g_layer_non_zero = \
				compute_layer_sim(self.g_sim_layer_list, self.g_vars)

			self.d_sim_layer_list = ['conv0', 'conv1', 'conv2', 'conv3', 'fco']
			self.d_backup, self.d_layer_sim, self.d_layer_non_zero  = \
				compute_layer_sim(self.d_sim_layer_list, self.d_vars)

			### summaries **g_num**
			g_loss_sum = tf.summary.scalar("g_loss", self.g_loss_total)
			d_loss_sum = tf.summary.scalar("d_loss", self.d_loss_total)
			self.summary = tf.summary.merge([g_loss_sum, d_loss_sum])

	def build_dis_logits(self, im_data, g_data, train_phase, im_size, sub_scope='or', reuse=False):
		int_rand = tf.random_uniform([tf.shape(im_data)[0]], minval=0.0, maxval=1.0, dtype=tf_dtype)
		int_rand = tf.reshape(int_rand, [-1, 1, 1, 1])
		### build discriminator for low pass
		r_logits, r_hidden = self.build_dis(im_data, train_phase, im_size, sub_scope=sub_scope, reuse=reuse)
		g_logits, g_hidden = self.build_dis(g_data, train_phase, im_size, sub_scope=sub_scope, reuse=True)
		### real gen manifold interpolation
		rg_layer = (1.0 - int_rand) * g_data + int_rand * im_data
		rg_logits, _ = self.build_dis(rg_layer, train_phase, im_size, sub_scope=sub_scope, reuse=True)
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
		self.saver_var_only = tf.train.Saver(self.g_vars+self.bn_moving_vars)#+self.d_vars+self.bn_moving_vars)#+self.e_vars+[self.pg_var, self.pg_q])
		self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

	def save(self, fname):
		self.saver.save(self.sess, fname)

	def load(self, fname):
		self.saver_var_only.restore(self.sess, fname)

	def write_sum(self, sum_str, counter):
		self.writer.add_summary(sum_str, counter)

	def get_vars_array(self):
		d_vars_vals = self.sess.run(self.d_vars)
		g_vars_vals = self.sess.run(self.g_vars)
		d_vars_names = [v.name for v in self.d_vars]
		g_vars_names = [v.name for v in self.g_vars]
		return zip(d_vars_vals, d_vars_names), zip(g_vars_vals, g_vars_names)

	def step(self, batch_data, batch_size, gen_update=False, 
		dis_only=False, gen_only=False, stats_only=False, 
		g_layer_stats=False, d_layer_stats=False,
		zi_data=None, run_count=0.0, 
		filter_only=False, output_type='rec'):
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

		### only filter
		if filter_only:
			feed_dict = {self.im_input:batch_data, self.train_phase: False}
			if output_type == 'collect':
				go_layer = self.im_collect
			else:
				go_layer = [self.im_input_rec]
			imo_layer = self.sess.run(go_layer, feed_dict=feed_dict)
			return imo_layer

		### sample z from uniform (-1,1)
		if zi_data is None:
			zi_data = np.random.uniform(low=-self.z_range, high=self.z_range, 
				size=[batch_size, self.z_dim])
		zi_data = zi_data.astype(np_dtype)
		
		### only forward discriminator on batch_data
		if dis_only:
			feed_dict = {self.im_input: batch_data, self.zi_input: zi_data, self.train_phase: False}
			res_list = [self.r_logits_rec, self.rg_grad_norm_output]
			res_list = self.sess.run(res_list, feed_dict=feed_dict)
			return res_list[0].flatten(), res_list[1].flatten()

		### only forward generator on z
		if gen_only:
			feed_dict = {self.zi_input: zi_data, self.train_phase: False}
			if output_type == 'collect':
				go_layer = self.gen_collect
			else:
				go_layer = [self.comb_list[-1]] #self.g_layer_rec
			imo_layer = self.sess.run(go_layer, feed_dict=feed_dict)
			return imo_layer

		### select optimizers for the current time interval
		d_opt_ptr = self.d_opt
		g_opt_ptr = self.g_opt
		summary_ptr = self.summary
		#if run_count < 1e3:
		#	d_opt_ptr = self.d_opt_sub_or
		#	g_opt_ptr = self.g_opt_sub_or
		#	summary_ptr = self.summary_or
		#elif run_count < 2e3:
		#	d_opt_ptr = self.d_opt_sub_ords
		#	g_opt_ptr = self.g_opt_sub_ords
		#	summary_ptr = self.summary_or
		#else:
		#	d_opt_ptr = self.d_opt
		#	g_opt_ptr = self.g_opt
		#	summary_ptr = self.summary_or
		### run one training step on discriminator, otherwise on generator, and log **g_num**
		feed_dict = {self.im_input:batch_data, self.zi_input: zi_data, self.train_phase: True}
		if not gen_update:
			#d_opt_ptr = self.d_opt #if run_count < 5e4 else self.d_opt_sub
			res_list = [summary_ptr, d_opt_ptr]
			res_list = self.sess.run(res_list, feed_dict=feed_dict)
		else:
			#g_opt_ptr = self.g_opt #if run_count < 5e4 else self.g_opt_sub
			res_list = [summary_ptr, g_opt_ptr]
						#self.e_opt, self.pg_opt]
						#self.r_en_h, self.r_en_marg_hlb, self.gi_h, self.g_en_loss]
			res_list = self.sess.run(res_list, feed_dict=feed_dict)
			
		### return summary
		return res_list[0]
