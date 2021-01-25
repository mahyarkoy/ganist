import os
import sys
import argparse
import glob
import numpy as np
import tensorflow as tf
from collections import defaultdict
from util import plot_fft_1d, plot_simple, plot_multi, Logger

tf_dtype = tf.float32
np_dtype = np.float32

### Operations from pggan
def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
	if fan_in is None: fan_in = np.prod(shape[:-1])
	std = gain / np.sqrt(fan_in) # He init
	if use_wscale:
		wscale = tf.constant(np.float32(std), name='wscale')
		return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
	else:
		return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))

def dense_ws(x, fmaps, gain=np.sqrt(2), use_wscale=False, scope='dense', reuse=False):
	with tf.variable_scope(scope, reuse=reuse):
		if len(x.shape) > 2:
			x = tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])
		w = get_weight([x.shape[-1].value, fmaps], gain=gain, use_wscale=use_wscale)
		w = tf.cast(w, x.dtype)
		return tf.matmul(x, w)

def conv2d_ws(x, fmaps, kernel=3, gain=np.sqrt(2), use_wscale=False, scope='conv', reuse=False):
	with tf.variable_scope(scope, reuse=reuse):
		assert kernel >= 1 and kernel % 2 == 1
		w = get_weight([kernel, kernel, x.shape[-1].value, fmaps], gain=gain, use_wscale=use_wscale)
		w = tf.cast(w, x.dtype)
		return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

def conv1d_ws(x, fmaps, kernel=3, gain=np.sqrt(2), use_wscale=False, scope='conv', reuse=False):
	with tf.variable_scope(scope, reuse=reuse):
		assert kernel >= 1 and kernel % 2 == 1
		w = get_weight([kernel, x.shape[-1].value, fmaps], gain=gain, use_wscale=use_wscale)
		w = tf.cast(w, x.dtype)
		return tf.nn.conv1d(x, w, stride=1, padding='SAME')


def apply_bias(x, scope='bias', reuse=False):
	with tf.variable_scope(scope, reuse=reuse):
		b = tf.get_variable('bias', shape=[x.shape[-1]], initializer=tf.initializers.zeros())
		b = tf.cast(b, x.dtype)
		return x + b

def lrelu(x, alpha=0.2):
	with tf.name_scope('LeakyRelu'):
		alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
		return tf.maximum(x * alpha, x)

def upscale2d(x, factor=2):
	assert isinstance(factor, int) and factor >= 1
	if factor == 1: return x
	s = x.shape
	x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
	x = tf.tile(x, [1, 1, factor, 1, factor, 1])
	x = tf.reshape(x, [-1, s[1] * factor, s[2] * factor, s[3]])
	return x

def upscale1d(x, factor=2):
	assert isinstance(factor, int) and factor >= 1
	if factor == 1: return x
	s = x.shape
	x = tf.reshape(x, [-1, s[1], 1, s[2]])
	x = tf.tile(x, [1, 1, factor, 1])
	x = tf.reshape(x, [-1, s[1] * factor, s[2]])
	return x

class ToyGAN:
	def __init__(self, sess, log_dir):
		self.log_dir = log_dir
		self.g_lr = 2e-4
		self.g_beta1 = 0.9
		self.g_beta2 = 0.999
		self.data_dim = [128, 1]
		self.z_dim = 32
		self.sess = sess
		self.build_graph()
		self.start()
		pass

	def build_graph(self):
		### placeholders for image and z inputs
		self.im_in = tf.placeholder(tf_dtype, [None]+self.data_dim, name='im_input')
		self.z_in = tf.placeholder(tf_dtype, [None, self.z_dim], name='z_input')
		self.train_phase = tf.placeholder(tf.bool, name='phase')

		### model
		with tf.variable_scope('g_net'):
			fmap = 32
			act = tf.nn.relu
			wscale = True

			self.h1 = act(apply_bias(dense_ws(self.z_in, fmap*32, scope='dense', use_wscale=wscale), scope='bias_dense'))
			self.h1 = tf.reshape(self.h1, [-1, 32, fmap])
			
			self.h1_us = upscale1d(self.h1)
			self.h2 = act(apply_bias(conv1d_ws(self.h1_us, fmap, scope='conv1', use_wscale=wscale), scope='bias_conv1'))
			
			self.h2_us  = upscale1d(self.h2)
			self.h3 = apply_bias(conv1d_ws(self.h2_us, self.data_dim[-1], scope='conv2', use_wscale=wscale), scope='bias_conv2')
			self.out = self.h3

		### loss
		self.g_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.out - self.im_in), axis=[1,2]))

		### optimize
		self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'g_net')
		self.g_opt_handle = tf.train.AdamOptimizer(self.g_lr, beta1=self.g_beta1, beta2=self.g_beta2)
		self.g_grads_vars = self.g_opt_handle.compute_gradients(self.g_loss, self.g_vars)
		self.g_opt = self.g_opt_handle.apply_gradients(self.g_grads_vars)

		### compute stat of weights
		self.nan_vars = 0.
		self.inf_vars = 0.
		self.zero_vars = 0.
		self.big_vars = 0.
		self.count_vars = 0
		for v in self.g_vars:
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

		### summaries
		for g, v in self.g_grads_vars:
			tf.summary.histogram(v.name+'_gradient', g)
			tf.summary.histogram(v.name+'_value', v)
		tf.summary.scalar('g_loss', self.g_loss),
		tf.summary.scalar('nan_vars', self.nan_vars),
		tf.summary.scalar('inf_vars', self.inf_vars),
		tf.summary.scalar('zero_vars', self.zero_vars),
		tf.summary.scalar('big_vars', self.zero_vars),
		tf.summary.scalar('count_vars', self.count_vars)
		self.summary = tf.summary.merge_all()

	def start(self):
		self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
		self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

	def flush(self):
		self.writer.flush()
		self.writer.close()

	def save(self, fname):
		self.saver.save(self.sess, fname)

	def load(self, fname):
		self.saver.restore(self.sess, fname)

	def write_sum(self, sum_str, counter):
		self.writer.add_summary(sum_str, counter)

	def print(self):
		print('>>> printing network:')
		for v in self.g_vars:
			if 'Adam' not in v.name:
				print(f'{v.name} {v.get_shape()}')
		print(f'>>> total vars: {np.sum([np.prod(v.get_shape().as_list()) for v in self.g_vars])}')

def print_grads_vars_stats(vars_, grads_, vals_):
	for v, g, val in zip(vars_, grads_, vals_):
		print(f'>>> variable {v.name} gradient={np.mean(g)}_sd_{np.std(g)} value={np.mean(val)}_sd_{np.std(val)}')

if __name__ == '__main__':
	'''
	Script Setup
	'''
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('-l', '--log-path', dest='log_path', required=True, help='log directory to store logs.')
	arg_parser.add_argument('-s', '--seed', dest='seed', default=0, help='random seed.')
	arg_parser.add_argument('-e', '--eval', dest='eval_int', required=True, help='eval intervals.')
	arg_parser.add_argument('-g', '--gpus', dest='gpus', default='0', help='visible gpu ids.')
	arg_parser.add_argument('-f', '--freq', dest='freq', default='0', help='frequency to use.')
	args = arg_parser.parse_args()
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
	os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.gpus) # "0, 1" for multiple
	log_dir = args.log_path
	eval_int = int(args.eval_int)
	run_seed = int(args.seed)
	np.random.seed(run_seed)
	tf.set_random_seed(run_seed)
	os.mkdir(log_dir)
	log_dir_snaps = os.path.join(log_dir, 'snapshots')
	os.mkdir(log_dir_snaps)
	log_dir_sums = os.path.join(log_dir, 'summary')
	os.mkdir(log_dir_sums)
	sys.stdout = Logger(log_dir)
	sys.stderr = sys.stdout

	'''
	TENSORFLOW SETUP
	'''
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
	config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	### create a ganist instance
	toygan = ToyGAN(sess, log_dir_sums)
	toygan.print()
	### init variables
	sess.run(tf.global_variables_initializer())

	'''
	Train
	'''
	g_itrs = int(1e4)
	freq = int(args.freq)
	z_in = np.random.uniform(-1, 1, toygan.z_dim).reshape([-1, toygan.z_dim])
	im_in = np.cos(2*np.pi*freq / toygan.data_dim[0] * np.arange(toygan.data_dim[0])).reshape([1, toygan.data_dim[0], 1])
	plot_fft_1d(im_in, os.path.join(log_dir, f'true_cos{freq}_size{toygan.data_dim[0]}.png'))
	for i in range(g_itrs+1):
		if i % eval_int == 0 or i == g_itrs:
			toygan.save(os.path.join(log_dir_snaps, f'model_{i:06d}.h5'))
			im_test, loss_test, grads_vars, sums = sess.run([toygan.out, toygan.g_loss, toygan.g_grads_vars, toygan.summary],
				feed_dict={toygan.z_in: z_in, toygan.im_in: im_in, toygan.train_phase: False})
			
			toygan.write_sum(sums, i)
			print_grads_vars_stats(toygan.g_vars, [gv[0] for gv in grads_vars], [gv[1] for gv in grads_vars])
			
			plot_fft_1d(im_test, 
				os.path.join(log_dir, f'gen_cos{freq}_size{toygan.data_dim[0]}_{i:06d}.png'))
			plot_fft_1d(im_test-im_in, 
				os.path.join(log_dir, f'gen_cos{freq}_size{toygan.data_dim[0]}_{i:06d}_diff.png'), ylim=False)
			print(f'>>> At iteration {i:06d}: loss = {loss_test}')
			if i == g_itrs:
				break

		feed_dict = {toygan.z_in: z_in, toygan.im_in: im_in, toygan.train_phase: True}
		res_ops = [toygan.out, toygan.g_loss, toygan.g_opt]
		res = sess.run(res_ops, feed_dict=feed_dict)

	toygan.flush()
	sess.close()

	'''
	Plot summaries
	'''
	metrics = defaultdict(list)
	histo_mean_std = defaultdict(list)
	steps = list()
	for e in tf.train.summary_iterator(glob.glob(os.path.join(log_dir_sums, '*'))[0]):
		steps.append(e.step)
		for v in e.summary.value:
			if v.HasField('simple_value'):
				metrics[v.tag].append(v.simple_value)
			elif v.HasField('histo'):
				mean = np.float(v.histo.sum)/v.histo.num
				std = np.sqrt(np.float(v.histo.sum_squares)/v.histo.num - mean**2)
				histo_mean_std[v.tag].append((mean, std))
	
	for k, v in metrics.items():
		if len(v) > 1:
			plot_simple(k, v, os.path.join(log_dir, f'sums_{k}.png'))
		else:
			print(f'>>> summary of {k}: {v}')

	if len(histo_mean_std) > 0:
		plot_multi(histo_mean_std, os.path.join(log_dir, f'histograms.png'))







