import os
import sys
import argparse
import glob
import numpy as np
import tensorflow as tf
from collections import defaultdict
import pickle as pk
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
	def __init__(self, freqs=None):
		self.g_lr = 2e-4
		self.g_beta1 = 0.9
		self.g_beta2 = 0.999
		self.data_dim = [32, 1]
		self.z_dim = 128
		self.freqs = freqs
		self.build_graph()
		self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
		pass

	def build_graph(self):
		### placeholders for image and z inputs
		self.im_in = tf.placeholder(tf_dtype, [None]+self.data_dim, name='im_input')
		self.z_in = tf.placeholder(tf_dtype, [None, self.z_dim], name='z_input')
		self.train_phase = tf.placeholder(tf.bool, name='phase')

		### model
		with tf.variable_scope('g_net'):
			fmap = 1
			act = tf.nn.relu
			wscale = True

			self.h1 = act(apply_bias(dense_ws(self.z_in, fmap*self.data_dim[0]//4, scope='dense', use_wscale=wscale), scope='bias_dense'))
			self.h1 = tf.reshape(self.h1, [-1, self.data_dim[0]//4, fmap])
			
			self.h1_us = upscale1d(self.h1)
			#self.h1_us = tf.reshape(self.z_in, [-1, self.z_dim, 1])

			self.h2 = act(apply_bias(conv1d_ws(self.h1_us, 1, scope='conv1', use_wscale=wscale), scope='bias_conv1'))
			
			self.h2_us  = upscale1d(self.h2)

			self.h3 = act(apply_bias(conv1d_ws(self.h2_us, 1, scope='conv2', use_wscale=wscale), scope='bias_conv2'))
			self.h4 = apply_bias(conv1d_ws(self.h3, self.data_dim[-1], scope='conv3', use_wscale=wscale), scope='bias_conv3')
			self.out = self.h4

		### loss
		def compute_fft_1d(input_):
			input_ = tf.reshape(input_, [-1, np.prod(input_.get_shape().as_list()[1:])])
			fft = tf.fft(tf.cast(input_, tf.complex64)) / tf.cast(input_.shape[-1], tf.complex64)
			return fft

		def freq_mask_1d(input_, freqs=None):
			if freqs is None:
				return input_
			fft = compute_fft_1d(input_)
			fft_stack = tf.stack([tf.math.real(fft), tf.math.imag(fft)], axis=-1)
			fft_select = tf.gather(fft_stack, freqs, axis=1)
			print(f'>>> fft_select shape: {fft_select.get_shape().as_list()}')
			return fft_select

		def fft_gradient(input_, freq, vars_):
			fft = compute_fft_1d(self.out)[:, freq]
			power_grad = tf.gradients(tf.square(tf.math.real(fft))+tf.square(tf.math.imag(fft)), vars_)
			angle_grad = tf.gradients(tf.math.angle(fft), vars_)
			return power_grad, angle_grad

		def fft_gradient_cos_sim(input_, freqs, vars_):
			f1, f2 = freqs
			p1, a1 = fft_gradient(input_, f1, vars_)
			p2, a2 = fft_gradient(input_, f2, vars_)
			p1 = tf.concat([tf.reshape(t, [-1]) for t in p1], axis=-1)
			a1 = tf.concat([tf.reshape(t, [-1]) for t in a1], axis=-1)
			p2 = tf.concat([tf.reshape(t, [-1]) for t in p2], axis=-1)
			a2 = tf.concat([tf.reshape(t, [-1]) for t in a2], axis=-1)
			
			power_sim = tf.abs(tf.reduce_sum(p1 * p2) / tf.sqrt(tf.reduce_sum(p1**2)*tf.reduce_sum(p2**2)+1e-10))
			angle_sim = tf.abs(tf.reduce_sum(a1 * a2) / tf.sqrt(tf.reduce_sum(a1**2)*tf.reduce_sum(a2**2)+1e-10))
			return power_sim, angle_sim

		self.out_freq = freq_mask_1d(self.out, self.freqs)
		self.im_in_freq = freq_mask_1d(self.im_in, self.freqs)
		self.g_loss = tf.reduce_sum(tf.reduce_mean(tf.square(self.out_freq - self.im_in_freq), axis=0))

		### optimize
		self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'g_net')
		self.g_opt_handle = tf.train.AdamOptimizer(self.g_lr, beta1=self.g_beta1, beta2=self.g_beta2)
		self.g_grads_vars = self.g_opt_handle.compute_gradients(self.g_loss, self.g_vars)
		self.g_opt = self.g_opt_handle.apply_gradients(self.g_grads_vars)

		### compute gradient of power and phase cos similarity
		self.fft_cos_sim = list()
		self.fft_cos_sim.append(fft_gradient_cos_sim(self.out, [1, 2], self.g_vars))
		self.fft_cos_sim.append(fft_gradient_cos_sim(self.out, [self.data_dim[0]//2-2, self.data_dim[0]//2-1], self.g_vars))

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

	def add_session(self, sess, log_dir):
		self.sess = sess
		self.log_dir = log_dir
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

def run_toy(toygan, sess, log_dir, eval_int, freq, g_itrs=int(5e4), verbose=True):
	print(f'>>> Run settings: freq={freq} eval_int={eval_int} g_itrs={g_itrs}')
	os.mkdir(log_dir)
	log_dir_snaps = os.path.join(log_dir, 'snapshots')
	os.mkdir(log_dir_snaps)
	log_dir_sums = os.path.join(log_dir, 'summary')
	os.mkdir(log_dir_sums)
	sys.stdout = Logger(log_dir)
	sys.stderr = sys.stdout

	### add a session to toygan for summary writing
	toygan.add_session(sess, log_dir_sums)
	if verbose:
		toygan.print()
	### init variables
	sess.run(tf.global_variables_initializer())

	'''
	Train
	'''
	fft_guides = [-freq, freq]
	z_in = np.random.uniform(-1, 1, toygan.z_dim).reshape([-1, toygan.z_dim])
	plot_fft_1d(z_in, os.path.join(log_dir, f'z_in_size{toygan.z_dim}.png'), fft_guides=fft_guides)
	#im_in = np.random.uniform(-1, 1, toygan.data_dim).reshape([1, toygan.data_dim[0], 1])
	im_in = np.cos(2*np.pi*freq / toygan.data_dim[0] * np.arange(toygan.data_dim[0])).reshape([1, toygan.data_dim[0], 1])
	plot_fft_1d(im_in, os.path.join(log_dir, f'true_cos{freq}_size{toygan.data_dim[0]}.png'), fft_guides=fft_guides)
	#im_fft = sess.run(tf.fft(tf.convert_to_tensor(im_in.reshape([-1]).astype(complex), dtype=tf.complex64)))
	#plot_fft_1d(im_in, os.path.join(log_dir, f'true_cos{freq}_size{toygan.data_dim[0]}_tf.png'), fft=im_fft)
	fft_cos_sims = list()
	loss_test_pre = 1e10
	for i in range(g_itrs+1):
		if i % eval_int == 0 or i == g_itrs:
			if verbose:
				print(f'\n>>> At iteration {i:06d}:')

			toygan.save(os.path.join(log_dir_snaps, f'model_{i:06d}.h5'))
			im_test, loss_test, grads_vars, sums, fft_cos_sim = sess.run(
				[toygan.out, toygan.g_loss, toygan.g_grads_vars, toygan.summary, toygan.fft_cos_sim],
				feed_dict={toygan.z_in: z_in, toygan.im_in: im_in, toygan.train_phase: False})
			loss_test_delta = loss_test_pre - loss_test
			loss_test_pre = loss_test
			fft_cos_sims.append(fft_cos_sim)
			#out_freq, im_in_freq = sess.run([toygan.out_freq, toygan.im_in_freq],
			#	feed_dict={toygan.z_in: z_in, toygan.im_in: im_in, toygan.train_phase: False})
			
			toygan.write_sum(sums, i)
			
			plot_fft_1d(im_test, 
				os.path.join(log_dir, f'gen_cos{freq}_size{toygan.data_dim[0]}_{i:06d}.png'), fft_guides=fft_guides)
			plot_fft_1d(im_test-im_in, 
				os.path.join(log_dir, f'gen_cos{freq}_size{toygan.data_dim[0]}_{i:06d}_diff.png'), ylim=False, fft_guides=fft_guides)

			if verbose:
				print(f'>>> loss = {loss_test}') # im_in_freq={im_in_freq} out_freq={out_freq}
				#print_grads_vars_stats(toygan.g_vars, [gv[0] for gv in grads_vars], [gv[1] for gv in grads_vars])
			#print(f'>>> pow_sim_low={fft_cos_sim[0][0]:.3f} pow_sim_high={fft_cos_sim[1][0]:.3f} angle_sim_low={fft_cos_sim[0][1]:.3f} angle_sim_high={fft_cos_sim[1][1]:.3f}\n')
			if i == g_itrs:
				break

		feed_dict = {toygan.z_in: z_in, toygan.im_in: im_in, toygan.train_phase: True}
		res_ops = [toygan.out, toygan.g_loss, toygan.g_opt]
		res = sess.run(res_ops, feed_dict=feed_dict)

	toygan.flush()

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
		elif verbose:
			print(f'>>> summary of {k}: {v}')

	if len(histo_mean_std) > 0:
		plot_multi(histo_mean_std, os.path.join(log_dir, f'histograms.png'))

	'''
	PLot fft cosine similarities
	'''
	fft_cos_sim_mean_std = defaultdict(list)
	for vl, vh in fft_cos_sims:
		fft_cos_sim_mean_std['power_low'].append((vl[0], 0))
		fft_cos_sim_mean_std['power_high'].append((vh[0], 0))
		fft_cos_sim_mean_std['phase_low'].append((vl[1], 0))
		fft_cos_sim_mean_std['phase_high'].append((vh[1], 0))
	plot_multi(fft_cos_sim_mean_std, os.path.join(log_dir, f'fft_cos_sims.png'))

	return loss_test, loss_test_delta


if __name__ == '__main__':
	'''
	Script Setup
	'''
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('-l', '--log-path', dest='log_path', required=True, help='log directory to store logs.')
	arg_parser.add_argument('-s', '--seed', dest='seed', default=0, help='random seed.')
	arg_parser.add_argument('-e', '--eval', dest='eval_int', required=True, help='eval intervals.')
	arg_parser.add_argument('-g', '--gpus', dest='gpus', default='', help='visible gpu ids.')
	arg_parser.add_argument('-f', '--freq', dest='freq', default='0', help='frequency to use.')
	args = arg_parser.parse_args()
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
	os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.gpus) # "0, 1" for multiple
	log_dir = args.log_path
	eval_int = int(args.eval_int)
	run_seed = int(args.seed)
	freq = int(args.freq)
	np.random.seed(run_seed)
	tf.set_random_seed(run_seed)
	os.mkdir(log_dir)
	sys.stdout = Logger(log_dir)
	sys.stderr = sys.stdout

	'''
	TENSORFLOW SETUP
	'''
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
	config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
	config.gpu_options.allow_growth = True
	### create a ganist instance
	toygan = ToyGAN()

	'''
	Run GAN
	'''
	losses = list()
	deltas = list()
	freqs = np.arange(17)
	for i, freq in enumerate(freqs):
		print(f'>>> At {i}th frequency: {freq}')
		with tf.Session(config=config) as sess:
			loss, delta = run_toy(toygan, sess, os.path.join(log_dir, f'_{freq:03d}'), eval_int, freq, g_itrs=int(5e4), verbose=True)
			losses.append(loss)
			deltas.append(delta)
			print(f'>>> loss = {loss} delta = {delta}\n')

	with open(os.path.join(log_dir, 'losses.cpk'), 'wb+') as fs:
		pk.dump([losses, deltas, freqs], fs)

	plot_simple('power diff', losses, os.path.join(log_dir, 'losses.png'), steps=freqs)
	plot_simple('last loss drop', deltas, os.path.join(log_dir, 'deltas.png'), steps=freqs)
	







