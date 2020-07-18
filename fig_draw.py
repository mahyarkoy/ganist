import numpy as np
import tensorflow as tf
from run_ganist import block_draw, im_block_draw
from run_ganist import TFutil, sample_ganist, create_lsun, CUB_Sampler, PG_Sampler
import matplotlib.pyplot as plt
from PIL import Image
import tf_ganist
import sys
from os.path import join
from util import apply_fft_win, COS_Sampler, freq_density, read_celeba, apply_fft_images, apply_ifft_images, pyramid_draw
from util import eval_toy_exp, mag_phase_wass_dist, mag_phase_total_variation
from util import Logger, readim_path_from_dir, readim_from_path
import glob
import os
import pickle as pk
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1" for multiple

'''
Drawing Freq Components
'''
def freq_shift(im, fc_x, fc_y):
	im_size = im.shape[1]
	kernel_loc = 2.*np.pi*fc_x * np.arange(im_size).reshape((1, im_size, 1)) + \
		2.*np.pi*fc_y * np.arange(im_size).reshape((im_size, 1, 1))
	kernel_cos = np.cos(kernel_loc)
	return im * kernel_cos

def draw_freq_comps():
	comp_size = 32
	freq_list = np.array([(0., 1.), (1., 0), (1., 1.), (1., -1.)]) / 4.
	im = np.ones((comp_size, comp_size, 3))
	freq_comps = list()
	freq_comps_sh = list()
	for f in freq_list.tolist():
		im_sh = freq_shift(im, f[0], f[1])
		freq_comps.append(im_sh)
		freq_comps_sh.append(freq_shift(im, f[0]/4., f[1]/4.))
	
	freq_comps = np.array(freq_comps).reshape((1, -1, comp_size, comp_size, 3))
	freq_comps_sh = np.array(freq_comps_sh).reshape((1, -1, comp_size, comp_size, 3))
	block_draw(freq_comps, '/home/mahyar/miss_details_images/temp/freq_comps.png', border=True)
	block_draw(freq_comps_sh, '/home/mahyar/miss_details_images/temp/freq_comps_sh.png', border=True)

def single_draw(im, path):
	imc = im.shape[-1]
	im_draw = (im + 1.0) / 2.0
	im_draw = im_draw if imc == 3 else np.repeat(im_draw[:,:,:1], 3, axis=2)
	
	### new draw without matplotlib
	im_out = np.clip(np.rint(im_draw * 255.0), 0.0, 255.0).astype(np.uint8)
	Image.fromarray(im_out, 'RGB').save(path)
	return

def leakage_test(log_dir, im_size=128, ksize=128, fc_x=43./128, fc_y=1./128):
	kernel_loc = 2.*np.pi*fc_x * np.arange(ksize).reshape((1, 1, ksize, 1)) + \
		2.*np.pi*fc_y * np.arange(ksize).reshape((1, ksize, 1, 1))
	kernel_cos = np.cos(kernel_loc)
	im_data = np.zeros((1, im_size, im_size, 1))
	im_data[0, :ksize, :ksize, :1] = kernel_cos
	#im_data *= (im_data > 0).astype(int)
	im_data = np.tanh(im_data)
	apply_fft_win(im_data, 
		join(log_dir, 'tanh_fft_im{}_cos{}_fx{}_fy{}.png'.format(im_size, ksize, int(im_size*fc_x), int(im_size*fc_y))), 
		windowing=False)
	single_draw(im_data[0], 
		join(log_dir, 'tanh_im{}_cos{}_fx{}_fy{}.png'.format(im_size, ksize, int(im_size*fc_x), int(im_size*fc_y))))

def fft_test(log_dir, sess=None, run_seed=0):
	data_size = 10000
	im_size = 128

	use_shifter = True ## set to True for freq shift dataset
	def shifter(x): return TFutil.get().freq_shift(x, 0.5, 0.5) if use_shifter else x	

	### ganist load g_samples
	#g_name = '5_logs_wganbn_celeba128cc_fid50'
	#g_name = '14_logs_wganbn_celeba128cc_fssetup_fshift'
	#g_name = '38_logs_wganbn_bedroom128cc'
	#g_name = '46_logs_wgan_sbedroom128cc_hpfid'
	g_name = '22_logs_wganbn_gshift_celeba128cc_fshift'
	#g_name = '49_logs_wgan_gshift_sbedroom128cc_hpfid'
	ganist = tf_ganist.Ganist(sess, log_dir)
	sess.run(tf.global_variables_initializer())
	net_path = f'/dresden/users/mk1391/evl/ganist_lap_logs/{g_name}/run_{run_seed}/snapshots/model_best.h5'
	ganist.load(net_path)
	g_samples = sample_ganist(ganist, data_size, output_type='rec')[0]

	### pggan load g_samples
	#sys.path.insert(1, '/dresden/users/mk1391/evl/Data/pggan_model')
	#g_name = f'gdsmall_results_{run_seed}'
	#net_path = f'/dresden/users/mk1391/evl/pggan_logs/logs_celeba128cc/{g_name}/000-pgan-celeba-preset-v2-2gpus-fp32/network-snapshot-010211.pkl'
	#net_path = f'/dresden/users/mk1391/evl/pggan_logs/logs_bedroom128cc/{g_name}/000-pgan-lsun-bedroom-preset-v2-2gpus-fp32/network-snapshot-010211.pkl'
	#g_name = f'results_gdsmall_outsh_nomirror_sceleba_{run_seed}'
	#net_path = f'/dresden/users/mk1391/evl/pggan_logs/logs_celeba128cc_sh/{g_name}/000-pgan-celeba-preset-v2-2gpus-fp32/network-snapshot-010211.pkl'
	#g_name = f'results_gdsmall_sbedroom_{run_seed}'
	#g_name = f'results_gdsmall_outsh_nomirror_sbedroom_{run_seed}'
	#net_path = f'/dresden/users/mk1391/evl/pggan_logs/logs_bedroom128cc_sh/{g_name}/000-pgan-lsun-bedroom-preset-v2-2gpus-fp32/network-snapshot-010211.pkl'
	#pg_sampler = PG_Sampler(net_path, sess, net_type='tf')
	#g_samples = pg_sampler.sample_data(data_size)

	### load g_samples from pickle file
	#with open(join(log_dir, f'{g_name}_samples_{data_size}.pk'), 'rb') as fs:
	#	g_samples = pk.load(fs)

	### load g_samples from image file
	#g_name = 'logs_ganms_or_celeba128cc'
	#g_sample_dir = f'/dresden/users/mk1391/evl/ganist_lsun_logs/layer_stats/temp/{g_name}/run_0/samples/'
	#g_samples = readim_from_path(
	#	readim_path_from_dir(g_sample_dir, im_type='*.jpg')[:data_size], im_size, center_crop=(64,64), verbose=True)

	### load r_samples
	r_name = 'sceleba128cc'
	r_samples = read_celeba(im_size, data_size)

	#r_name = 'sbedroom128cc'
	#lsun_lmdb_dir = '/dresden/users/mk1391/evl/data_backup/lsun/bedroom_train_lmdb/'
	#r_samples, idx_list = create_lsun(lsun_lmdb_dir, resolution=im_size, max_images=data_size)

	r_samples = shifter(r_samples)
	
	assert r_samples.shape[0] == data_size, f'r_samples count of {r_samples.shape[0]} != data_size count of {data_size}'
	Logger.add_file_handler(f'log_{g_name}_{r_name}_{data_size}')
	im_block_draw(r_samples, 5, join(log_dir, f'{r_name}_samples.png'), border=True)
	im_block_draw(g_samples, 5, join(log_dir, f'{g_name}_{r_name}_samples.png'), border=True)
	im_block_draw(shifter(r_samples), 5, join(log_dir, f'{r_name}_samples_sh.png'), border=True)
	im_block_draw(shifter(g_samples), 5, join(log_dir, f'{g_name}_{r_name}_samples_sh.png'), border=True)
	#with open(join(log_dir, f'{g_name}_samples.pk'), 'wb+') as fs:
	#	pk.dump(g_samples, fs)

	def windowing(imgs, skip=False):
		if skip: return imgs
		win_size = imgs.shape[1]
		win = np.hanning(imgs.shape[1])
		win = np.outer(win, win).reshape((win_size, win_size, 1))
		return win*imgs

	def fft_norm(imgs):
		imgs_fft, _ = apply_fft_images(imgs, reshape=False)
		fft_power = np.abs(imgs_fft)**2
		fft_avg = np.mean(fft_power, axis=0)
		np.clip(fft_avg, 1e-20, None, out=fft_avg)
		return fft_avg

	### calculate fft mean
	r_fft_mean = fft_norm(windowing(r_samples))
	g_fft_mean = fft_norm(windowing(g_samples))

	### read fft mean data from file
	#with open(join(log_dir, f'{g_name}_{r_name}_{data_size}_fft_mean.pk'), 'rb') as fs:
	#	g_fft_mean, r_fft_mean = pk.load(fs)

	### write fft mean data to file
	with open(join(log_dir, f'{g_name}_{r_name}_{data_size}_fft_mean.pk'), 'wb+') as fs:
		pk.dump([g_fft_mean, r_fft_mean], fs)

	### compute aggregated frequency difference
	blur_levels = [1.]
	blur_num = 7
	blur_delta = 1. / 8
	blur_init = blur_levels[-1]
	for i in range(blur_num):
		blur_levels.append(
			1. / ((1. / blur_levels[-1]) - (blur_delta / blur_init)))
	r_fft_density = r_fft_mean / np.sum(r_fft_mean)
	g_fft_density = g_fft_mean / np.sum(g_fft_mean)
	fft_diff = np.abs(np.log(r_fft_mean) - np.log(g_fft_mean))
	total_var = 100. * np.sum(np.abs(r_fft_density - g_fft_density)) / 2.
	Logger.print(f'>>> fft_test_{g_name}_{r_name}: Leakage percentage (TV): {total_var}')
	bins_loc = [0.] + [1./(np.pi*2*s) for s in blur_levels[::-1]]
	#bins_loc = [0.] + list(np.arange(1, 50) / 100.)
	bins = np.zeros(len(bins_loc))
	bins_count = np.array(bins)
	fft_h, fft_w = fft_diff.shape
	Logger.print('>>> freq bins:')
	for v in range(fft_h):
		bin_str = ''
		for u in range(fft_w):
			fft_hc = fft_h//2
			fft_wc = fft_w//2
			freq = np.sqrt(((v - fft_hc + 1 - fft_h%2)/fft_hc)**2. + ((u - fft_wc)/fft_wc)**2)
			for i, bin_freq in enumerate(bins_loc[::-1]):
				if freq >= bin_freq:
					bin_id = len(bins_loc) - i - 1
					bins[bin_id] += fft_diff[v, u]
					bins_count[bin_id] += 1
					break
			bin_str += f'{bin_id}'
		Logger.print(bin_str)

	Logger.print(f'>>> freq bins:\t{bins_loc}')
	Logger.print(f'>>> freq diff density:\t{bins / bins_count}')

	### prepare figure data
	fig_data = list()
	fig_data.append(np.abs(np.log(r_fft_mean) - np.log(g_fft_mean)))
	fig_data.append(np.log(r_fft_mean / np.amax(r_fft_mean)))
	fig_data.append(np.log(g_fft_mean / np.amax(g_fft_mean)))
	
	### draw figure
	fig_names = ['diff', 'true', 'gan', 'agg']
	fig_opts = [{'cmap': plt.get_cmap('inferno'), 'vmin': 0}, 
				{'cmap': plt.get_cmap('inferno'), 'vmin': -13, 'vmax': 0}, 
				{'cmap': plt.get_cmap('inferno'), 'vmin': -13, 'vmax': 0}]
	#fig, axes = plt.subplots(1, 4, figsize=(24, 6), num=0, gridspec_kw={'width_ratios': [1, 1, 1, 2]})
	fig = plt.figure(0, figsize=(8,6))
	for i in range(len(fig_names)):
		fig.clf()
		ax = fig.add_subplot(1,1,1)
		if fig_names[i] == 'agg':
			ax.grid(True, which='both', linestyle='dotted')
			ax.set_xlabel(r'Frequency bins')
			ax.set_ylabel('Power diff density')
			ax.set_title('Power diff density')
			ax.plot(bins / bins_count)
			ax.set_xticks(range(len(bins_loc)))
			xticklabels = [np.format_float_positional(v, 2) for v in bins_loc]
			ax.set_xticklabels(xticklabels)
			fig.savefig(join(log_dir, f'fft_diff_{g_name}_{r_name}_{fig_names[i]}.png'), dpi=300)
			break

		im = ax.imshow(fig_data[i], **fig_opts[i])
		ax.set_title(fig_names[i])
		dft_size = im_size
		ticks_loc_x = [0, dft_size//2]
		ticks_loc_y = [0, dft_size//2-1, dft_size-dft_size%2-1]
		ax.set_xticks(ticks_loc_x)
		ax.set_xticklabels([-0.5, 0])
		ax.set_yticks(ticks_loc_y)
		ax.set_yticklabels(['', 0, -0.5])
		plt.colorbar(im)#, ax=ax, fraction=0.046, pad=0.04)

		fig.savefig(join(log_dir, f'fft_diff_{g_name}_{r_name}_{fig_names[i]}.png'), dpi=300)

#celeba_data = read_celeba(32)
#apply_fft_win(celeba_data, log_dir+'/fft_celeba32cc_hann.png')

#celeba_data = read_celeba(64)
#apply_fft_win(celeba_data, log_dir+'/fft_celeba64cc_hann.png')

#celeba_data = freq_shift(read_celeba(128), 0.5, 0.5)
#apply_fft_win(celeba_data, log_dir+'/fft_celeba128cc_sh_hann.png')

'''
FFT on GANIST
'''
if __name__ == '__main__':
	#log_dir = 'logs_draw/'
	#log_dir = '/dresden/users/mk1391/evl/eval_samples/'
	#os.makedirs(log_dir, exist_ok=True)
	
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('-l', '--log-dir', dest='log_dir', required=True, help='log directory to store logs.')
	arg_parser.add_argument('-s', '--seed', dest='seed', default=0, help='random seed.')
	args = arg_parser.parse_args()
	log_dir = args.log_dir
	run_seed = int(args.seed)
	np.random.seed(run_seed)
	tf.set_random_seed(run_seed)
	
	Logger(log_dir, fname='log_file')
	
	'''
	Eval toy experiments.
	'''
	#for i in range(13, 20):
	#	exp_dir = glob.glob('/dresden/users/mk1391/evl/ganist_toy_logs/{}_*/'.format(i))[0]
	#	eval_toy_exp(exp_dir, im_size=128)
	#	break

	'''
	Eval distribution distance
	'''
	#data_size = 10000
	#mag = np.clip(np.random.normal(loc=0.5, scale=0.1, size=(data_size)), 0., 1.)
	#phase = np.clip(np.random.normal(loc=0., scale=0.2*np.pi, size=(data_size)), -np.pi, np.pi)
	#fig = plt.figure(0, figsize=(8,6))
	#fig.clf()
	#ax_mag = fig.add_subplot(2,1,1)
	#ax_phase = fig.add_subplot(2,1,2)
	#mag_count, mag_bins, _ = ax_mag.hist(mag, 100, range=(0., 1.), density=True)
	#phase_count, phase_bins, _ = ax_phase.hist(phase, 100, range=(-np.pi, np.pi), density=True)
	#true_hist = [mag_bins, mag_count, phase_bins, phase_count]
	#fig.savefig(join(log_dir, 'true_mag_phase.png'), dpi=300)
	#
	#mag = np.clip(np.random.normal(loc=0.5, scale=0.01, size=(data_size)), 0., 1.)
	#phase = np.clip(np.random.normal(loc=0., scale=0.2*np.pi, size=(data_size)), -np.pi, np.pi)
	#fig = plt.figure(1, figsize=(8,6))
	#fig.clf()
	#ax_mag = fig.add_subplot(2,1,1)
	#ax_phase = fig.add_subplot(2,1,2)
	#mag_count, mag_bins, _ = ax_mag.hist(mag, 100, range=(0., 1.), density=True)
	#phase_count, phase_bins, _ = ax_phase.hist(phase, 100, range=(-np.pi, np.pi), density=True)
	#gen_hist = [mag_bins, mag_count, phase_bins, phase_count]
	#fig.savefig(join(log_dir, 'gen_mag_phase.png'), dpi=300)
	#
	#mag_wd, phase_wd = mag_phase_wass_dist(true_hist, gen_hist)
	#mag_tv, phase_tv = mag_phase_total_variation(true_hist, gen_hist)
	#print('mag_wd: {} --- phase_wd: {}'.format(mag_wd, phase_wd))
	#print('mag_tv: {} --- phase_tv: {}'.format(mag_tv, phase_tv))

	'''
	Filter draw
	'''
	### kernel
	#im_size = 128
	#ksize = 41
	#sigma = 1.
	#t = np.linspace(-20, 20, ksize)
	##t = np.linspace(-20, 20, 81) ## for 128x128 images
	#bump = np.exp(0.5 * -t**2/sigma**2)
	#bump /= np.sum(bump) # normalize the integral to 1
	#kernel = bump[:, np.newaxis] * bump[np.newaxis, :]
	#im_data = np.zeros((1, im_size, im_size, 1))
	#im_data[0, :ksize, :ksize, 0] = kernel
	#apply_fft_win(im_data, 
	#		join(log_dir, 'fft_gauss_kernel_ksize{}_imsize{}.png'.format(ksize, im_size)), 
	#		windowing=False)
	#apply_fft_win(im_data, 
	#		join(log_dir, 'fft_gauss_kernel_ksize{}_imsize{}_hann.png'.format(ksize, im_size)), 
	#		windowing=True)
	#single_draw(im_data[0], join(log_dir, 'gauss_kernel_ksize{}_imsize{}.png'.format(ksize, im_size)))

	'''
	Filter draw range
	'''
	### kernel
	#im_size = 128
	#krange = 25
	#ksize = 2*krange+1
	#sigma = 1.
	#t = np.linspace(-krange, krange, ksize)
	#blur_levels = [0., 1.]
	#blur_num = 7
	#blur_delta = 1. / 8
	#### reducing the filter radius by blur_delta every step
	#blur_init = blur_levels[-1]
	#for i in range(blur_num):
	#	blur_levels.append(
	#		1. / ((1. / blur_levels[-1]) - (blur_delta / blur_init)))
	###t = np.linspace(-20, 20, 81) ## for 128x128 images
	##blur_levels = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
	#fig = plt.figure(0, figsize=(8*len(blur_levels),6))
	#fig.clf()
	#for i, sigma in enumerate(blur_levels):
	#	if sigma != 0:
	#		bump = np.exp(0.5 * -t**2/sigma**2)
	#		bump /= np.sum(bump) # normalize the integral to 1
	#		kernel = bump[:, np.newaxis] * bump[np.newaxis, :]
	#		im_data = np.zeros((1, im_size, im_size, 1))
	#		im_data[0, :ksize, :ksize, 0] = kernel
	#		#im_in = np.zeros((1, im_size, im_size, 1))
	#		#im_in[0, krange, krange, 0] = 1.
	#		#im_data = im_in - im_data
	#	else:
	#		im_data = np.zeros((1, im_size, im_size, 1))
	#		im_data[0, 0, 0, 0] = 1.
	#	ax = fig.add_subplot(1, len(blur_levels), i+1)
	#	apply_fft_win(im_data, None, windowing=False, plot_ax=ax)
	#	ax.set_title(r'Normalized Power Spectrum $\sigma=${:.2f}'.format(sigma))
#
	#fig.savefig(join(log_dir, 'gauss_response_blur_levels_delta_krange{}_ir1.png'.format(krange)), dpi=300)

	'''
	FFT and IFFT
	'''
	#celeba_data = read_celeba(128, data_size=10)
	#ffts, greys = apply_fft_images(celeba_data, reshape=True)
	#phase = np.angle(ffts)
	#mag = np.random.uniform(0., 8240., ffts.shape) #np.abs(ffts)
	#ffts = mag * np.exp(phase * 1.j)
	#revs = apply_ifft_images(ffts[:, :, :, 0])
	#pyramid_draw([greys, revs, greys-revs], join(log_dir, 'mag_rand_uni_revs.png'))

	'''
	Leakage test
	'''
	#leakage_test(log_dir)

	'''
	Cosine sampler
	'''
	#data_size = 50000
	#freq_centers = [(61/128., 0/128.)]
	#im_size = 128
	#im_data = np.zeros((data_size, im_size, im_size, 1))
	#freq_str = ''
	#for fc in freq_centers:
	#	sampler = COS_Sampler(im_size=im_size, fc_x=fc[0], fc_y=fc[1], channels=1)
	#	im_data += sampler.sample_data(data_size)
	#	freq_str += '_fx{}_fy{}'.format(int(fc[0]*im_size), int(fc[1]*im_size))
	#im_data /= len(freq_centers)
	#true_fft = apply_fft_win(im_data[:1000], 
	#		join(log_dir, 'fft_true{}_size{}'.format(freq_str, im_size)), windowing=False)
	##true_fft_hann = apply_fft_win(im_data[:1000], 
	##		join(log_dir, 'fft_true{}_size{}_hann'.format(freq_str, im_size)), windowing=True)
	#freq_density(true_fft, freq_centers, im_size, join(log_dir, 'freq_density_size{}'.format(im_size)))
	
	'''
	CUB FFT test
	'''
	#cub_dir = '/media/evl/Public/Mahyar/Data/cub/CUB_200_2011/'
	#im_size = 128
	#data_size = 1000
	#test_size = 1000
	#sampler = CUB_Sampler(cub_dir, im_size=im_size, order='test', test_size=test_size)
	#cub_data = sampler.sample_data(data_size)
	#print('>>> CUB shape: {}'.format(cub_data.shape))
	#print('>>> CUB size: {}'.format(sampler.total_count))
	#print('>>> CUB number of classes: {}'.format(1 + np.amax(sampler.cls)))
	#print('>>> CUB average bbox (h, w): ({}, {})'.format(np.mean(sampler.bbox[:, 2]), np.mean(sampler.bbox[:, 3])))
	#im_block_draw(cub_data, 5, log_dir+'/cub{}bb_samples.png'.format(im_size), border=True)
	#apply_fft_win(cub_data, log_dir+'/fft_cub{}bb_hann.png'.format(im_size))
	#sys.exit(0)

	'''
	LSUN FFT test
	'''
	#lsun_lmdb_dir = '/media/evl/Public/Mahyar/Data/lsun/bedroom_train_lmdb/'
	##lsun_lmdb_dir = '/media/evl/Public/Mahyar/Data/lsun/church_outdoor_train_lmdb/'
	#lsun_data, idx_list = create_lsun(lsun_lmdb_dir, resolution=64, max_images=1000)
	#print('>>> LSUN shape: {}'.format(lsun_data.shape))
	#print(idx_list)
	#apply_fft_win(lsun_data, log_dir+'/fft_bedroom64_hann.png')

	'''
	TENSORFLOW SETUP
	'''
	sess = None
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
	config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	TFutil(sess)

	'''
	FFT Test
	'''
	fft_test(log_dir, sess=sess, run_seed=run_seed)


	### close session
	#sess.close()








