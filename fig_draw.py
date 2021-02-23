import numpy as np
import tensorflow as tf
from run_ganist import block_draw, im_block_draw
import sys

#sys.path.insert(1, '/dresden/users/mk1391/evl/Data/stylegan2_model')
#import dnnlib
#import dnnlib.tflib as tflib
#import pretrained_networks

from run_ganist import TFutil, sample_ganist, create_lsun, CUB_Sampler, PG_Sampler, StyleGAN2_Sampler
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import tf_ganist
#import sys
from os.path import join
from util import apply_fft_win, COS_Sampler, freq_density, read_celeba, apply_fft_images, apply_ifft_images, pyramid_draw
from util import eval_toy_exp, mag_phase_wass_dist, mag_phase_total_variation
from util import Logger, readim_path_from_dir, readim_from_path, cosine_eval, create_cosine
from util import make_koch_snowflake, fractal_dimension, fractal_eval
from util import windowing, fft_norm, fft_test_by_samples
from util import fft_corr_eff, fft_corr_point, add_freq_noise, mask_frequency_band
import glob
import os
import pickle as pk
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "3" # "0, 1" for multiple

SMALL_SIZE = 14
MEDIUM_SIZE = 14
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

'''
Drawing Freq Components
'''
def freq_shift(im, fc_x, fc_y, phase=None):
	if phase is None:
		phase = 0
	im_size = im.shape[1]
	kernel_loc = 2.*np.pi*fc_x * np.arange(im_size).reshape((1, im_size, 1)) + \
		2.*np.pi*fc_y * np.arange(im_size).reshape((im_size, 1, 1))
	kernel_cos = np.cos(kernel_loc + phase)
	return im * kernel_cos

def draw_freq_comps(log_dir, comp_size, freqs, num=10):
	im = np.ones((comp_size, comp_size, 3))
	for f in freqs:
		freq_x = np.linspace(-f, f, num)
		freq_y = np.sqrt(f**2 - freq_x**2)
		freq_comps = list()
		for fx, fy in zip(freq_x, freq_y):
			im_sh = freq_shift(im, fx/comp_size, fy/comp_size)
			freq_comps.append(im_sh)
		freq_comps = np.array(freq_comps).reshape((-1, 1, comp_size, comp_size, 3))
		block_draw(freq_comps, os.path.join(log_dir, f'freq_comps_{f}.png'), border=True)

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

def fft_power_diff_test(log_dir):
	eps = 1e-10
	paths = glob.glob(os.path.join(log_dir, 'run_*/'))
	log_power_diff = list()
	log_power_diff_hann = list()
	for p in paths:
		pk_paths = glob.glob(os.path.join(p, '*.pk'))
		for pkp in pk_paths:
			if 'fft' in pkp:
				with open(pkp, 'rb') as fs:
					if 'mean' in pkp:
						fft_mean = pk.load(fs)
					else:
						fft_mean = np.mean(pk.load(fs), axis=0)
				if 'true' in pkp:
					if 'hann' in pkp:
						true_fft_mean_hann = fft_mean
					else:
						true_fft_mean = fft_mean
				else:
					if 'hann' in pkp:
						gen_fft_mean_hann = fft_mean
					else:
						gen_fft_mean = fft_mean
		log_power_diff.append(np.sum(np.abs(np.log(true_fft_mean+eps) - np.log(gen_fft_mean+eps))))
		log_power_diff_hann.append(np.sum(np.abs(np.log(true_fft_mean_hann+eps) - np.log(gen_fft_mean_hann+eps))))
	with open(os.path.join(log_dir, 'log_power_diff.txt'), 'w+') as fs:
		print(f'log_power_diff: {np.mean(log_power_diff)} SD {np.std(log_power_diff)}', file=fs)
		print(f'log_power_diff_hann: {np.mean(log_power_diff_hann)} SD {np.std(log_power_diff_hann)}', file=fs)
	return

def read_model_layers(log_dir, sess=None, run_seed=0, data_size=25):
	### read ganist network
	g_name = '5_logs_wganbn_celeba128cc_fid50'
	#g_name = '14_logs_wganbn_celeba128cc_fssetup_fshift'
	#g_name = '38_logs_wganbn_bedroom128cc'
	#g_name = '46_logs_wgan_sbedroom128cc_hpfid'
	#g_name = '22_logs_wganbn_gshift_celeba128cc_fshift'
	#g_name = '49_logs_wgan_gshift_sbedroom128cc_hpfid'
	ganist = tf_ganist.Ganist(sess, log_dir)
	sess.run(tf.global_variables_initializer())
	net_path = f'/dresden/users/mk1391/evl/ganist_lap_logs/{g_name}/run_{run_seed}/snapshots/model_best.h5'
	ganist.load(net_path)
	g_samples = sample_ganist(ganist, data_size, output_type='rec')[0]
	d_vars, g_vars = ganist.get_vars_array()

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
	im_block_draw(g_samples, 5, join(log_dir, f'layer_reader_samples_{g_name}.png'), border=True)
	apply_fft_win(g_samples[0:1,:,:,0:], join(log_dir, f'model_reader_fft_avg_{g_name}.png'), windowing=True, plot_ax=None, drop_dc=True)

	def box_upsample(im):
		h, w = im.shape
		return np.repeat(np.repeat(im.reshape((h, 1, w, 1)), 2, axis=3), 2, axis=1).reshape(h*2, w*2)

	fig = plt.figure(0, figsize=(8,6))
	im_size = 128
	layer_size_dict = {'conv0': 16, 'conv1': 32, 'conv2': 64, 'conv3': 128}
	model_layers = list()
	model_layers_name = list()
	model_layers_size = list()
	for val, name in g_vars:
		if 'kernel' in name and 'conv' in name:
			scopes = name.split('/')
			net_name = next(v for v in scopes if 'net' in  v)
			conv_name = next(v for v in scopes if 'conv' in  v)
			full_name = '-'.join(scopes[:-1])
			#save_path = '{}/{}_{}'.format(save_dir, net_name, conv_name)
			save_path = '{}/{}'.format(log_dir, full_name)
			layer_name = '{}_{}_'.format(net_name, conv_name) + '_'.join(map(str, val.shape))
			layer_size = layer_size_dict[conv_name]
			print(f'>>> name: {name}')
			print(f'>>> save_path: {save_path}')
			print(f'>>> layer_name: {layer_name}')
			print(f'>>> layer_size: {layer_size}')
			layer_filter = np.zeros((val.shape[2], val.shape[3], layer_size, layer_size), dtype=complex)
			fft_avg_full = 1e-20 + np.zeros((im_size, im_size))
			for i in range(val.shape[2]):
				for j in range(val.shape[3]):
					im = np.zeros((layer_size, layer_size))
					im[:val.shape[0], :val.shape[1]] = val[:, :, i, j]
					#for _ in range(int(np.log2(im_size / layer_size))): 
					#	im = box_upsample(im)
					layer_filter[i, j, ...] = im

			model_layers.append(layer_filter)
			model_layers_name.append(layer_name)
			model_layers_size.append(layer_size)
	return model_layers, model_layers_name, model_layers_size

def fft_layer_test(log_dir, sess=None, run_seed=0, data_size=25):
	### read ganist network
	g_name = '5_logs_wganbn_celeba128cc_fid50'
	#g_name = '14_logs_wganbn_celeba128cc_fssetup_fshift'
	#g_name = '38_logs_wganbn_bedroom128cc'
	#g_name = '46_logs_wgan_sbedroom128cc_hpfid'
	#g_name = '22_logs_wganbn_gshift_celeba128cc_fshift'
	#g_name = '49_logs_wgan_gshift_sbedroom128cc_hpfid'
	ganist = tf_ganist.Ganist(sess, log_dir)
	sess.run(tf.global_variables_initializer())
	net_path = f'/dresden/users/mk1391/evl/ganist_lap_logs/{g_name}/run_{run_seed}/snapshots/model_best.h5'
	ganist.load(net_path)
	g_samples = sample_ganist(ganist, data_size, output_type='rec')[0]
	d_vars, g_vars = ganist.get_vars_array()

	block_draw(g_samples[:data_size].reshape(
		(int(np.ceil(np.sqrt(data_size))), int(np.ceil(np.sqrt(data_size))))+g_samples.shape[1:]), 
		os.path.join(log_dir, f'fft_layer_samples.png'))

	def box_upsample(im):
		h, w = im.shape
		return np.repeat(np.repeat(im.reshape((h, 1, w, 1)), 2, axis=3), 2, axis=1).reshape(h*2, w*2)

	fig = plt.figure(0, figsize=(8,6))
	im_size = 128
	layer_size_dict = {'conv0': 16, 'conv1': 32, 'conv2': 64, 'conv3': 128}
	for val, name in g_vars:
		if 'kernel' in name and 'conv' in name:
			scopes = name.split('/')
			net_name = next(v for v in scopes if 'net' in  v)
			conv_name = next(v for v in scopes if 'conv' in  v)
			full_name = '-'.join(scopes[:-1])
			#save_path = '{}/{}_{}'.format(save_dir, net_name, conv_name)
			save_path = '{}/{}'.format(log_dir, full_name)
			layer_name = '{}_{}_'.format(net_name, conv_name) + '_'.join(map(str, val.shape))
			layer_size = layer_size_dict[conv_name]
			print(f'>>> name: {name}')
			print(f'>>> save_path: {save_path}')
			print(f'>>> layer_name: {layer_name}')
			print(f'>>> layer_size: {layer_size}')
			fft_mat = np.zeros((val.shape[2], val.shape[3], layer_size, layer_size), dtype=complex)
			fft_avg_full = 1e-20 + np.zeros((im_size, im_size))
			for i in range(val.shape[2]):
				for j in range(val.shape[3]):
					im = np.zeros((layer_size, layer_size))
					im[:val.shape[0], :val.shape[1]] = val[:, :, i, j]
					#for _ in range(int(np.log2(im_size / layer_size))): 
					#	im = box_upsample(im)
					fft_mat[i, j, ...], _ = apply_fft_images(im[np.newaxis, :, :, np.newaxis], reshape=False)
			fft_power = np.abs(fft_mat)**2
			fft_avg = np.mean(fft_power, axis=(0, 1))
			print('>>> FFT MIN: {}'.format(fft_avg.min()))
			print('>>> FFT MAX: {}'.format(fft_avg.max()))
			np.clip(fft_avg, 1e-20, None, out=fft_avg)
			fft_avg_full[:layer_size, :layer_size] = fft_avg

			### plot mean fft
			fig.clf()
			ax = fig.add_subplot(1,1,1)
			pa = ax.imshow(np.log(fft_avg_full / np.amax(fft_avg_full)), cmap=plt.get_cmap('inferno'), vmin=-13, vmax=0)
			ax.set_title(layer_name)
			ticks_loc_x = [0, im_size//2]
			ticks_loc_y = [0, im_size//2-1, im_size-im_size%2-1]
			ax.set_xticks(ticks_loc_x)
			ax.set_xticklabels([-0.5, 0])
			ax.set_yticks(ticks_loc_y)
			ax.set_yticklabels(['', 0, -0.5])
			plt.colorbar(pa)#, ax=ax, fraction=0.046, pad=0.04)
			fig.savefig(os.path.join(log_dir, f'fft_{layer_name}.png'), dpi=300)

	with open(os.path.join(log_dir, 'layer_g_vars.cpk'), 'wb+') as fs:
		pk.dump(g_vars, fs)

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

	im_block_draw(r_samples, 5, join(log_dir, f'fft_test_{r_name}_samples.png'), border=True)
	im_block_draw(g_samples, 5, join(log_dir, f'fft_test_{g_name}_{r_name}_samples.png'), border=True)
	im_block_draw(shifter(r_samples), 5, join(log_dir, f'fft_test_{r_name}_samples_sh.png'), border=True)
	im_block_draw(shifter(g_samples), 5, join(log_dir, f'fft_test_{g_name}_{r_name}_samples_sh.png'), border=True)
	#with open(join(log_dir, f'{g_name}_samples.pk'), 'wb+') as fs:
	#	pk.dump(g_samples, fs)

	fft_test_by_samples(log_dir, r_samples, g_samples, r_name, g_name)

def run_cos_eval(log_dir, sess=None, run_seed=0):
	data_size = 10000
	im_size = 128

	### craete cosine dataset
	freq_centers = [(61/128., 61/128.)]
	r_samples, freq_str = create_cosine(data_size, freq_centers, resolution=im_size, channels=1)

	### pggan load g_samples
	sys.path.insert(1, '/dresden/users/mk1391/evl/Data/pggan_model')
	g_name = f'gdsmall_results_{run_seed}'
	#net_path = f'/dresden/users/mk1391/evl/pggan_logs/logs_celeba128cc/{g_name}/000-pgan-celeba-preset-v2-2gpus-fp32/network-snapshot-010211.pkl'
	#net_path = f'/dresden/users/mk1391/evl/pggan_logs/logs_bedroom128cc/{g_name}/000-pgan-lsun-bedroom-preset-v2-2gpus-fp32/network-snapshot-010211.pkl'
	#g_name = f'results_gdsmall_outsh_nomirror_sceleba_{run_seed}'
	#net_path = f'/dresden/users/mk1391/evl/pggan_logs/logs_celeba128cc_sh/{g_name}/000-pgan-celeba-preset-v2-2gpus-fp32/network-snapshot-010211.pkl'
	#g_name = f'results_gdsmall_sbedroom_{run_seed}'
	#g_name = f'results_gdsmall_outsh_nomirror_sbedroom_{run_seed}'
	#net_path = f'/dresden/users/mk1391/evl/pggan_logs/logs_bedroom128cc_sh/{g_name}/000-pgan-lsun-bedroom-preset-v2-2gpus-fp32/network-snapshot-010211.pkl'
	net_path = '/dresden/users/mk1391/evl/pggan_logs/logs_cos/results_gdsmall_cos_fx61fy61_size128_mnorm01p0pi_0/000-pgan-cosine-preset-v2-2gpus-fp32/network-snapshot-010211.pkl'
	pg_sampler = PG_Sampler(net_path, sess, net_type='tf')
	g_samples = pg_sampler.sample_data(data_size)

	true_fft, true_fft_hann, true_hist = cosine_eval(r_samples, 'true', freq_centers, log_dir=log_dir)
	cosine_eval(g_samples, 'gen', freq_centers, log_dir=log_dir, true_fft=true_fft, true_fft_hann=true_fft_hann, true_hist=true_hist)

def read_model_samples(log_dir, sess=None, run_seed=0, data_size=1000, im_size=128):
	use_shifter = True ## set to True for freq shift dataset
	def shifter(x): return TFutil.get().freq_shift(x, 0.5, 0.5) if use_shifter else x	

	### GANIST load g_samples
	#g_name = 'init'
	g_name = '5_logs_wganbn_celeba128cc_fid50'
	#g_name = '14_logs_wganbn_celeba128cc_fssetup_fshift'
	#g_name = '38_logs_wganbn_bedroom128cc'
	#g_name = '46_logs_wgan_sbedroom128cc_hpfid'
	#g_name = '22_logs_wganbn_gshift_celeba128cc_fshift'
	#g_name = '49_logs_wgan_gshift_sbedroom128cc_hpfid'
	#ganist = tf_ganist.Ganist(sess, log_dir)
	#sess.run(tf.global_variables_initializer())
	#net_path = f'/dresden/users/mk1391/evl/ganist_lap_logs/{g_name}/run_{run_seed}/snapshots/model_best.h5'
	#ganist.load(net_path)
	#g_samples = sample_ganist(ganist, data_size, output_type='rec')[0]
	#g_samples = sample_ganist(ganist, data_size, output_type='collect')

	### PGGAN load g_samples
	#sys.path.insert(1, '/dresden/users/mk1391/evl/Data/pggan_model')
	g_name = f'results_org_koch_snowflakes_l5_s1024'
	net_path = f'/dresden/users/mk1391/evl/pggan_logs/logs_koch_1024_l5/{g_name}/000-pgan-fractal-preset-v2-4gpus-fp32/network-snapshot-010004.pkl'
	#g_name = f'gdsmall_results_{run_seed}'
	#net_path = f'/dresden/users/mk1391/evl/pggan_logs/logs_celeba128cc/{g_name}/000-pgan-celeba-preset-v2-2gpus-fp32/network-snapshot-010211.pkl'
	#net_path = f'/dresden/users/mk1391/evl/pggan_logs/logs_bedroom128cc/{g_name}/000-pgan-lsun-bedroom-preset-v2-2gpus-fp32/network-snapshot-010211.pkl'
	#g_name = f'results_gdsmall_outsh_nomirror_sceleba_{run_seed}'
	#net_path = f'/dresden/users/mk1391/evl/pggan_logs/logs_celeba128cc_sh/{g_name}/000-pgan-celeba-preset-v2-2gpus-fp32/network-snapshot-010211.pkl'
	#g_name = f'results_gdsmall_sbedroom_{run_seed}'
	#g_name = f'results_gdsmall_outsh_nomirror_sbedroom_{run_seed}'
	#net_path = f'/dresden/users/mk1391/evl/pggan_logs/logs_bedroom128cc_sh/{g_name}/000-pgan-lsun-bedroom-preset-v2-2gpus-fp32/network-snapshot-010211.pkl'
	pg_sampler = PG_Sampler(net_path, sess, net_type='tf')
	g_samples = pg_sampler.sample_data(data_size)

	### StyleGAN2 load g_samples
	#g_name = f'results_sg_koch_l5_s1024'
	#net_path = f'/dresden/users/mk1391/evl/stylegan2_logs/logs_koch_1024_l5/{g_name}/00000-stylegan2-koch-8gpu-config-e/network-snapshot-010000.pkl'
	#g_name = f'results_sg_small_fsg_finalstylemix_celeba128cc_{run_seed}'
	#net_path = f'/dresden/users/mk1391/evl/stylegan2_logs/logs_celeba128cc/{g_name}/00000-stylegan2-celeba-4gpu-config-e/network-final.pkl'
	#g_name = f'results_sg_small_celeba128cc_{run_seed}'
	#net_path = f'/dresden/users/mk1391/evl/stylegan2_logs/logs_celeba128cc/{g_name}/00000-stylegan2-celeba-4gpu-config-e/network-final.pkl'
	#g_name = f'results_sg_small_bedroom128cc_{run_seed}'
	#net_path = f'/dresden/users/mk1391/evl/stylegan2_logs/logs_bedroom128cc/{g_name}/00000-stylegan2-lsun-bedroom-100k-4gpu-config-e/network-final.pkl'
	#sty_sampler = StyleGAN2_Sampler(net_path, sess)
	#g_samples = sty_sampler.sample_data(data_size)

	### load g_samples from pickle file
	#with open(join(log_dir, f'{g_name}_samples_{data_size}.pk'), 'rb') as fs:
	#	g_samples = pk.load(fs)

	### load g_samples from image file
	#g_name = 'logs_ganms_or_celeba128cc'
	#g_sample_dir = f'/dresden/users/mk1391/evl/ganist_lsun_logs/layer_stats/temp/{g_name}/run_0/samples/'
	#g_samples = readim_from_path(
	#	readim_path_from_dir(g_sample_dir, im_type='*.jpg')[:data_size], im_size, center_crop=(64,64), verbose=True)

	im_block_draw(g_samples, 5, join(log_dir, f'sample_reader_{g_name}.png'), border=True)
	im_block_draw(shifter(g_samples), 5, join(log_dir, f'sample_reader_{g_name}_sh.png'), border=True)
	#apply_fft_win(g_samples[1][0:1,:,:,0:], join(log_dir, f'sample_reader_fft_impulse_1_avg_{g_name}.png'), windowing=True, plot_ax=None, drop_dc=True)
	#apply_fft_win(g_samples[2][0:1,:,:,0:], join(log_dir, f'sample_reader_fft_impulse_2_avg_{g_name}.png'), windowing=True, plot_ax=None, drop_dc=True)
	#apply_fft_win(g_samples[3][0:1,:,:,0:], join(log_dir, f'sample_reader_fft_impulse_3_avg_{g_name}.png'), windowing=True, plot_ax=None, drop_dc=True)
	#apply_fft_win(g_samples[4][0:1,:,:,0:], join(log_dir, f'sample_reader_fft_impulse_4_avg_{g_name}.png'), windowing=True, plot_ax=None, drop_dc=True)
	#with open(join(log_dir, f'{g_name}_samples.pk'), 'wb+') as fs:
	#	pk.dump(g_samples, fs)

	return g_samples

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
	os.makedirs(log_dir, exist_ok=True)
	sys.stdout = Logger(log_dir)
	sys.stderr = sys.stdout
	sess = None

	'''
	Draw frequency components
	'''
	#draw_freq_comps(log_dir, comp_size=128, freqs=[4, 8, 16], num=10)

	'''
	TENSORFLOW SETUP
	'''
	#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
	#config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
	#config.gpu_options.allow_growth = True
	#sess = tf.Session(config=config)
	#TFutil(sess)
	
	'''
	Model effective correlation
	'''
	#def stat_corr(corr, corr_rev):
	#corr_mean = list()
	#corr_sd = list()
	#for c, cr in zip(corr, corr_rev):
	#	c_total = np.concatenate([c, cr])
	#	corr_mean.append(np.mean(c_total))
	#	corr_sd.append(np.std(c_total))
	#return corr_mean, corr_sd

	#im_size = 128
	#data_size = 100
	##corr_eff_per_layer = list()
	#freq_bands = np.array([16, 32, 64, 128]) // 2
	#g_samples = read_model_samples(log_dir, sess, run_seed, data_size)
	
	#corr_eff_per_layer = fft_corr_eff(g_samples[1], freq_bands)
	#print(f'corr eff per layer: {corr_eff_per_layer}')
	#corr_g_mean = list()
	#corr_g_sd = list()
	#for gi, g in enumerate(g_samples[1:]):
	#	fb = freq_bands[:1] if gi == 0 else freq_bands[gi-1:gi+1]
	#	corr, corr_rev = fft_corr_point(np.transpose(g,(0, 3, 1, 2)).reshape(-1, im_size, im_size), freq_bands=fb)
	#	corr_mean, corr_sd = stat_corr(corr, corr_rev)
	#	print(f'>>> layer eff corr at freq {freq_bands[gi]}: min={np.amin(corr[-1])} min_rev={np.amin(corr_rev[-1])} max={np.amax(corr[-1])} max_rev={np.amax(corr_rev[-1])}')
	#	#corr_eff_per_layer.append(fft_corr_eff(g, fb)[-1])
	#	corr_g_mean.append(corr_mean[-1])
	#	corr_g_sd.append(corr_sd[-1])
	#	print(f'>>> layer eff corr at freq {freq_bands[gi]}: {corr_mean} sd {corr_sd}')
	#	gi_pre = gi
	
	#r_samples = read_celeba(im_size, data_size)
	#fft_corr_g, fft_corr_g_rev = fft_corr_point(np.transpose(g_samples,(0, 3, 1, 2)).reshape(-1, im_size, im_size), freq_bands)
	#fft_corr_r, fft_corr_r_rev = fft_corr_point(np.transpose(r_samples,(0, 3, 1, 2)).reshape(-1, im_size, im_size), freq_bands)
	

	#corr_g_mean, corr_g_sd = stat_corr(fft_corr_g, fft_corr_g_rev)
	#print('>>> gen im corr mean: ', corr_g_mean)
	#print('>>> gen im corr sd: ', corr_g_sd)
	#corr_r_mean, corr_r_sd = stat_corr(fft_corr_r, fft_corr_r_rev)
	#print('>>> true im corr mean: ', corr_r_mean)
	#print('>>> true im corr sd: ', corr_r_sd)

	'''
	Model layer correlation
	'''
	#im_size = 128
	#data_size = 1000
	#layers, layer_names, layer_sizes = read_model_layers(log_dir, sess, run_seed, data_size=25)
	#print(f'>>> layer names: {layer_names}')
	#corr_layers_mean = list()
	#corr_layers_sd = list()
	#for l, n, s in zip(layers, layer_names, layer_sizes):
	#	l = l.reshape((np.prod(l.shape[:2]), l.shape[2], l.shape[3]))
	#	fft_corr_l, fft_corr_l_rev = fft_corr_point(l, freq_bands=None)
	#	corr_l_mean, corr_l_sd = stat_corr(fft_corr_l, fft_corr_l_rev)
	#	print(f'>>> at layer {n} with size {s} corr mean: ', corr_l_mean)
	#	print(f'>>> at layer {n} with size {s} corr sd: ', corr_l_sd)
	#	corr_layers_mean.append(corr_l_mean[0])
	#	corr_layers_sd.append(corr_l_sd[0])

	#	corr_layers.append(fft_corr_eff(l, freq_bands=None)[0])
	#	print(f'>>> corr_layers at layer {n} with size {s} is equal to {corr_layers[-1]}')

	'''
	Theorem
	'''
	#dk_val = 5
	#im_size = 128
	#delta = 1
	#assert delta != 0
	#dl = np.arange(dk_val, im_size+1)
	##dl = np.array([16, 32, 64, 128])
	#corr_true = np.zeros(dl.size)
	#def sinc_conv(sinc, d): return np.array([sinc[d-i] if d-i >= 0 else sinc[-(d-i)] for i in range(sinc.size)])
	#for i, dl_val in enumerate(dl):
	#	corr_true[i] = np.sin(np.pi*delta*dk_val/dl_val)**2 / (dk_val *np.sin(np.pi*delta/dl_val))**2
	#fig = plt.figure(0, figsize=(8,6))
	#fig.clf()
	#ax = fig.add_subplot(1,1,1)
	#ax.plot(dl, corr_true, linestyle='--', label='Theorem')
	##dl_specific = np.array(layer_sizes)
	#dl_specific = np.array([16, 32, 64, 128])
	#ax.plot(dl_specific, corr_true[dl_specific - dk_val], linestyle='None', marker='s', color='orange', label='WGAN-GP')
	##ax.errorbar(dl_specific, corr_layers_mean, corr_layers_sd, linestyle='None', marker='s', label='WGAN-GP', capsize=5)
	#ax.grid(True, which='both', linestyle='dotted')
	#ax.set_ylabel('corr')
	#ax.set_xlabel(r'Layer Spatial Size ($d_l$)')
	#ax.set_xticks([dk_val,] + list(dl_specific))
	#ax.set_xticklabels(map(str, [dk_val,] + list(dl_specific)))
	#ax.legend(loc=4)
	#fig.savefig(join(log_dir, f'theorem_true_d{delta}_k{dk_val}_s{im_size}.png'), dpi=300)

	#freq_bands = np.array([16, 32, 64, 128]) // 2
	#num_layers = freq_bands.size
	#corr_eff = np.zeros(num_layers)
	#k_eff = dk_val
	#for i in range(corr_eff.size, 0, -1):
	#	corr_eff[i-1] = np.sin(np.pi*delta*k_eff/im_size)**2 / (k_eff * np.sin(np.pi*delta/im_size))**2
	#	print(f'k_eff={k_eff}, corr_eff={corr_eff[i-1]}')
	#	k_eff += 2**(corr_eff.size - i + 1) * (dk_val - 1)
	#fig.clf()
	#ax = fig.add_subplot(1,1,1)
	#ax.plot(np.arange(num_layers), corr_eff, linestyle='--', label='Theorem')
	#ax.errorbar(np.arange(num_layers), corr_g_mean, corr_g_sd, linestyle='None', marker='s', label='WGAN-GP', capsize=5)
	##ax.errorbar(np.arange(num_layers), corr_r_mean, corr_r_sd, linestyle='None', marker='s', color='blue', label='True')
	#ax.grid(True, which='both', linestyle='dotted')
	#ax.set_ylabel('corr')
	#ax.set_xlabel('Frequency Band')
	#ax.set_xticks(range(num_layers))
	#freq_bands = [0, ] + list(dl_specific//2)
	#ax.set_xticklabels(map(lambda x: r'[$\frac{{{}}}{{128}}$, $\frac{{{}}}{{128}}$)'.format(*x), zip(freq_bands[:-1], freq_bands[1:])))
	##ax.set_xticklabels(map(lambda x: r'[{}, {}) / 128'.format(*x), zip(freq_bands[:-1], freq_bands[1:])))
	#fig.savefig(join(log_dir, f'theorem_eff_d{delta}_k{dk_val}_s{im_size}.png'), dpi=300)


	'''
	Power Diff Eval
	'''
	#eval_dir = '/dresden/users/mk1391/evl/ganist_toy_logs/17_logs_wgan_cos128_fx0y0_mnorm01p0pi'
	#eval_dir = '/dresden/users/mk1391/evl/ganist_toy_logs/36_logs_wgan_cos128_fx3y3_mnorm01p0pi'
	#eval_dir = '/dresden/users/mk1391/evl/ganist_toy_logs/38_logs_wgan_cos128_fx61y61_mnorm01p0pi'
	#eval_dir = '/dresden/users/mk1391/evl/ganist_toy_logs/19_logs_wgan_cos128_fx64y64_mnorm01p0pi'
	#fft_power_diff_test(eval_dir)

	'''
	Koch Snowflakes
	'''
	#data_size = 1000
	#im_size = 1024
	#koch_level = 5
	#channels = 1
	
	#im = make_koch_snowflake(koch_level, 0., im_size, channels)
	#def make_single_image():
	#	im = -np.ones((im_size, im_size, channels))
	#	im[0, 0, :] = 1.
	#	return im
	##im = make_single_image()
	#coeffs, sizes, counts = fractal_dimension(im[:,:,0], threshold=0.)
	#box_dim = -coeffs[0]
	#print(coeffs)
	#fig = plt.figure(0, figsize=(8,6))
	#fig.clf()
	#ax = fig.add_subplot(1,1,1)
	#ax.plot(-np.log(np.array(sizes)), np.log(np.array(counts)), 's-b')
	#ax.plot(-np.log(np.array(sizes)), coeffs[0] * np.log(np.array(sizes)) + coeffs[1], '-r')
	#ax.grid(True, which='both', linestyle='dotted')
	#fig.savefig(join(log_dir, f'box_count_dim_{koch_level}_{im_size}.png'), dpi=300)
	#single_draw(im, os.path.join(log_dir, f'koch_snowflake_{koch_level}_{im_size}.png'))
	#apply_fft_win((im[np.newaxis, ...] + 1.) / 2., #- np.mean(im), 
	#	os.path.join(log_dir, f'fft_koch_snowflake_single_{koch_level}_{im_size}.png'), windowing=False)
	#print(f'>>> single image: box_dim={box_dim} and path_length={np.sum(im > 0.)}')
	#
	
	### many samples
	#im_data = np.zeros((data_size, im_size, im_size, channels))
	#for i in range(data_size):
	#	rot = np.random.uniform(-30, 30)
	#	im_data[i, ...] = make_koch_snowflake(koch_level, rot, im_size, channels)

	### read samples from models and eval
	#gname = 'pggan'
	#im_data = read_model_samples(log_dir, sess, run_seed, data_size)
	
	#for i, im in enumerate(im_data[:10]):
	#	single_draw(im, os.path.join(log_dir, f'koch_sample_{gname}_{i}.png'))
	#	apply_fft_win(im_data[i:i+1,:,:,0:1], join(log_dir, f'koch_fft_{gname}_{i}.png'), windowing=True, drop_dc=True)


	#im_block_draw(im_data, 5, join(log_dir, f'koch_snowflake_{gname}_{koch_level}_{im_size}_samples.png'), border=True)
	#fractal_eval(im_data, f'koch_snowflake_{gname}_{koch_level}_{im_size}', log_dir)

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
	1D Effect of Upsampling and Sigma
	'''
	#dsize = 128
	#ksize = 4
	#def triangle_1d(center, length):
	#	tri = np.arange(1, length+1, dtype=np.float) / length
	#	return np.concatenate((tri, tri[::-1][1:]))
	#	
	#def fft_1d(signal):
	#	fft = np.fft.fft(signal)
	#	fft = np.fft.fftshift(fft)
	#	return fft
	#	
	#def ifft_1d(signal):
	#	signal = np.fft.ifftshift(signal)
	#	ifft = np.fft.ifft(signal)
	#	return ifft
	#	
	#triangles = [(dsize//2, ksize, 10), (dsize//2-3*ksize, ksize, 8), (dsize//2+3*ksize, ksize, 8)]
	##triangles = [(dsize//2, ksize, 10)]
	#signal = np.zeros(dsize, dtype=np.float)
	##signal[dsize//2-ksize] = 10.
	##signal[dsize//2+ksize] = 10.
	#for c, l, m in triangles:
	#	#signal[c-l+1:c+l] = m * triangle_1d(c, l)
	#	signal[c-l+1:c+l] = m
	#	
	##signal = np.cos(2*np.pi/16*np.arange(dsize))
	##signal = 1-signal
	#
	#fig = plt.figure(0, (8, 9))
	#ax = fig.add_subplot(3, 1, 1)
	#ax.grid(True, which='both', linestyle='dotted')
	#ax.set_xlabel(r'Frequency')
	#ax.plot(range(-dsize//2, dsize//2), signal)
	#x_ticks_loc = [-dsize//2] + [c-dsize//2 for c, l, m in triangles] + [dsize//2]
	#ax.set_xticks(x_ticks_loc)
	#ax.set_xticklabels(map('{}'.format, x_ticks_loc))
	#
	#ifft = np.real(ifft_1d(signal))
	#### Relu
	#ifft = np.clip(ifft, 0, None)
	#ifft = np.concatenate([ifft[..., np.newaxis], np.zeros((ifft.shape[0], 1))], axis=1).reshape(dsize*2)
	#dsize *= 2
	#fft = fft_1d(ifft)
	#
	#ax = fig.add_subplot(3, 1, 2)
	#ax.grid(True, which='both', linestyle='dotted')
	#ax.set_xlabel(r'Location')
	#ax.plot(ifft)
	##ax.plot(range(-dsize//2, dsize//2), np.concatenate([ifft[dsize//2:], ifft[:dsize//2]]))
	#x_ticks_loc = range(0, dsize+1, dsize//8)
	#ax.set_xticks(x_ticks_loc)
	#ax.set_xticklabels(map('{}'.format, x_ticks_loc))
	#
	#ax = fig.add_subplot(3, 1, 3)
	#ax.grid(True, which='both', linestyle='dotted')
	#ax.set_xlabel(r'Frequency')
	#ax.plot(range(-dsize//2, dsize//2), np.real(fft))
	#x_ticks_loc = [-dsize//2] + [c-dsize//4 for c, l, m in triangles] + [dsize//2]
	#ax.set_xticks(x_ticks_loc)
	#ax.set_xticklabels(map('{}'.format, x_ticks_loc))
	#
	#fig.tight_layout()
	#fig.savefig(join(log_dir, f'rect_1d_relu_up.png'), dpi=300)

	'''
	FFT and IFFT
	'''
	#draw_size = 10
	#data_size = 100
	#celeba_data = read_celeba(128, data_size=data_size)
	#ffts, greys = apply_fft_images(celeba_data, reshape=True)
	##phase = np.angle(ffts)
	##mag = np.abs(ffts) #np.random.uniform(0., 8240., ffts.shape)
	##ffts = mag * np.exp(phase * 1.j)
	#revs = apply_ifft_images(ffts[:, :, :, 0])
	#diffs = greys - revs
	#pyramid_draw([greys[:draw_size], revs[:draw_size], diffs[:draw_size]], join(log_dir, 'mag_rand_uni_revs.png'))
	#print(f'>>> fft_ifft_diff_[-1,1]: {np.mean(np.abs(diffs))} sd {np.std(np.abs(diffs))}')
	#
	#def dynamic_range_255(im):
	#	im = (im + 1.0) / 2.0 * 255.
	#	return np.clip(np.rint(im * 255.0), 0.0, 255.0)
	#
	#diffs_int = dynamic_range_255(greys) - dynamic_range_255(revs)
	#print(f'>>> fft_ifft_diff_rint_[0,255]: {np.mean(np.abs(diffs_int))} sd {np.std(np.abs(diffs_int))}')

	'''
	High vs Low frequency of image
	'''
	draw_size = 10
	freq_low = 1/8
	freq_high = 1
	ims = read_celeba(128, data_size=draw_size)
	b, h, w, c = ims.shape
	_, mask = add_freq_noise(ims[0], low=freq_low, high=freq_high, channels=1)
	ims_re = np.zeros(ims.shape)
	for i, im in enumerate(ims):
		ims_re[i] = mask_frequency_band(im, mask)

	ims = np.array(ims).reshape((-1, 1, h, w, 3))
	block_draw(ims, os.path.join(log_dir, f'freq_masked_images_no_mask.png'), border=True)

	ims_re = np.array(ims_re).reshape((-1, 1, h, w, 3))
	block_draw(ims_re, os.path.join(log_dir, f'freq_masked_images_low{freq_low:.2f}_high{freq_high:.2f}.png'), border=True)

	'''
	Add spectral noise test
	'''
	#draw_size = 10
	#data_size = 10
	#c = 3
	#celeba_data = read_celeba(128, data_size=data_size)
	#noise_low = np.zeros(celeba_data.shape[:-1]+(c,))
	#noise_high = np.zeros(celeba_data.shape[:-1]+(c,))
	#noise_full = np.zeros(celeba_data.shape[:-1]+(c,))
	#noise_scale = 0.2
	#mask_low = None
	#mask_high = None
	#mask_full = None
	#for i, im in enumerate(celeba_data):
	#	print(f'>>> processing image {i}')
	#	noise_low[i], mask_low = add_freq_noise(im, low=0, high=1/8, scale=noise_scale, mask=mask_low, channels=c)
	#	noise_high[i], mask_high = add_freq_noise(im, low=1/8, high=None, scale=noise_scale, mask=mask_high, channels=c)
	#	noise_full[i], mask_full = add_freq_noise(im, low=0, high=None, scale=noise_scale, mask=mask_full, channels=c)
	#
	#apply_fft_win(celeba_data, os.path.join(log_dir, 'fft_celeba.png'))
	#apply_fft_win(celeba_data+noise_low, os.path.join(log_dir, f'fft_celeba_noise_scale{noise_scale}_low.png'))
	#apply_fft_win(celeba_data+noise_high, os.path.join(log_dir, f'fft_celeba_noise_scale{noise_scale}_high.png'))
	#apply_fft_win(celeba_data+noise_full, os.path.join(log_dir, f'fft_celeba_noise_scale{noise_scale}_full.png'))
	#apply_fft_win(noise_low, os.path.join(log_dir, f'fft_noise_scale{noise_scale}_low.png'))
	#apply_fft_win(noise_high, os.path.join(log_dir, f'fft_noise_scale{noise_scale}_high.png'))
	#apply_fft_win(noise_full, os.path.join(log_dir, f'fft_noise_scale{noise_scale}_full.png'))
	#pyramid_draw([celeba_data,
	#	np.broadcast_to(mask_low[..., np.newaxis], celeba_data.shape), 
	#	np.repeat(noise_low, celeba_data.shape[-1]//c, axis=3),
	#	celeba_data+noise_low,
	#	np.broadcast_to(mask_high[..., np.newaxis], celeba_data.shape),
	#	np.repeat(noise_high, celeba_data.shape[-1]//c, axis=3),
	#	celeba_data+noise_high,
	#	np.broadcast_to(mask_full[..., np.newaxis], celeba_data.shape),
	#	np.repeat(noise_full, celeba_data.shape[-1]//c, axis=3),
	#	celeba_data+noise_full], os.path.join(log_dir, 'noisy_celeba_samples.png'))

	'''
	Leakage test
	'''
	#leakage_test(log_dir)

	'''
	Cosine sampler
	'''
	#data_size = 10000
	#freq_centers = [(0/128., 0/128.)]
	#im_size = 128
	#im_data = np.zeros((data_size, im_size, im_size, 1))
	#freq_str = ''
	#for fc in freq_centers:
	#	sampler = COS_Sampler(im_size=im_size, fc_x=fc[0], fc_y=fc[1], channels=1)
	#	im_data += sampler.sample_data(data_size)
	#	freq_str += '_fx{}_fy{}'.format(int(fc[0]*im_size), int(fc[1]*im_size))
	#im_data /= len(freq_centers)
	#im_data = 255. * (im_data + 1.) / 2.
	#im_data = im_data.astype(np.float32)
	#im_data = np.rint(im_data).clip(0, 255).astype(np.uint8)
	#im_data = im_data / 255. * 2. - 1.
	#true_fft, true_fft_hann, true_hist = cosine_eval(im_data, 'true', freq_centers, log_dir=log_dir)
	#true_fft = apply_fft_win(im_data[:1000], 
	#		join(log_dir, 'fft_true{}_size{}'.format(freq_str, im_size)), windowing=False)
	#true_fft_hann = apply_fft_win(im_data[:1000], 
	#		join(log_dir, 'fft_true{}_size{}_hann'.format(freq_str, im_size)), windowing=True)
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
	FFT Test
	'''
	#fft_test(log_dir, sess=sess, run_seed=run_seed)

	'''
	Cos Eval
	'''
	#run_cos_eval(log_dir, sess=sess, run_seed=0)

	'''
	Layer FFT test
	'''
	#fft_layer_test(sess, log_dir, 0)

	### close session
	if sess is not None:
		sess.close()








