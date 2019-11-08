import numpy as np
import tensorflow as tf
from run_ganist import readim_path_from_dir, readim_from_path, block_draw, im_block_draw
from run_ganist import TFutil, sample_ganist, create_lsun, CUB_Sampler
from fft_test import apply_fft_images
import matplotlib.pyplot as plt
from PIL import Image
import tf_ganist
import sys
					
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

'''
Read Data
'''
def read_celeba(im_size, data_size=1000):
	### read celeba 128
	celeba_dir = '/media/evl/Public/Mahyar/Data/celeba/img_align_celeba/'
	celeba_paths = readim_path_from_dir(celeba_dir, im_type='/*.jpg')
	### prepare train images and features
	celeba_data = readim_from_path(celeba_paths[:data_size], 
			im_size, center_crop=(121, 89), verbose=True)
	return celeba_data

'''
Apply FFT
'''
def apply_fft_win(im_data, path, windowing=True):
	### windowing
	win_size = im_data.shape[1]
	win = np.hanning(im_data.shape[1])
	win = np.outer(win, win).reshape((win_size, win_size, 1))
	#single_draw(win, '/home/mahyar/miss_details_images/temp/hann_win.png')
	#single_draw(win*im_data[0], '/home/mahyar/miss_details_images/temp/hann_win_im.png')
	im_data = im_data * win if windowing is True else im_data
	
	### apply fft
	print('>>> fft image shape: {}'.format(im_data.shape))
	im_fft, _ = apply_fft_images(im_data, reshape=False)
	### copy nyquist freq component to positive side of x and y axis
	#im_fft_ext = np.concatenate((im_fft_mean, im_fft_mean[:, :1]/2.), axis=1)
	#im_fft_ext = np.concatenate((im_fft_ext[-1:, :]/2., im_fft_ext), axis=0)
	
	### normalize fft
	fft_max_power = np.amax(im_fft, axis=(1, 2), keepdims=True)
	im_fft_norm = im_fft / fft_max_power
	im_fft_mean = np.mean(im_fft_norm, axis=0)
	
	### plot mean fft
	fig = plt.figure(0, figsize=(8,6))
	fig.clf()
	ax = fig.add_subplot(1,1,1)
	np.clip(im_fft_mean, 1e-20, None, out=im_fft_mean)
	pa = ax.imshow(np.log(im_fft_mean), cmap=plt.get_cmap('inferno'), vmin=-13)
	ax.set_title('Log Average Frequency Spectrum')
	dft_size = im_data.shape[1]
	print('dft_size: {}'.format(dft_size))
	print('fft_shape: {}'.format(im_fft_mean.shape))
	#dft_size = None
	if dft_size is not None:
		ticks_loc_x = [0, dft_size//2]
		ticks_loc_y = [0, dft_size//2-1, dft_size-dft_size%2-1]
		ax.set_xticks(ticks_loc_x)
		ax.set_xticklabels([-0.5, 0])
		ax.set_yticks(ticks_loc_y)
		ax.set_yticklabels(['', 0, -0.5])
	fig.colorbar(pa)
	fig.savefig(path, dpi=300)
	return

def leakage_test(log_dir, im_size=128, ksize=16, fc_x=1./4, fc_y=0.):
	kernel_loc = 2.*np.pi*fc_x * np.arange(ksize).reshape((1, 1, ksize, 1)) + \
		2.*np.pi*fc_y * np.arange(ksize).reshape((1, ksize, 1, 1))
	kernel_cos = np.cos(kernel_loc)
	im_data = np.zeros((1, im_size, im_size, 1))
	im_data[0, :ksize, :ksize, :1] = kernel_cos
	apply_fft_win(im_data, log_dir+'/fft_im{}_cos{}.png'.format(im_size, ksize), windowing=False)	

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
	log_dir = '/home/mahyar/miss_details_images/temp/'
	
	'''
	Leakage test
	'''
	#leakage_test(log_dir)
	
	'''
	CUB FFT test
	'''
	cub_dir = '/media/evl/Public/Mahyar/Data/cub/CUB_200_2011/'
	im_size = 128
	data_size = 1000
	test_size = 1000
	sampler = CUB_Sampler(cub_dir, im_size=im_size, order='test', test_size=test_size)
	cub_data = sampler.sample_data(data_size)
	print('>>> CUB shape: {}'.format(cub_data.shape))
	print('>>> CUB size: {}'.format(sampler.total_count))
	print('>>> CUB number of classes: {}'.format(1 + np.amax(sampler.cls)))
	print('>>> CUB average bbox (h, w): ({}, {})'.format(np.mean(sampler.bbox[:, 2]), np.mean(sampler.bbox[:, 3])))
	im_block_draw(cub_data, 5, log_dir+'/cub{}bb_samples.png'.format(im_size), border=True)
	apply_fft_win(cub_data, log_dir+'/fft_cub{}bb_hann.png'.format(im_size))
	sys.exit(0)

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
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
	config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	### init TFutil
	tfutil = TFutil(sess)

	data_size = 1000
	### create a ganist instance
	ganist = tf_ganist.Ganist(sess, log_dir)
	### init variables
	sess.run(tf.global_variables_initializer())
	### load ganist
	#load_path = log_dir_snap+'/model_best.h5'
	#ganist.load(load_path.format(run_seed))
	### sample
	g_samples = sample_ganist(ganist, data_size, output_type='rec')[0]
	apply_fft_win(g_samples, log_dir+'fft_wganbn_sh16_comb9_hann.png')
	
	### close session
	sess.close()