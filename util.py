"""
Created on Fri Jan 24 20:34:55 2020

@author: mahyar koy
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import glob
import sys
import pickle as pk
from os.path import join
from progressbar import ETA, Bar, Percentage, ProgressBar
import matplotlib.cm as mat_cm
import re
import scipy
import logging
import os

'''
Logger
'''
class Logger(object):
	def __init__(self, log_dir):
		self.terminal = sys.stdout
		self.log = open(os.path.join(log_dir, 'terminal.log'), 'a+')

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)  

	def flush(self):
		#this flush method is needed for python 3 compatibility.
		#this handles the flush command by doing nothing.
		#you might want to specify some extra behavior here.
		pass

class SingleLogger:
	__instance = None
	@staticmethod
	def print(msg):
		if Logger.__instance == None:
			raise Exception('Logger class is not initialized!')
		else:
			Logger.__instance.logger.debug(msg)

	@staticmethod
	def add_file_handler(fname):
		if Logger.__instance == None:
			raise Exception('Logger class is not initialized!')
		else:
			inst = Logger.__instance
			log_path = inst.get_unique_path(join(inst.log_dir, fname))
			output_file_handler = logging.FileHandler(log_path)
			inst.logger.addHandler(output_file_handler)

	def __init__(self, log_dir, fname='log'):
		if Logger.__instance != None:
			raise Exception('Logger is a singleton class and is already initialized!')
		else:
			Logger.__instance = self
			self.logger = logging.getLogger(__name__)
			self.logger.setLevel(logging.DEBUG)
			self.log_dir = log_dir
			stdout_handler = logging.StreamHandler(sys.stdout)
			self.logger.addHandler(stdout_handler)
			self.add_file_handler(fname)

	def get_unique_path(self, path, default_ext='txt'):
		ext = path.strip().split('.')[-1] if '.' in path else default_ext
		uni_path = path[:-len(ext)] if '.' in path else path
		for i in range(100):
			if not os.path.exists(f'{uni_path}.{ext}'):
				break
			uni_path = uni_path[:-3] + f'_{i:02}' if i > 0 else uni_path + f'_{i:02}'
		return f'{uni_path}.{ext}'

'''
Reads CelebA Data.
'''
def read_celeba(im_size, data_size=1000):
	### read celeba 128
	celeba_dir = '/dresden/users/mk1391/evl/Data/celeba/img_align_celeba/'
	celeba_paths = readim_path_from_dir(celeba_dir, im_type='*.jpg')
	### prepare train images and features
	celeba_data = readim_from_path(celeba_paths[:data_size], 
			im_size, center_crop=(121, 89), verbose=True)
	return celeba_data

'''
Reads path names given the directory and type extension.
'''
def readim_path_from_dir(im_dir, im_type='*.jpg'):
	im_path = join(im_dir, im_type)
	im_paths = [fn for fn in glob.glob(im_path)]
	if len(im_paths) == 0:
		raise NameError('{} does not exists!'.format(im_path))
	return im_paths

'''
Reads 3-channel images from the given paths.
returns: image array with shape (len(im_paths), im_size, im_size, 3)
'''
def readim_from_path(im_paths, im_size=64, verbose=False, **kwargs):
	data_size = len(im_paths)
	im_data = np.zeros((data_size, im_size, im_size, 3))
	if verbose:
		widgets = ["Reading Images", Percentage(), Bar(), ETA()]
		pbar = ProgressBar(maxval=data_size, widgets=widgets)
		pbar.start()
	for i, fn in enumerate(im_paths):
		if verbose:
			pbar.update(i)
		im_data[i, ...] = read_image(fn, im_size, **kwargs)
	return im_data

'''
Reads single image from path.
Return image [h, w, 3] and in [-1, 1] float format.
'''
def read_image(im_path, im_size, sqcrop=True, bbox=None, verbose=False, center_crop=None):
	im = Image.open(im_path)
	w, h = im.size
	crop_size = 128
	### celebA specific center crop: im_size cut around center
	if center_crop is not None:
		cy, cx = center_crop
		im_array = np.asarray(im)
		im_crop = im_array[cy-crop_size//2:cy+crop_size//2, cx-crop_size//2:cx+crop_size//2]
		im.close()
		im_re = im_crop
		if im_size != crop_size:
			im_pil = Image.fromarray(im_crop)
			im_re_pil = im_pil.resize((im_size, im_size), Image.BILINEAR)
			im_re = np.asarray(im_re_pil)
		im_o = (im_re / 255.0) * 2.0 - 1.0
		im_o = im_o[:, :, :3]
		return im_o if not verbose else (im_o, w, h)
	### crop and resize for all other datasets
	if sqcrop:
		im_cut = min(w, h)
		left = (w - im_cut) //2
		top = (h - im_cut) //2
		right = (w + im_cut) //2
		bottom = (h + im_cut) //2
		im_sq = im.crop((left, top, right, bottom))
	elif bbox is not None:
		left = bbox[0]
		top = bbox[1]
		right = bbox[2]
		bottom = bbox[3]
		im_sq = im.crop((left, top, right, bottom))
	else:
		im_sq = im
	im_re_pil = im_sq.resize((im_size, im_size), Image.BILINEAR)
	im_re = np.array(im_re_pil.getdata())
	## next line is because pil removes the channels for black and white images!!!
	im_re = im_re if len(im_re.shape) > 1 else np.repeat(im_re[..., np.newaxis], 3, axis=1)
	im_re = im_re.reshape((im_size, im_size, -1))
	im.close()
	im_o = (im_re / 255.0) * 2.0 - 1.0 
	im_o = im_o[:, :, :3]
	return im_o if not verbose else (im_o, w, h)

'''
Draws the given layers of the pyramid.
if im_shape is provided, it assumes that as the shape of final output, otherwise uses reconst.
pyramid: [l0, l1, ..., reconst] each shape [b, h, w, c] with values (-1,1)
'''
def pyramid_draw(pyramid, path, im_shape=None):
	n, h, w, c = pyramid[-1].shape if im_shape is None else im_shape
	im_comb = np.zeros((n, len(pyramid), h, w, c))
	for i, pi in enumerate(pyramid):
		im_comb[:, i, ...] = im_resize(pi, h, w)
	block_draw(im_comb, path, border=True)

'''
Resizes image using PIL.
ims: array with shape (b, h, w, 3)
'''
def im_resize(ims, hsize, wsize):
	b, h, w, c = ims.shape
	if hsize == h and wsize == w:
		return ims
	im_re = np.zeros((b, hsize, wsize, c))
	for i, im in enumerate(ims):
		im_pil = Image.fromarray(im)
		im_re_pil = im_pil.resize((hsize, wsize), Image.BILINEAR)
		im_re[i, ...] = np.asarray(im_re_pil)
	return im_re

'''
im_data should be a (columns, rows, imh, imw, imc).
im_data values should be in [-1, 1].
If c is not 3 then draws first channel only.
'''
def block_draw(im_data, path, separate_channels=False, border=False):
	cols, rows, imh, imw, imc = im_data.shape
	im_data = im_data if imc == 3 else np.repeat(im_data[:,:,:,:,:1], 3, axis=4)
	imc = 3
	### border
	if border:
		im_draw = im_color_borders(im_data.reshape((-1, imh, imw, imc)), 
			1.*np.ones(cols*rows, dtype=np.int32), max_label=0., color_map='RdBu')
		imb, imh, imw, imc = im_draw.shape
		im_draw = im_draw.reshape((cols, rows, imh, imw, imc))
	else:
		im_draw = im_data
	### block shape
	im_draw = im_draw.reshape([cols, imh*rows, imw, imc])
	im_draw = np.concatenate([im_draw[i, ...] for i in range(im_draw.shape[0])], axis=1)
	im_draw = (im_draw + 1.0) / 2.0
	
	### new draw without matplotlib
	images = np.clip(np.rint(im_draw * 255.0), 0.0, 255.0).astype(np.uint8)
	Image.fromarray(images, 'RGB').save(path)
	return


'''
Adds a color border to im_data corresponding to its im_label.
im_data must have shape (imb, imh, imw, imc) with values in [-1,1].
'''
def im_color_borders(im_data, im_labels, max_label=None, color_map=None, color_set=None):
	imb, imh, imw, imc = im_data.shape
	fh = imh // 32 + 1
	fw = imw // 32 + 1
	max_label = im_labels.max() if max_label is None else max_label
	if imc == 1:
		im_data_t = np.tile(im_data, (1, 1, 1, 3))
	else:
		im_data_t = np.array(im_data)
	im_labels_norm = 1. * im_labels.reshape([-1]) / (max_label + 1)
	### pick rgb color for each label: (imb, 3) in [-1,1]
	if color_map is None:
		assert color_set is not None
		rgb_colors = color_set[im_labels, ...][:, :3] * 2. - 1.
	else:
		cmap = mat_cm.get_cmap(color_map)
		rgb_colors = cmap(im_labels_norm)[:, :3] * 2. - 1.
	rgb_colors_t = np.tile(rgb_colors.reshape((imb, 1, 1, 3)), (1, imh+2*fh, imw+2*fw, 1))

	### put im into rgb_colors_t
	for b in range(imb):
		rgb_colors_t[b, fh:imh+fh, fw:imw+fw, :] = im_data[b, ...]

	return rgb_colors_t

'''
Shifts im frequencies with fc_x, fc_y and returns cos and sin components.
im: shape [b, h, w, c]
fc: f_center/f_sample which must be in [0, 0.5]
'''
def np_freq_shift(im, fc_x, fc_y):
	b, h, w, c = im.shape
	kernel_loc = 2.*np.pi*fc_x * np.arange(w).reshape((1, 1, w, 1)) + \
		2.*np.pi*fc_y * np.arange(h).reshape((1, h, 1, 1))
	kernel_cos = np.cos(kernel_loc)
	kernel_sin = np.sin(kernel_loc)
	return im * kernel_cos, im * kernel_sin

'''
DFT of the 2d discrete non-periodic input image.
im: must be 2d array
freqs: a 1d array of freqs to evaluate the dft at (max 0.5)
'''
def compute_dft(im, freqs=None):
	im_size = im.shape[0]
	freqs = 1. * np.arange(im_size) / im_size if freqs is None else freqs
	freq_size = freqs.shape[0]
	dft = 0j * np.zeros((freq_size, freq_size))
	x_arr = np.arange(im_size)
	y_arr = np.arange(im_size)
	for ui, u in enumerate(freqs):
		u_kernel = np.exp(-2j*np.pi*x_arr*u)
		for vi, v in enumerate(freqs):
			v_kernel = np.exp(-2j*np.pi*y_arr*v)
			### because u correspond to x axis
			dft[vi,ui] = im.dot(u_kernel).dot(v_kernel)
	### reshape
	even = (freq_size+1) % 2
	odd = (freq_size) % 2
	center = freq_size // 2
	dft_full = 0j * np.zeros((freq_size+even, freq_size+even))
	dft_full[:center+1, center:] = np.flip(dft[:center+1, :center+1], 0)
	dft_full[:center+1, :center] = np.flip(dft[:center+1, center+odd:], 0)
	dft_full[center+1:, center:] = np.flip(dft[center+odd:, :center+1], 0)
	dft_full[center+1:, :center] = np.flip(dft[center+odd:, center+odd:], 0)
	return dft_full, dft

'''
Apply FFT to a single greyscaled image.
im: (h, w, c)
return: fft image, greyscaled image
'''
def apply_fft(im, freqs=None):
	im_gray = np.mean(im, axis=-1)
	im_t = im_gray
	if freqs is None:
		imf = np.fft.fftn(im_t)
		imf_s = np.flip(np.fft.fftshift(imf), 0)
	else:
		imf_s, imf = compute_dft(im_t, freqs)
	return imf_s, im_gray

'''
Reverse apply_fft function.
'''
def apply_ifft(fft):
	im = np.fft.ifftn(np.fft.ifftshift(np.flip(fft, 0)))
	return im

'''
Apply FFT to greyscaled images (intensity only).
ims shape: (b, h, w, c)
return: fft images, greyscale images
'''	
def apply_fft_images(ims, reshape=False):
	data_size, h, w, c = ims.shape
	im_data = np.zeros((data_size, h, w))
	imf_data = np.zeros((data_size, h, w), dtype=complex)
	for cntr, im in enumerate(ims):
		imf_proc, im_gray = apply_fft(im)
		im_data[cntr, ...] = im_gray
		imf_data[cntr, ...] = imf_proc

	if reshape:
		im_data = im_data.reshape((data_size, h, w, 1))
		imf_data = imf_data.reshape((data_size, h, w, 1))
	return imf_data, im_data

'''
Reverse apply_fft_images.
'''
def apply_ifft_images(ffts):
	data_size, h, w = ffts.shape
	im_data = np.zeros((data_size, h, w, 1))
	for i, fft in enumerate(ffts):
		im = apply_ifft(fft)
		im_data[i, :, :, 0] = im
	return im_data

'''
Apply FFT to greyscaled images, then average power, normalize and plot.
im_data shape: (b, h, w, c)
return: shifted and flipped axis 1 fft, shape (b, h, w)
'''
def apply_fft_win(im_data, path, windowing=True, plot_ax=None):
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
	
	### compute power and normalize fft
	#fft_max_power = np.amax(im_fft, axis=(1, 2), keepdims=True)
	#im_fft_norm = im_fft / fft_max_power
	#im_fft_mean = np.mean(im_fft_norm, axis=0)
	im_fft_power = np.abs(im_fft)**2
	im_fft_mean = np.mean(im_fft_power, axis=0)
	print(f'fft_max_power: {np.amax(im_fft_power)}')
	fft_max_power = np.amax(im_fft_mean)
	im_fft_norm = im_fft_mean / fft_max_power  
	
	### plot mean fft
	if plot_ax is None:
		fig = plt.figure(0, figsize=(8,6))
		fig.clf()
		ax = fig.add_subplot(1,1,1)
	else:
		ax = plot_ax

	np.clip(im_fft_norm, 1e-20, None, out=im_fft_norm)
	pa = ax.imshow(np.log(im_fft_norm), cmap=plt.get_cmap('inferno'), vmin=-13)
	ax.set_title('Log Average Frequency Spectrum')
	dft_size = im_data.shape[1]
	#print('dft_size: {}'.format(dft_size))
	#print('fft_shape: {}'.format(im_fft_norm.shape))
	#dft_size = None
	if dft_size is not None:
		ticks_loc_x = [0, dft_size//2]
		ticks_loc_y = [0, dft_size//2-1, dft_size-dft_size%2-1]
		ax.set_xticks(ticks_loc_x)
		ax.set_xticklabels([-0.5, 0])
		ax.set_yticks(ticks_loc_y)
		ax.set_yticklabels(['', 0, -0.5])
	
	if plot_ax is None:
		fig.colorbar(pa)

		### save (if image prefix is not provided, save both image and pickle data)
		if path[-4:] != '.png':
			with open(path+'.pk', 'wb+') as fs:
				pk.dump(im_fft_mean, fs)
			path += '.png'
		fig.savefig(path, dpi=300)

	return im_fft

'''
Computes the ratio of power that has leaked.
true_power, gen_power: (h, w)
path: saves result as text file if given.
'''
def freq_leakage(true_fft, gen_fft, path=None):
	true_power = np.mean(np.abs(true_fft)**2, axis=0)
	gen_power = np.mean(np.abs(gen_fft)**2, axis=0)
	true_power_dist = 1. * true_power / np.sum(true_power)
	gen_power_dist = 1. * gen_power / np.sum(gen_power)
	leakage = np.sum(np.abs(true_power_dist - gen_power_dist)) / 2.
	if path is not None:
		with open(path+'.txt', 'w+') as fs:
			print('>>> frequency leakage: {}'.format(leakage), file=fs)
	return leakage

'''
Draws the distributions formed on each frequency location (mag and phase).
Assumes signal was real (does not work for complex signals) and in [-1,1]
fft: fast fourier transfor with shape [b, h, w], shifted and fliped on y (1 axis)
'''
def freq_density(fft, freqs, im_size, path, mag_density=None, phase_density=None, draw=True):
	mag_density = lambda x, mu=0.5, sigma=0.1: 1./(sigma*np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2/(2.*sigma**2))
	phase_density = lambda x, mu=0., sigma=0.2*np.pi: 1./(sigma*np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2/(2.*sigma**2))
	freqs = np.rint(np.array(freqs)*im_size).astype(int)
	fft = np.flip(fft, 1)
	fig = plt.figure(0, figsize=(8,6))
	mag_samples = list()
	phase_samples = list()
	freq_hist = dict()
	for fx, fy in freqs:
		assert np.abs(fx) <= im_size//2 and np.abs(fy) <= im_size//2
		data = fft[:, (im_size//2+fy) % im_size, (im_size//2+fx) % im_size]
		mag = np.abs(data) / im_size**2 * freqs.shape[0]
		mag *= 1. if (fx == 0 and fy == 0) or \
				(fx == 0 and np.abs(fy) == im_size//2) or \
				(fy == 0 and np.abs(fx) == im_size//2) or \
				(np.abs(fx) == im_size//2 and np.abs(fy) == im_size//2) else 2.
		phase = -np.angle(data)
		mag_samples.append(mag)
		phase_samples.append(phase)
		fig.clf()
		ax_mag = fig.add_subplot(2,1,1)
		ax_phase = fig.add_subplot(2,1,2)
		mag_count, mag_bins, _ = ax_mag.hist(mag, 128, range=(0., 1.), density=True)
		phase_count, phase_bins, _ = ax_phase.hist(phase, 128, range=(-np.pi, np.pi), density=True)
		freq_hist[(fx, fy)] = [mag_bins, mag_count, phase_bins, phase_count]
		with open(path+'_fx{}_fy{}'.format(fx, fy)+'.pk', 'wb+') as fs:
			pk.dump([mag_bins, mag_count, phase_bins, phase_count], fs)
		if mag_density is not None:
			ax_mag.plot(mag_bins, mag_density(mag_bins), linewidth=2, color='r')
		if phase_density is not None:
			ax_phase.plot(phase_bins, phase_density(phase_bins), linewidth=2, color='r')
		ax_mag.set_xlabel('magnitude')
		ax_phase.set_xlabel('phase')
		if draw:
			fig.savefig(path+'_fx{}_fy{}'.format(fx, fy)+'.png', dpi=300)

	return freq_hist

'''
Sampler for planar 2D cosine with uniform amplitude [-1, 1)
'''
class COS_Sampler:
	def __init__(self, im_size, fc_x, fc_y, channels=1):
		self.im_size = im_size
		self.ch = channels
		self.ksize = im_size
		self.fc_x = fc_x
		self.fc_y = fc_y
		self.kernel_loc = 2.*np.pi*fc_x * np.arange(self.ksize).reshape((1, 1, self.ksize, 1)) + \
			2.*np.pi*fc_y * np.arange(self.ksize).reshape((1, self.ksize, 1, 1))

	def sample_data(self, data_size):
		#amps = np.random.uniform(size=(data_size, 1, 1, self.ch)) * 2. - 1.
		mag = np.clip(np.random.normal(loc=0.5, scale=0.1, size=(data_size, 1, 1, self.ch)), 0., 1.)
		phase = np.clip(np.random.normal(loc=0., scale=0.2*np.pi, size=(data_size, 1, 1, self.ch)), -np.pi, np.pi)
		phase = np.random.choice([0., np.pi], size=(data_size, 1, 1, self.ch)) \
				if (self.fc_x == 0 and self.fc_y == 0) or \
				(np.abs(self.fc_x) == 0.5 and self.fc_y == 0) or \
				(self.fc_x == 0 and np.abs(self.fc_y) == 0.5) or \
				(np.abs(self.fc_x) == 0.5 and np.abs(self.fc_y) == 0.5) else phase
		return mag * np.cos(self.kernel_loc + phase)

def create_cosine(data_size, freq_centers, resolution=128, channels=1):
	im_size = resolution
	im_data = np.zeros((data_size, im_size, im_size, channels))
	freq_str = ''
	for fc in freq_centers:
		sampler = COS_Sampler(im_size=im_size, fc_x=fc[0], fc_y=fc[1], channels=channels)
		im_data += sampler.sample_data(data_size)
		freq_str += 'fx{}_fy{}_'.format(int(fc[0]*im_size), int(fc[1]*im_size))
	im_data /= len(freq_centers)
	freq_str += f'size{im_size}'
	print(f'Created Cosine: {freq_str}')
	return im_data, freq_str

'''
Evaluate the mean and average of stats in toy experiments over several runs.
log_dir: directory containing run_i directories.
'''
def eval_toy_exp(log_dir, im_size):
	run_dirs = glob.glob(join(log_dir, 'run_*/'))
	### find freqs
	freq_re = re.compile('_fx([0-9-]+)_fy([0-9-]+)')
	freqs = np.array([(int(fx), int(fy)) for fx, fy in \
			freq_re.findall(glob.glob(join(run_dirs[0], 'leakage_*_size{}.txt'.format(im_size)))[0])]).astype(int)
	mag_wd_list = list()
	phase_wd_list = list()
	mag_tv_list = list()
	phase_tv_list = list()
	leak_list = list()
	leak_hann_list = list()
	for rdir in run_dirs:
		### compute the dists
		print(rdir)
		with open(glob.glob(join(rdir, 'fft_true_*_size{}.pk'.format(im_size)))[0], 'rb') as fs:
			fft = pk.load(fs)
			true_hist = freq_density(fft, freqs/im_size, im_size, join(rdir, 'true_freq_density_size{}_data'.format(im_size)))
		with open(glob.glob(join(rdir, 'fft_gen_*_size{}.pk'.format(im_size)))[0], 'rb') as fs:
			fft = pk.load(fs)
			gen_hist = freq_density(fft, freqs/im_size, im_size, join(rdir, 'gen_freq_density_size{}_data'.format(im_size)))
		### compute the wasserstein distance between the dists for each freq
		with open(join(rdir, 'wd_mag_phase.txt'), 'w+') as fs:
			for fx, fy in freqs:
				mag_wd, phase_wd = mag_phase_wass_dist(true_hist[(fx, fy)], gen_hist[(fx, fy)])
				mag_tv, phase_tv = mag_phase_total_variation(true_hist[(fx, fy)], gen_hist[(fx, fy)])
				mag_wd_list.append(mag_wd)
				phase_wd_list.append(phase_wd)
				mag_tv_list.append(mag_tv)
				phase_tv_list.append(phase_tv)
				print('mag_wd_fx{}_fy{}: {}'.format(fx, fy, mag_wd), file=fs)
				print('phase_wd_fx{}_fy{}: {}'.format(fx, fy, phase_wd), file=fs)
				print('mag_tv_fx{}_fy{}: {}'.format(fx, fy, mag_tv), file=fs)
				print('phase_tv_fx{}_fy{}: {}'.format(fx, fy, phase_tv), file=fs)

		### read mag and phase wd
		#with open(glob.glob(join(rdir, 'wd_mag_phase.txt')), 'r') as fs:
		#	for i, l in enumerate(fs):
		#		fstr, vstr = l.strip().split()
		#		lfx, lfy = tuple(map(int, freq_re.findall(fstr)[0]))
		#		assert (lfx, lfy) == freqs[i//4]
		#		ltype = '_'.join(l.strip().split()[0].split('_')[:2])
		#		if ltype == 'mag_wd':
		#			mag_wd_list.append(float(l.strip().split()[-1]))
		#		elif ltype == 'phase_wd':
		#			phase_wd_list.append(float(l.strip().split()[-1]))
		#		elif ltype == 'mag_tv':
		#			mag_tv_list.append(float(l.strip().split()[-1]))
		#		elif ltype == 'phase_tv':
		#			phase_tv_list.append(float(l.strip().split()[-1]))

		### read the leakage
		with open(glob.glob(join(rdir, 'leakage_*_size{}.txt'.format(im_size)))[0], 'r') as fs:
			leak_list.append(float(fs.readline().strip().split()[-1]))
		with open(glob.glob(join(rdir, 'leakage_*_size{}_hann.txt'.format(im_size)))[0], 'r') as fs:
			leak_hann_list.append(float(fs.readline().strip().split()[-1]))
	
	mag_wd_mat = np.array(mag_wd_list).reshape((len(run_dirs), freqs.shape[0]))
	mag_wd_mean = np.mean(mag_wd_mat, 0)
	mag_wd_std = np.std(mag_wd_mat, 0)
	phase_wd_mat = np.array(phase_wd_list).reshape((len(run_dirs), freqs.shape[0]))
	phase_wd_mean = np.mean(phase_wd_mat, 0)
	phase_wd_std = np.std(phase_wd_mat, 0)
	mag_tv_mat = np.array(mag_tv_list).reshape((len(run_dirs), freqs.shape[0]))
	mag_tv_mean = np.mean(mag_tv_mat, 0)
	mag_tv_std = np.std(mag_tv_mat, 0)
	phase_tv_mat = np.array(phase_tv_list).reshape((len(run_dirs), freqs.shape[0]))
	phase_tv_mean = np.mean(phase_tv_mat, 0)
	phase_tv_std = np.std(phase_tv_mat, 0)
	with open(join(log_dir, 'sum_stats.txt'), 'w+') as fs:
		for i, (fx, fy) in enumerate(freqs):
			print('mag_wd_fx{}_fy{}: {} std {}'.format(fx, fy, mag_wd_mean[i], mag_wd_std[i]), file=fs)
			print('phase_wd_fx{}_fy{}: {} std {}'.format(fx, fy, phase_wd_mean[i], phase_wd_std[i]), file=fs)
			print('mag_tv_fx{}_fy{}: {} std {}'.format(fx, fy, mag_tv_mean[i], mag_tv_std[i]), file=fs)
			print('phase_tv_fx{}_fy{}: {} std {}'.format(fx, fy, phase_tv_mean[i], phase_tv_std[i]), file=fs)
		print('leak: {} std {}'.format(np.mean(leak_list), np.std(leak_list)), file=fs)
		print('leak_hann: {} std {}'.format(np.mean(leak_hann_list), np.std(leak_hann_list)), file=fs)
	return

'''
Computes wasserstein distance between true and gen distributions for mag and phase.
true_hist, gen_hist: each are a list of [mag_bins, mag_weights, phase_bins, phase_weights].
'''
def mag_phase_wass_dist(true_hist, gen_hist):
	mag_wd = scipy.stats.wasserstein_distance(
			true_hist[0][:-1], gen_hist[0][:-1], true_hist[1], gen_hist[1])
	phase_wd = scipy.stats.wasserstein_distance(
			true_hist[2][:-1], gen_hist[2][:-1], true_hist[3], gen_hist[3])
	return mag_wd, phase_wd

'''
Computes toatl variation between true and gen distributions for mag and phase.
true_hist, gen_hist: each are a list of [mag_bins, mag_weights, phase_bins, phase_weights].
'''
def mag_phase_total_variation(true_hist, gen_hist):
	mag_tv = np.sum(np.abs(
			true_hist[1]/np.sum(true_hist[1]) - gen_hist[1]/np.sum(gen_hist[1]))) / 2.
	phase_tv = np.sum(np.abs(
			true_hist[3]/np.sum(true_hist[3]) - gen_hist[3]/np.sum(gen_hist[3]))) / 2.
	return mag_tv, phase_tv

def fft_eval(im_data, dname, freq_centers, freq_str, log_dir):
	im_size = im_data.shape[2]
	im_fft = apply_fft_win(im_data, 
		os.path.join(log_dir, f'fft_{dname}{freq_str}_size{im_size}'), windowing=False)
	im_fft_hann = apply_fft_win(im_data, 
		os.path.join(log_dir, f'fft_{dname}{freq_str}_size{im_size}_hann'), windowing=True)
	im_hist = freq_density(im_fft, freq_centers, im_size, 
		os.path.join(log_dir, f'freq_density_{dname}_size{im_size}'))
	return im_fft, im_fft_hann, im_hist
	
def cosine_eval(gen_samples, dname, freq_centers, log_dir, true_fft=None, true_fft_hann=None, true_hist=None):
	im_size = gen_samples.shape[2]
	freq_str = ''.join([f'_fx{int(fx*im_size)}fy{int(fy*im_size)}' for fx, fy in freq_centers])
	gen_fft, gen_fft_hann, gen_hist = fft_eval(gen_samples, dname, freq_centers, freq_str, log_dir=log_dir)
	if true_fft is not None and true_fft_hann is not None and true_hist is not None:
		fft_leakage = freq_leakage(true_fft, gen_fft)
		fft_leakage_hann = freq_leakage(true_fft_hann, gen_fft_hann)
		freqs = np.rint(np.array(freq_centers)*im_size).astype(int)
		with open(join(log_dir, f'wd_mag_phase_{dname}{freq_str}.txt'), 'w+') as fs:
			for fx, fy in freqs:
				mag_wd, phase_wd = mag_phase_wass_dist(true_hist[(fx, fy)], gen_hist[(fx, fy)])
				mag_tv, phase_tv = mag_phase_total_variation(true_hist[(fx, fy)], gen_hist[(fx, fy)])
				print(f'mag_wd_fx{fx}_fy{fy}: {mag_wd}', file=fs)
				print(f'phase_wd_fx{fx}_fy{fy}: {phase_wd}', file=fs)
				print(f'mag_tv_fx{fx}_fy{fy}: {mag_tv}', file=fs)
				print(f'phase_tv_fx{fx}_fy{fy}: {phase_tv}', file=fs)
			print(f'>>> true_total_power: {np.sum(np.mean(np.abs(true_fft)**2, axis=0))}', file=fs)
			print(f'>>> gen_total_power: {np.sum(np.mean(np.abs(gen_fft)**2, axis=0))}', file=fs)
			print(f'>>> leakage ratio: {fft_leakage}', file=fs)
			print(f'>>> true_total_power_hann: {np.sum(np.mean(np.abs(true_fft_hann)**2, axis=0))}', file=fs)
			print(f'>>> gen_total_power_hann: {np.sum(np.mean(np.abs(gen_fft_hann)**2, axis=0))}', file=fs)
			print(f'>>> leakage ratio hann: {fft_leakage_hann}', file=fs)
	return gen_fft, gen_fft_hann, gen_hist

def rotate(vec, deg):
		deg = deg * np.pi / 180.
		return np.matmul(np.array([[np.cos(deg), -np.sin(deg)], [np.sin(deg), np.cos(deg)]]), vec)

def koch(start_point, end_point, level):
	koch_trail = list()
	if level > 0:
		vec = end_point - start_point
		### 1st segment
		mid_start = start_point + vec / 3.
		koch_trail += koch(start_point, mid_start, level-1)
		koch_trail.append(mid_start)
		### 2nd segment
		mid_tip = mid_start + rotate(vec / 3., 60.)
		koch_trail += koch(mid_start, mid_tip, level-1)
		koch_trail.append(mid_tip)
		### 3rd segment
		mid_end = start_point + 2 * vec / 3.
		koch_trail += koch(mid_tip, mid_end, level-1)
		koch_trail.append(mid_end)
		### 4th segment
		koch_trail += koch(mid_end, end_point, level-1)
	return koch_trail

def make_koch_snowflake(level, rot, res, channels, line_width=1):
	center = np.array([0., 0.])
	scale = 1.
	vec = rotate(np.array([0., 1.]) * scale, rot)
	top = center + vec
	left = center + rotate(vec, 120.)
	right = center + rotate(vec, -120.)
	koch_trail = [top]
	### make koch
	koch_trail += koch(top, right, level)
	koch_trail.append(right)
	koch_trail += koch(right, left, level)
	koch_trail.append(left)
	koch_trail += koch(left, top, level)
	koch_trail.append(top)
	### make koch image mask
	def rescale(p):
		p_re = np.array(p)
		p_re[1] *= -1
		return np.rint(res / 2. * p_re + res / 2).tolist()
	im = Image.new('RGB', (res, res))
	draw = ImageDraw.Draw(im)
	for i in range(len(koch_trail) - 1):
		draw.line(rescale(koch_trail[i])+rescale(koch_trail[i+1]), fill=(255, 255, 255), width=line_width)
	return np.asarray(im)[:, :, :channels] / 255. * 2. - 1.

def rgb2gray(rgb):
	r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
	gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
	return gray

def fractal_dimension(Z, threshold=0.9):
	# Only for 2d image
	assert(len(Z.shape) == 2)

	# From https://github.com/rougier/numpy-100 (#87)
	def boxcount(Z, k):
		S = np.add.reduceat(
			np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
							   np.arange(0, Z.shape[1], k), axis=1)

		# We count non-empty (0) and non-full boxes (k*k)
		return len(np.where((S > 0) & (S < k*k))[0])

	# Transform Z into a binary array
	Z = (Z > threshold)

	# Minimal dimension of image
	p = min(Z.shape)

	# Greatest power of 2 less than or equal to p
	n = int(np.floor(np.log(p)/np.log(2))) - 3

	# Build successive box sizes (from 2**n down to 2**1)
	sizes = 2**np.arange(n, 1, -1)

	# Actual box counting with decreasing size
	counts = []
	for size in sizes:
		counts.append(boxcount(Z, size))

	# Fit the successive log(sizes) with log (counts)
	coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
	return coeffs, sizes, counts

def fractal_eval(im_data, log_name, log_dir):
	box_dims = list()
	threshold = 0.
	for im in im_data:
		coeffs, _, _ = fractal_dimension(im[:,:,0], threshold=threshold)
		box_dims.append(-coeffs[0])
	apply_fft_win(im_data - np.mean(im_data), 
		os.path.join(log_dir, f'fft_{log_name}'), windowing=False)
	draw_size = 1 if im_data.shape[0] < 25 else 5
	block_draw((im_data[:draw_size**2] > threshold).astype(np.float).reshape((draw_size, draw_size)+im_data.shape[1:]), os.path.join(log_dir, f'fractal_eval_samples_{log_name}.png'))
	path_length = np.sum(im_data > threshold, axis=(1,2,3))
	eval_str = f'>>> fractal eval {log_name}: box_dim={np.mean(box_dims)} sd {np.std(box_dims)} and path_length={np.mean(path_length)} sd {np.std(path_length)}'
	print(eval_str)
	with open(os.path.join(log_dir, f'fractal_eval.txt'), 'a+') as fs:
		print(eval_str, file=fs)

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

def fft_test_by_samples(log_dir, r_samples, g_samples, r_name='true', g_name='gen'):
	assert r_samples.shape == g_samples.shape, f'r_samples shape of {r_samples.shape} != g_samples shape of {g_samples.shape}'
	data_size = r_samples.shape[0]
	im_size = r_samples.shape[1]

	### calculate fft mean
	r_fft_mean = fft_norm(windowing(r_samples))
	g_fft_mean = fft_norm(windowing(g_samples))

	### read fft mean data from file
	#with open(join(log_dir, f'{g_name}_{r_name}_{data_size}_fft_mean.pk'), 'rb') as fs:
	#	g_fft_mean, r_fft_mean = pk.load(fs)

	### write fft mean data to file
	with open(join(log_dir, f'{g_name}_{r_name}_{data_size}_fft_mean.pk'), 'wb+') as fs:
		pk.dump([g_fft_mean, r_fft_mean], fs)

	### compute fft metrics
	r_fft_density = r_fft_mean / np.sum(r_fft_mean)
	g_fft_density = g_fft_mean / np.sum(g_fft_mean)
	fft_diff = np.abs(np.log(r_fft_mean) - np.log(g_fft_mean))
	total_var = 100. * np.sum(np.abs(r_fft_density - g_fft_density)) / 2.
	SingleLogger.print(f'>>> fft_test_{g_name}_{r_name}: Leakage percentage (TV): {total_var}')
	SingleLogger.print(f'>>> true_total_power_hann: {np.sum(r_fft_mean)}')
	SingleLogger.print(f'>>> gen_total_power_hann: {np.sum(g_fft_mean)}')

	### compute aggregated frequency difference
	#blur_levels = [1.]
	#blur_num = 7
	#blur_delta = 1. / 8
	#blur_init = blur_levels[-1]
	#for i in range(blur_num):
	#	blur_levels.append(
	#		1. / ((1. / blur_levels[-1]) - (blur_delta / blur_init)))
	#bins_loc = [0.] + [1./(np.pi*2*s) for s in blur_levels[::-1]]
	##bins_loc = [0.] + list(np.arange(1, 50) / 100.)
	#bins = np.zeros(len(bins_loc))
	#bins_count = np.array(bins)
	#fft_h, fft_w = fft_diff.shape
	#print('>>> freq bins:')
	#for v in range(fft_h):
	#	bin_str = ''
	#	for u in range(fft_w):
	#		fft_hc = fft_h//2
	#		fft_wc = fft_w//2
	#		freq = np.sqrt(((v - fft_hc + 1 - fft_h%2)/fft_hc)**2. + ((u - fft_wc)/fft_wc)**2)
	#		for i, bin_freq in enumerate(bins_loc[::-1]):
	#			if freq >= bin_freq:
	#				bin_id = len(bins_loc) - i - 1
	#				bins[bin_id] += fft_diff[v, u]
	#				bins_count[bin_id] += 1
	#				break
	#		bin_str += f'{bin_id}'
	#	print(bin_str)
	#print(f'>>> freq bins:\t{bins_loc}')
	#print(f'>>> freq diff density:\t{bins / bins_count}')

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
			continue
			#ax.grid(True, which='both', linestyle='dotted')
			#ax.set_xlabel(r'Frequency bins')
			#ax.set_ylabel('Power diff density')
			#ax.set_title('Power diff density')
			#ax.plot(bins / bins_count)
			#ax.set_xticks(range(len(bins_loc)))
			#xticklabels = [np.format_float_positional(v, 2) for v in bins_loc]
			#ax.set_xticklabels(xticklabels)
			#fig.savefig(join(log_dir, f'fft_test_{g_name}_{r_name}_{fig_names[i]}.png'), dpi=300)
			#break

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

		fig.savefig(join(log_dir, f'fft_test_{g_name}_{r_name}_{fig_names[i]}.png'), dpi=300)



