"""
Created on Fri Jan 24 20:34:55 2020

@author: mahyar koy
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import sys
import pickle as pk

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
Apply FFT to greyscaled images, then average power, normalize and plot.
im_data shape: (b, h, w, c)
return: none
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
	
	### compute power and normalize fft
	#fft_max_power = np.amax(im_fft, axis=(1, 2), keepdims=True)
	#im_fft_norm = im_fft / fft_max_power
	#im_fft_mean = np.mean(im_fft_norm, axis=0)
	im_fft_power = np.abs(im_fft)**2
	im_fft_mean = np.mean(im_fft_power, axis=0)
	fft_max_power = np.amax(im_fft_mean)
	im_fft_mean /= fft_max_power  
	
	### plot mean fft
	fig = plt.figure(0, figsize=(8,6))
	fig.clf()
	ax = fig.add_subplot(1,1,1)
	np.clip(im_fft_mean, 1e-20, None, out=im_fft_mean)
	pa = ax.imshow(np.log(im_fft_mean), cmap=plt.get_cmap('inferno'), vmin=-13)
	ax.set_title('Log Average Frequency Spectrum')
	dft_size = im_data.shape[1]
	#print('dft_size: {}'.format(dft_size))
	#print('fft_shape: {}'.format(im_fft_mean.shape))
	#dft_size = None
	if dft_size is not None:
		ticks_loc_x = [0, dft_size//2]
		ticks_loc_y = [0, dft_size//2-1, dft_size-dft_size%2-1]
		ax.set_xticks(ticks_loc_x)
		ax.set_xticklabels([-0.5, 0])
		ax.set_yticks(ticks_loc_y)
		ax.set_yticklabels(['', 0, -0.5])
	fig.colorbar(pa)

	### save (if image prefix is not provided, save both image and pickle data)
	if path[-4:] != '.png':
		with open(path+'.pk', 'wb+') as fs:
			pk.dump(im_fft, fs)
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
def freq_density(fft, freqs, im_size, path, mag_density=None, phase_density=None):
	mag_density = lambda x, mu=0.5, sigma=0.1: 1./(sigma*np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2/(2.*sigma**2))
	phase_density = lambda x, mu=0., sigma=0.2*np.pi: 1./(sigma*np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2/(2.*sigma**2))
	freqs = np.rint(np.array(freqs)*im_size).astype(int)
	fft = np.flip(fft, 1)
	fig = plt.figure(0, figsize=(8,6))
	mag_samples = list()
	phase_samples = list()
	for fx, fy in freqs:
		data = fft[:, im_size//2+fx, im_size//2+fy]
		mag = np.abs(data) / im_size**2 * freqs.shape[0]
		mag *= 1. if fx == 0 and fy == 0 else 2.
		phase = -np.angle(data)
		mag_samples.append(mag)
		phase_samples.append(phase)
		fig.clf()
		ax_mag = fig.add_subplot(2,1,1)
		ax_phase = fig.add_subplot(2,1,2)
		mag_count, mag_bins, _ = ax_mag.hist(mag, 100, range=(0., 1.), density=True)
		phase_count, phase_bins, _ = ax_phase.hist(phase, 100, range=(-np.pi, np.pi), density=True)
		if mag_density is not None:
			ax_mag.plot(mag_bins, mag_density(mag_bins), linewidth=2, color='r')
		if phase_density is not None:
			ax_phase.plot(phase_bins, phase_density(phase_bins), linewidth=2, color='r')
		ax_mag.set_xlabel('magnitude')
		ax_phase.set_xlabel('phase')
		fig.savefig(path+'_fx{}_fy{}'.format(fx, fy)+'.png', dpi=300)
	return mag_samples, phase_samples

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
		return mag * (np.cos(phase)*np.cos(self.kernel_loc) + np.sin(phase)*np.sin(self.kernel_loc))


