#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 20:34:55 2019

@author: mahyar
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import sys

def read_image(im_path, im_size, sqcrop=True, bbox=None, verbose=False, center_crop=None):
	im = Image.open(im_path)
	w, h = im.size
	### celebA specific center crop
	if center_crop is not None:
		cy, cx = center_crop
		im_array = np.asarray(im)
		im_crop = im_array[cy-im_size//2:cy+im_size//2, cx-im_size//2:cx+im_size//2]
		im.close()
		im_o = (im_crop / 255.0) * 2.0 - 1.0
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
	### next line is because pil removes the channels for black and white images!!!
	im_re = im_re if len(im_re.shape) > 1 else np.repeat(im_re[..., np.newaxis], 3, axis=1)
	im_re = im_re.reshape((im_size, im_size, -1))
	im.close()
	im_o = (im_re / 255.0) * 2.0 - 1.0 
	im_o = im_o[:, :, :3]
	return im_o if not verbose else (im_o, w, h)

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
	
def apply_fft(im, freqs=None):
	im_gray = np.mean(im, axis=-1) #np.mean((im + 1.0) / 2., axis=-1)
	im_t = im_gray #np.mean(im, axis=-1)
	if freqs is None:
		imf = np.fft.fftn(im_t)
		imf_s = np.flip(np.fft.fftshift(imf), 0)
	else:
		imf_s, imf = compute_dft(im_t, freqs)
	imf_proc = np.abs(imf_s)**2
	return imf_proc, im_gray

def apply_fft_file(im_path, im_size=64, center_crop=None):
	im = read_image(im_path, im_size, center_crop)
	imf_proc, im_gray = apply_fft(im)
	return imf_proc, im_gray

def apply_fft_dir(im_dir, data_size=100, im_size=64, im_type='/*.jpg', center_crop=None):
	im_data = np.zeros((data_size, im_size, im_size))
	imf_data = np.zeros((data_size, im_size, im_size))
	cntr = 0
	for fn in glob.glob(im_dir+im_type):
		imf_proc, im_gray = apply_fft_file(fn, im_size, center_crop)
		im_data[cntr, ...] = im_gray
		imf_data[cntr, ...] = imf_proc
		cntr += 1
		if cntr == data_size:
			break
	return imf_data, im_data

def apply_fft_images(ims, reshape=False):
	data_size, h, w, c = ims.shape
	im_data = np.zeros((data_size, h, w))
	imf_data = np.zeros((data_size, h, w))
	for cntr, im in enumerate(ims):
		imf_proc, im_gray = apply_fft(im)
		im_data[cntr, ...] = im_gray
		imf_data[cntr, ...] = imf_proc

	if reshape:
		im_data = im_data.reshape((data_size, h, w, 1))
		imf_data = imf_data.reshape((data_size, h, w, 1))
	return imf_data, im_data

if __name__ == '__main__':
	### single image test fft vs dft
	#im_path = '/home/mahyar/celeba_3.jpg'
	#im_size = 64
	#im = read_image(im_path, 64)
	#imf_proc, im_gray, = apply_fft(im)
	#imf_dft, im_gray = apply_fft(im, freqs=1. * np.arange(im_size) / im_size)
	#plt.imshow(im_gray, cmap='gray', vmin=0., vmax=1.)
	#plt.show()
	#plt.imshow(np.log(np.flip(imf_proc, 0)))
	##xlabs = list(map('{:+.2f}'.format, 1. * np.arange(-im_size//2, im_size//2+1, im_size//8) / im_size))
	##plt.xticks(np.arange(0, im_size+1, im_size//8), xlabs)
	#plt.show()
	#plt.imshow(np.log(imf_dft))
	#ticks = [-(im_size//2), 0, im_size//2]
	#ticks_loc = [0, im_size//2, im_size+im_size%2]
	#plt.xticks(ticks_loc, ticks)
	#plt.yticks(ticks_loc, ticks[::-1])
	#plt.show()
	
	### read multiple images
	#im_dir = '/media/evl/Public/Mahyar/Data/image_net/train_128'
	#g_dir = '/media/evl/Public/Mahyar/Data/image_net/biggan/per_class_samples'
	#im_dir = '/media/evl/Public/Mahyar/Data/lsun/cat/'
	#g_dir = '/media/evl/Public/Mahyar/Data/stylegan/cat/'
	im_dir = '/media/evl/Public/Mahyar/Data/celeba/img_align_celeba/'
	#g_dir = '/media/evl/Public/Mahyar/Data/prog_gan/celeba_128/'
	g_dir = '/media/evl/Public/Mahyar/ganist_lap_logs/temp/logs_wganbn_lap3_celeba128cc/run_0/samples/'
	save_path = '/home/mahyar/'
	data_size = 1000
	im_size = 128
	imf, img = apply_fft_dir(im_dir, data_size=data_size, im_size=im_size, center_crop=(121, 89))
	g_imf, g_img = apply_fft_dir(g_dir, data_size=data_size, im_size=im_size, center_crop=(64, 64))
	#imf, img = apply_fft_images(
	#	read_imagenet(im_dir, 10, im_size=im_size).reshape((-1, im_size, im_size, 3)))
	#g_imf, g_img = apply_fft_images(
	#	read_imagenet(g_dir, 10, im_size=im_size).reshape((-1, im_size, im_size, 3)))
	
	imf_agg = np.mean(imf, axis=0)
	g_imf_agg = np.mean(g_imf, axis=0)
	
	### draw and save
	fig = plt.figure(0, figsize=(8,6))
	ax = fig.add_subplot(1,1,1)
	pa = ax.imshow(np.log(imf_agg) - np.log(g_imf_agg), cmap=plt.get_cmap('bwr'), vmin=-5, vmax=5)
	fig.colorbar(pa)
	fig.savefig(save_path+'/fft_diff_lap3_wganbn_celeba_128.jpg', dpi=300)
	
	fig.clf()
	ax = fig.add_subplot(1,1,1)
	pa = ax.imshow(np.log(imf_agg), cmap=plt.get_cmap('hot'), vmin=0, vmax=20)
	fig.colorbar(pa)
	fig.savefig(save_path+'/fft_celeba_128.jpg', dpi=300)
	
	fig.clf()
	ax = fig.add_subplot(1,1,1)
	pa = ax.imshow(np.log(g_imf_agg), cmap=plt.get_cmap('hot'), vmin=0, vmax=20)
	fig.colorbar(pa)
	fig.savefig(save_path+'/fft_lap3_wganbn_celeba_128.jpg', dpi=300)
	
	#plt.imshow(np.log(imf_agg) - np.log(g_imf_agg))
	#plt.xlim(44, 84)
	#plt.ylim(44, 84)
	#plt.colorbar()
	#plt.show()
