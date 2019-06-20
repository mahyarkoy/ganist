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
from run_ganist import read_imagenet

def read_image(im_path, im_size, sqcrop=True, bbox=None, verbose=False):
	im = Image.open(im_path)
	w, h = im.size
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
	im_re = im_re.reshape((im_size, im_size, 3))
	im.close()
	im_o = im_re / 128.0 - 1.0 
	return im_o if not verbose else (im_o, w, h)

def apply_fft(im):
	im_gray = np.mean((im + 1.0) / 2., axis=-1)
	im_t = im_gray #np.mean(im, axis=-1)
	imf = np.fft.fftn(im_t)
	imf_proc = np.abs(np.fft.fftshift(imf))**2
	return imf_proc, im_gray

def apply_fft_file(im_path, im_size=64):
	im = read_image(im_path, im_size)
	imf_proc, im_gray = apply_fft(im)
	return imf_proc, im_gray

def apply_fft_dir(im_dir, data_size=100, im_size=64):
	im_data = np.zeros((data_size, im_size, im_size))
	imf_data = np.zeros((data_size, im_size, im_size))
	cntr = 0
	for fn in glob.glob(im_dir+'/*.jpg'):
		imf_proc, im_gray = apply_fft_file(fn, im_size)
		im_data[cntr, ...] = im_gray
		imf_data[cntr, ...] = imf_proc
		cntr += 1
		if cntr == data_size:
			break
	return imf_data, im_data

def apply_fft_images(ims):
	data_size, h, w, c = ims.shape
	im_data = np.zeros((data_size, h, w))
	imf_data = np.zeros((data_size, h, w))
	for cntr, im in enumerate(ims):
		imf_proc, im_gray = apply_fft(im)
		im_data[cntr, ...] = im_gray
		imf_data[cntr, ...] = imf_proc
	return imf_data, im_data

#im_path = '/home/mahyar/celeba_1.jpg'
#im_size = 64
#imf_proc, im_gray = apply_fft(im_path, im_size)
#plt.imshow(im_gray, cmap='gray', vmin=0., vmax=1.)
#plt.show()
#plt.imshow(np.log(imf_proc))
#plt.show()

### read multiple images
im_dir = '/media/evl/Public/Mahyar/Data/image_net/train_128'
g_dir = '/media/evl/Public/Mahyar/Data/image_net/biggan/per_class_samples'
save_path = '/home/mahyar'
im_size = 128
#imf, img = apply_fft_dir(im_dir, data_size=64, im_size=im_size)
#g_imf, g_img = apply_fft_dir(g_dir, data_size=64, im_size=im_size)
imf, img = apply_fft_images(
	read_imagenet(im_dir, 10, im_size=im_size).reshape((-1, im_size, im_size, 3)))
g_imf, g_img = apply_fft_images(
	read_imagenet(g_dir, 10, im_size=im_size).reshape((-1, im_size, im_size, 3)))

imf_agg = np.mean(imf, axis=0)
g_imf_agg = np.mean(g_imf, axis=0)

### draw and save
fig = plt.figure(0, figsize=(8,6))
ax = fig.add_subplot(1,1,1)
pa = ax.imshow(np.log(imf_agg) - np.log(g_imf_agg), vmin=-5, vmax=5)
fig.colorbar(pa)
fig.savefig(save_path+'/fft_diff_imagenet_128_10perclass.jpg', dpi=300)

fig.clf()
ax = fig.add_subplot(1,1,1)
pa = ax.imshow(np.log(imf_agg), cmap=plt.get_cmap('hot'), vmin=0, vmax=20)
fig.colorbar(pa)
fig.savefig(save_path+'/fft_imagenet_128_10perclass.jpg', dpi=300)

fig.clf()
ax = fig.add_subplot(1,1,1)
pa = ax.imshow(np.log(g_imf_agg), cmap=plt.get_cmap('hot'), vmin=0, vmax=20)
fig.colorbar(pa)
fig.savefig(save_path+'/fft_gan_imagenet_128_10perclass.jpg', dpi=300)

#plt.imshow(np.log(imf_agg) - np.log(g_imf_agg))
#plt.xlim(44, 84)
#plt.ylim(44, 84)
#plt.colorbar()
#plt.show()
