#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 11:10:34 2017

@author: mahyar
"""
### To convert to mp4 in command line
# ffmpeg -framerate 25 -i fields/field_%d.png -c:v libx264 -pix_fmt yuv420p baby_log_15.mp4
### To speed up mp4
# ffmpeg -i baby_log_57.mp4 -r 100 -filter:v "setpts=0.1*PTS" baby_log_57_100.mp4
# for i in {0..7}; do mv baby_log_a"$((i))" baby_log_"$((i+74))"; done

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as mat_cm
import os
from progressbar import ETA, Bar, Percentage, ProgressBar
import argparse
print matplotlib.get_backend()
import cPickle as pk
import gzip
from skimage.transform import resize
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.utils.graph import graph_shortest_path
import sys
import scipy
import skimage.io as skio
import glob
from PIL import Image
from scipy.stats import beta as beta_dist
from scipy import signal

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0, 1" for multiple

'''
Reads mnist data from file and return (data, labels) for train, val, test respctively.
'''
def read_mnist(mnist_path):
	### read mnist data
	f = gzip.open(mnist_path, 'rb')
	train_set, val_set, test_set = pk.load(f)
	f.close()
	return train_set, val_set, test_set

'''
Resizes images to im_size and scale to (-1,1)
'''
def im_process(im_data, im_size=28):
	im_data = im_data.reshape((im_data.shape[0], 28, 28, 1))
	### resize
	#im_data_re = np.zeros((im_data.shape[0], im_size, im_size, 1))
	#for i in range(im_data.shape[0]):
	#	im_data_re[i, ...] = resize(im_data[i, ...], (im_size, im_size), preserve_range=True)
	im_data_re = np.array(im_data)

	### rescale
	im_data_re = im_data_re * 2.0 - 1.0
	return im_data_re

def read_image(im_path, im_size, sqcrop=True, bbox=None, verbose=False, center_crop=None):
	im = Image.open(im_path)
	w, h = im.size
	### celebA specific center crop: im_size cut around center
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
	## next line is because pil removes the channels for black and white images!!!
	im_re = im_re if len(im_re.shape) > 1 else np.repeat(im_re[..., np.newaxis], 3, axis=1)
	im_re = im_re.reshape((im_size, im_size, -1))
	im.close()
	im_o = (im_re / 255.0) * 2.0 - 1.0 
	im_o = im_o[:, :, :3]
	return im_o if not verbose else (im_o, w, h)

def read_imagenet(im_dir, data_size, im_size=64):
	im_data = np.zeros((1000, data_size, im_size, im_size, 3))
	print '>>> Reading ImageNet from: '+ im_dir
	widgets = ["ImageNet", Percentage(), Bar(), ETA()]
	pbar = ProgressBar(maxval=1000, widgets=widgets)
	pbar.start()
	for c in range(1000):
		pbar.update(c)
		i = 0
		for fn in glob.glob(im_dir+'/{}/*.jpg'.format(c)):
			im_data[c, i, ...] = read_image(fn, im_size)
			i += 1
			if i == data_size:
				break
	return im_data

def readim_from_path(im_paths, im_size=64, center_crop=None):
	data_size = len(im_paths)
	im_data = np.zeros((data_size, im_size, im_size, 3))
	print '>>> Reading Images'
	widgets = ["Image Read", Percentage(), Bar(), ETA()]
	pbar = ProgressBar(maxval=data_size, widgets=widgets)
	pbar.start()
	for i, fn in enumerate(im_paths):
		pbar.update(i)
		im_data[i, ...] = read_image(fn, im_size, center_crop=center_crop)
	return im_data

def readim_path_from_dir(im_dir, im_type='/*.jpg'):
	return [fn for fn in glob.glob(im_dir+im_type)]

def read_lsun(lsun_path, data_size, im_size=64):
	im_data = np.zeros((data_size, im_size, im_size, 3))
	i = 0
	print '>>> Reading LSUN from: '+lsun_path
	widgets = ["LSUN", Percentage(), Bar(), ETA()]
	pbar = ProgressBar(maxval=data_size, widgets=widgets)
	pbar.start()
	for fn in glob.glob(lsun_path+'/*.jpg'):
		pbar.update(i)
		im_data[i, ...] = read_image(fn, im_size)
		i += 1
		if i == data_size:
			break
	return im_data

def read_art(art_path, data_size=None, im_size=64):
	print '>>> Reading ART from: '+art_path
	with open(art_path+'/annotation_pruned.cpk', 'rb') as fs:
		datadict = pk.load(fs)
	fn_list = datadict['image_file_name']
	data_size = len(fn_list) if data_size is None else data_size
	im_data = np.zeros((data_size, im_size, im_size, 3))
	widgets = ["ART", Percentage(), Bar(), ETA()]
	pbar = ProgressBar(maxval=data_size, widgets=widgets)
	pbar.start()
	for i in range(data_size):
		pbar.update(i)
		fn = fn_list[i]
		im_data[i, ...] = read_image(art_path+'/art_images/'+fn, im_size)
	
	return im_data, datadict['style_idx'][:data_size]

def read_cifar(cifar_path):
	with open(cifar_path, 'rb') as fs:
		datadict = pk.load(fs)
	data = datadict['data'].reshape((-1, 3, 32, 32))
	labs = np.array(datadict['labels'])
	data_proc = data / 128.0 - 1.0
	return np.transpose(data_proc, axes=(0,2,3,1)), labs

def read_stl(stl_data_path, stl_lab_path, im_size=64):
	with open(stl_data_path, 'rb') as f:
		# read whole file in uint8 chunks
		everything = np.fromfile(f, dtype=np.uint8)
		im_data = np.reshape(everything, (-1, 3, 96, 96))
		im_data = np.transpose(im_data, (0, 3, 2, 1))

		### resize
		im_data_re = np.zeros((im_data.shape[0], im_size, im_size, 3))
		for i in range(im_data.shape[0]):
			im_data_re[i, ...] = resize(im_data[i, ...], (im_size, im_size), preserve_range=True)
		im_data_re = im_data_re / 128.0 - 1.0

	with open(stl_lab_path, 'rb') as f:
		labels = np.fromfile(f, dtype=np.uint8) - 1

	return im_data_re, labels

def shuffle_data(im_data, im_bboxes=None, label=None):
	order = np.arange(im_data.shape[0])
	np.random.shuffle(order)
	im_data_sh = im_data[order, ...]
	if im_bboxes is None and label is None:
		return im_data_sh, None, None
	elif im_bboxes is None:
		return im_data_sh, None, label[order]
	elif label is None:
		im_bboxes_sh = [im_bboxes[i] for i in order]
		return im_data_sh, im_bboxes_sh, None
	else:
		im_bboxes_sh = [im_bboxes[i] for i in order]
		return im_data_sh, im_bboxes_sh, label[order]

'''
Stacks images randomly on RGB channels, im_data shape must be (N, d, d, 1).
'''
def get_stack_mnist(im_data, labels=None, stack_size=3):
	order = np.arange(im_data.shape[0])
	
	np.random.shuffle(order)
	im_data_r = im_data[order, ...]
	labs_r = labels[order] if labels is not None else None

	if stack_size != 3:
		return im_data_r, labs_r

	np.random.shuffle(order)
	im_data_g = im_data[order, ...]
	labs_g = labels[order] if labels is not None else None

	np.random.shuffle(order)
	im_data_b = im_data[order, ...]
	labs_b = labels[order] if labels is not None else None

	### stack shuffled channels
	im_data_stacked = np.concatenate((im_data_r, im_data_g, im_data_b), axis=3)
	labs_stacked = labs_r + 10 * labs_g + 100 * labs_b if labels is not None else None
	
	return im_data_stacked, labs_stacked

def plot_time_series(name, vals, fignum, save_path, color='b', ytype='linear', itrs=None):
	fig, ax = plt.subplots(figsize=(8, 6))
	ax.clear()
	if itrs is None:
		ax.plot(vals, color=color)	
	else:
		ax.plot(itrs, vals, color=color)
	ax.grid(True, which='both', linestyle='dotted')
	ax.set_title(name)
	ax.set_xlabel('Iterations')
	ax.set_ylabel('Values')
	if ytype=='log':
		ax.set_yscale('log')
	fig.savefig(save_path, dpi=300)
	plt.close(fig)

def plot_time_mat(mat, mat_names, fignum, save_path, ytype=None, itrs=None):
	for n in range(mat.shape[1]):
		fig_name = mat_names[n]
		if not ytype:
			ytype = 'log' if 'param' in fig_name else 'linear'
		plot_time_series(fig_name, mat[:,n], fignum, save_path+'/'+fig_name+'.png', ytype=ytype, itrs=itrs)

'''
Samples sample_size images from each ganist generator, draws with color.
im_data must have shape (imb, imh, imw, imc) with values in [-1,1]
'''
def gset_block_draw(ganist, sample_size, path, en_color=True, border=False):
	im_draw = np.zeros([ganist.g_num, sample_size]+ganist.data_dim)
	z_data = np.zeros(ganist.g_num*sample_size, dtype=np.int32)
	im_size = ganist.data_dim[0]
	for g in range(ganist.g_num):
		z_data[g*sample_size:(g+1)*sample_size] = g * np.ones(sample_size, dtype=np.int32)
		im_draw[g, ...] = sample_ganist(ganist, sample_size, z_data=z_data[g*sample_size:(g+1)*sample_size])
	#im_draw = (im_draw + 1.0) / 2.0
	if border:
		im_draw = im_color_borders(im_draw.reshape([-1]+ganist.data_dim), z_data)
		im_block_draw(im_draw, sample_size, path, z_data)
	elif en_color:
		en_block_draw(ganist, im_draw, path)
	else:
		block_draw(im_draw, path)

'''
Similar to gset_block_draw, except only draw high probability generators
'''
def gset_block_draw_top(ganist, sample_size, path, pr_th=0.05, en_color=False, g_color=True):
	g_pr = np.exp(ganist.pg_temp * ganist.g_rl_pvals)
	g_pr = g_pr / np.sum(g_pr)
	top_g_count = np.sum(g_pr > pr_th)
	im_draw = np.zeros([top_g_count, sample_size]+ganist.data_dim)
	z_data = np.zeros([top_g_count, sample_size], dtype=np.int32)
	im_size = ganist.data_dim[0]
	i = 0
	for g in range(ganist.g_num):
		if g_pr[g] <= pr_th:
			continue
		z_data[i, ...] = g * np.ones(sample_size, dtype=np.int32)
		im_draw[i, ...] = sample_ganist(ganist, sample_size, z_data=z_data[i, ...])
		i += 1
	#im_draw = (im_draw + 1.0) / 2.0
	if g_color is True:
		im_draw_flat = im_draw.reshape([-1]+ganist.data_dim)
		z_data_flat = z_data.reshape([-1])
		im_draw_color = im_color_borders(im_draw_flat, z_data_flat, max_label=ganist.g_num-1)
		_, imh, imw, imc = im_draw_color.shape
		im_draw = im_draw_color.reshape([top_g_count, sample_size, imh, imw, imc])
	if en_color is True:
		en_block_draw(ganist, im_draw, path)
	else:
		block_draw(im_draw, path)
'''
Draws im_data images one by one as separate image files in the save_dir.
im_data shape: (B, H, W, C)
'''
def im_separate_draw(im_data, save_dir):
	images = np.clip(np.rint((im_data + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)
	for i, im in enumerate(images):
		fname = save_dir + '/{}.jpg'.format(i)
		Image.fromarray(im, 'RGB').save(fname, 'JPEG')

'''
Draws sample_size**2 randomly selected images from im_data.
If im_labels is provided: selects sample_size images for each im_label and puts in columns.
If ganist is provided: classifies selected images and adds color border.
im_data must have shape (imb, imh, imw, imc) with values in [-1,1].
'''
def im_block_draw(im_data, sample_size, path, im_labels=None, ganist=None, border=False):
	imb, imh, imw, imc = im_data.shape
	if im_labels is not None:
		max_label = im_labels.max()
		im_draw = np.zeros([max_label+1, sample_size, imh, imw, imc])
		### select sample_size images from each label
		for g in range(max_label+1):
			im_draw[g, ...] = im_data[im_labels == g, ...][:sample_size, ...]
	else:
		draw_ids = np.random.choice(imb, size=sample_size**2, replace=False)
		im_draw = im_data[draw_ids, ...].reshape([sample_size, sample_size, imh, imw, imc])
	
	#im_draw = (im_draw + 1.0) / 2.0
	if ganist is not None:
		en_block_draw(ganist, im_draw, path)
	elif border:
		block_draw(im_draw, path, border=True)
	else:
		block_draw(im_draw, path)

'''
Classifies im_data with ganist e_net, draws with color borders.
im_data must have shape (cols, rows, imh, imw, imc) with values in [0,1]
'''
def en_block_draw(ganist, im_data, path, max_label=None):
	cols, rows, imh, imw, imc = im_data.shape
	max_label = ganist.g_num-1 if max_label is None else max_label
	im_draw_flat = im_data.reshape([-1]+ganist.data_dim)
	en_labels = np.argmax(eval_ganist_en(ganist, im_draw_flat), axis=1)
	im_draw_color = im_color_borders(im_draw_flat, en_labels, max_label=max_label)
	_, imh, imw, imc = im_draw_color.shape
	block_draw(im_draw_color.reshape([cols, rows, imh, imw, imc]), path)

'''
Adds a color border to im_data corresponding to its im_label.
im_data must have shape (imb, imh, imw, imc) with values in [-1,1].
'''
def im_color_borders(im_data, im_labels, max_label=None, color_map=None):
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
		rgb_colors = global_color_set[im_labels, ...][:, :3] * 2. - 1.
	else:
		cmap = mat_cm.get_cmap(color_map)
		rgb_colors = cmap(im_labels_norm)[:, :3] * 2. - 1.
	rgb_colors_t = np.tile(rgb_colors.reshape((imb, 1, 1, 3)), (1, imh+2*fh, imw+2*fw, 1))

	### put im into rgb_colors_t
	for b in range(imb):
		rgb_colors_t[b, fh:imh+fh, fw:imw+fw, :] = im_data[b, ...]

	return rgb_colors_t

	### create mask
	#box_mask = np.ones((imh, imw))
	#box_mask[fh+1:imh-fh, fw+1:imw-fw] = 0.
	#box_mask_t = np.tile(box_mask.reshape((1, imh, imw, 1)), (imb, 1, 1, 3))
	#box_mask_inv = np.abs(box_mask_t - 1.)

	### apply mask
	#im_data_border = im_data_t * box_mask_inv + rgb_colors_t * box_mask_t
	#return im_data_border

'''
Draws the given layers of the pyramid.
pyramid: [l0, l1, ..., reconst] each shape [b, h, w, c] with values (-1,1)
'''
def pyramid_draw(pyramid, path):
	n, h, w, c = pyramid[-1].shape
	im_l0 = TFutil.get().upsample(pyramid[0], times=2)
	im_l1 = TFutil.get().upsample(pyramid[1])
	im_l2 = pyramid[2]
	im_rec = pyramid[-1]
	im_comb = np.zeros((n, 4, h, w, c))
	im_comb[:, 0, ...] = im_l0
	im_comb[:, 1, ...] = im_l1
	im_comb[:, 2, ...] = im_l2
	im_comb[:, 3, ...] = im_rec
	block_draw(im_comb, path, border=True)

'''
Samples from all layers of the generator pyramid and draws them if path is specified.
'''
def sample_pyramid(ganist, path=None, sample_size=10):
	zi = np.random.uniform(low=-ganist.z_range, high=ganist.z_range, 
		size=[sample_size, ganist.z_dim]).astype(tf_ganist.np_dtype)
	samples_l0 = sample_ganist(ganist, sample_size, zi_data=zi, output_type='l0')
	samples_l1 = sample_ganist(ganist, sample_size, zi_data=zi, output_type='l1')
	samples_l2 = sample_ganist(ganist, sample_size, zi_data=zi, output_type='l2')
	samples_rec = sample_ganist(ganist, sample_size, zi_data=zi, output_type='rec')
	pyramid = [samples_l0, samples_l1, samples_l2, samples_rec]
	if path is not None:
		pyramid_draw(pyramid, path)
	return pyramid

'''
im_data should be a (columns, rows, imh, imw, imc).
im_data values should be in [-1, 1].
If c is not 3 then draws first channel only.
'''
def block_draw(im_data, path, separate_channels=False, border=False):
	cols, rows, imh, imw, imc = im_data.shape
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

	### plots
	#fig = plt.figure(0)
	#fig.clf()
	#if not separate_channels or im_draw.shape[-1] != 3:
	#	ax = fig.add_subplot(1, 1, 1)
	#	if im_draw.shape[-1] == 1:
	#		ims = ax.imshow(im_draw.reshape(im_draw.shape[:-1]))
	#	else:
	#		ims = ax.imshow(im_draw)
	#	ax.set_axis_off()
	#	#fig.colorbar(ims)
	#	fig.savefig(path, dpi=300)
	#else:
	#	im_tmp = np.zeros(im_draw.shape)
	#	ax = fig.add_subplot(1, 3, 1)
	#	im_tmp[..., 0] = im_draw[..., 0]
	#	ax.set_axis_off()
	#	ax.imshow(im_tmp)
	#	
	#	ax = fig.add_subplot(1, 3, 2)
	#	im_tmp[...] = 0.0
	#	im_tmp[..., 1] = im_draw[..., 1]
	#	ax.set_axis_off()
	#	ax.imshow(im_tmp)
	#	
	#	ax = fig.add_subplot(1, 3, 3)
	#	im_tmp[...] = 0.0
	#	im_tmp[..., 2] = im_draw[..., 2]
	#	ax.set_axis_off()
	#	ax.imshow(im_tmp)
	#	
	#	fig.subplots_adjust(wspace=0, hspace=0)
	#	fig.savefig(path, dpi=300)

'''
Plot and saves layer stats.
input_list: a list of dict with flat array val and layer name key.
itrs: iteration counter when input_list is stored.
'''
def plot_layer_stats(input_list, itrs, title, log_path, beta_conf=0.95, min_val=False):
	fig, ax = plt.subplots(figsize=(8, 6))
	ax.clear()
	cmap = mat_cm.get_cmap('tab10')
	c = [a*0.1 for a in range(10)]
	layer_names = input_list[0].keys()
	assert len(c) >= len(layer_names), 'Not enough colors to plot layer_stats!'
	layer_vals = list()
	for i, n in enumerate(layer_names):
		lmat = np.array([d[n] for d in input_list])
		lmean = np.mean(lmat, axis=1)
		lstd = np.std(lmat, axis=1)
		lmin = np.min(lmat, axis=1)
		if beta_conf > 0.:
			m = (lmean + 1.) / 2.
			s = lstd / 2.
			a = m**2.*((1.-m)/s**2. - 1./m)
			b = a*(1/m - 1.)
			confs = [beta_dist.interval(beta_conf, ai, bi) for ai, bi in zip(a, b)]
			err_up = np.array([ci[1]*2.-1. for ci in confs])
			err_low = np.array([ci[0]*2.-1. for ci in confs])
		else:
			err_up = lmean+lstd
			err_low = lmean-lstd

		if min_val:
			ax.plot(itrs, lmin, label=n, color=cmap(c[i]))
		else:
			ax.plot(itrs, lmean, label=n, color=cmap(c[i]))
			ax.plot(itrs, err_up, color=cmap(c[i]), linestyle='--', linewidth=0.5)
			ax.plot(itrs, err_low, color=cmap(c[i]), linestyle='--', linewidth=0.5)
	ax.grid(True, which='both', linestyle='dotted')
	ax.set_title(title)
	ax.set_xlabel('Iterations')
	ax.legend(loc=0)
	fig.savefig(log_path+'/'+title+'.pdf')
	plt.close(fig)

	with open(log_path+'/'+title+'.cpk', 'wb+') as fs:
		pk.dump([itrs, input_list], fs)

'''
Train Ganist
'''
def train_ganist(ganist, im_data, labels=None):
	### dataset definition
	train_size = im_data.shape[0]

	### training configs
	max_itr_total = 3e3
	d_updates = 5
	g_updates = 1
	batch_size = 32
	eval_step = eval_int
	draw_step = eval_int
	snap_step = max_itr_total // 5

	### logs initi
	g_logs = list()
	d_r_logs = list()
	d_g_logs = list()
	eval_logs = list()
	stats_logs = list()
	norms_logs = list()
	itrs_logs = list()
	rl_vals_logs = list()
	rl_pvals_logs = list()
	en_acc_logs = list()
	g_sim_list = list()
	g_nz_list = list()
	d_sim_list = list()
	d_nz_list = list()

	### training inits
	d_itr = 0
	g_itr = 0
	itr_total = 0
	epoch = 0
	fid_best = 1000
	batch_end = train_size
	fetch_batch = True
	d_update_flag = True
	widgets = ["Ganist", Percentage(), Bar(), ETA()]
	pbar = ProgressBar(maxval=max_itr_total, widgets=widgets)
	pbar.start()

	while True:
		pbar.update(itr_total)
		if fetch_batch:
			if batch_end >= train_size:
				epoch += 1
				print('>>> Epoch {} started.'.format(epoch))
				train_dataset, _, train_labs = shuffle_data(im_data, label=labels)
				batch_start = 0
			else:
				batch_start = batch_end
			batch_end = batch_start + batch_size
			batch_data = train_dataset[batch_start:batch_end, ...]
			fetch_batch = False
		
		### save network once per snap step total iterations
		if itr_total % snap_step == 0 or itr_total == max_itr_total:
			ganist.save(log_path_snap+'/model_{}_{}.h5'.format(g_itr, itr_total))

		if itr_total % eval_step == 0 or itr_total == max_itr_total:
			### evaluate FID distance between real and gen distributions
			itrs_logs.append(itr_total)
			draw_path = log_path_draw+'/gen_sample_%d' % itr_total if itr_total % draw_step == 0 \
				else None
			fid_dist, net_stats = eval_ganist(ganist, train_dataset, draw_path)
			eval_logs.append(fid_dist)
			stats_logs.append(net_stats)

			### log g layer stats
			#g_sim, g_nz = ganist.step(None, None, g_layer_stats=True)
			#g_sim_list.append(g_sim)
			#g_nz_list.append(g_nz)
			
			### log d layer stats
			#d_sim, d_nz = ganist.step(None, None, d_layer_stats=True)
			#d_sim_list.append(d_sim)
			#d_nz_list.append(d_nz)

			### log norms
			#d_sample_size = 100
			#_, grad_norms = run_ganist_disc(ganist, 
			#	train_dataset[0:d_sample_size, ...], batch_size=256)
			#norms_logs.append([np.max(grad_norms), np.mean(grad_norms), np.std(grad_norms)])

			### log rl vals and pvals **g_num**
			#rl_vals_logs.append(list(ganist.g_rl_vals))
			#rl_pvals_logs.append(list(ganist.g_rl_pvals))
			#z_pr = np.exp(ganist.pg_temp * ganist.g_rl_pvals)
			#z_pr = z_pr / np.sum(z_pr)
			#rl_pvals_logs.append(list(z_pr))

			### en_accuracy plots **g_num**
			#acc_array = np.zeros(ganist.g_num)
			#sample_size = 1000
			#for g in range(ganist.g_num):
			#	z = g * np.ones(sample_size)
			#	z = z.astype(np.int32)
			#	g_samples = sample_ganist(ganist, sample_size, z_data=z)
			#	acc_array[g] = eval_en_acc(ganist, g_samples, z)
			#en_acc_logs.append(list(acc_array))

			### draw real samples en classified **g_num**
			#d_sample_size = 1000
			#im_true_color = im_color_borders(train_dataset[:d_sample_size], 
			#	train_labs[:d_sample_size], max_label=9)
			#im_block_draw(im_true_color, 10, draw_path+'_t.png', 
			#	im_labels=train_labs[:d_sample_size])
			#im_block_draw(train_dataset[:d_sample_size], 10, draw_path+'_t.png', 
			#	im_labels=train_labs[:d_sample_size], ganist=ganist)

			### en_preds
			#en_preds = np.argmax(en_logits, axis=1)
			#en_order = np.argsort(en_preds)
			#preds_order = en_preds[en_order]
			#logits_order = en_logits[en_order]
			#labels_order = train_labs[en_order]
			#start = 0
			#preds_list = list()
			#for i, v in enumerate(labels_order):
			#	if i == len(labels_order)-1 or preds_order[i] < preds_order[i+1]:
			#		preds_list.append(list(labels_order[start:i+1]))
			#		start = i+1
			#
			#print '>>> EN_LABELS:'
			#for l in preds_list:
			#	print l

			### plots
			eval_logs_mat = np.array(eval_logs)
			stats_logs_mat = np.array(stats_logs)
			#norms_logs_mat = np.array(norms_logs)
			rl_vals_logs_mat = np.array(rl_vals_logs)
			rl_pvals_logs_mat = np.array(rl_pvals_logs)
			#en_acc_logs_mat = np.array(en_acc_logs)

			if len(stats_logs) > 1:
				stats_logs_names = ['nan_vars_ratio', 'inf_vars_ratio', 'tiny_vars_ratio', 
								'big_vars_ratio']
				plot_time_mat(stats_logs_mat, stats_logs_names, 1, log_path, itrs=itrs_logs)

			if len(eval_logs) > 1:
				fig, ax = plt.subplots(figsize=(8, 6))
				ax.clear()
				ax.plot(itrs_logs, eval_logs_mat, color='b')
				ax.grid(True, which='both', linestyle='dotted')
				ax.set_title('FID')
				ax.set_xlabel('Iterations')
				ax.set_ylabel('Values')
				ax.legend(loc=0)
				fig.savefig(log_path+'/fid_dist.png', dpi=300)
				plt.close(fig)
				### save eval_logs
				with open(log_path+'/fid_logs.cpk', 'wb+') as fs:
					pk.dump([itrs_logs, eval_logs_mat], fs)
			
			#if len(norms_logs) > 1:
			#	fig, ax = plt.subplots(figsize=(8, 6))
			#	ax.clear()
			#	ax.plot(itrs_logs, norms_logs_mat[:,0], color='r', label='max_norm')
			#	ax.plot(itrs_logs, norms_logs_mat[:,1], color='b', label='mean_norm')
			#	ax.plot(itrs_logs, norms_logs_mat[:,1]+norms_logs_mat[:,2], color='b', linestyle='--')
			#	ax.plot(itrs_logs, norms_logs_mat[:,1]-norms_logs_mat[:,2], color='b', linestyle='--')
			#	ax.grid(True, which='both', linestyle='dotted')
			#	ax.set_title('Norm Grads')
			#	ax.set_xlabel('Iterations')
			#	ax.set_ylabel('Values')
			#	ax.legend(loc=0)
			#	fig.savefig(log_path+'/norm_grads.png', dpi=300)
			#	plt.close(fig)
			#	with open(log_path+'/norm_grads.cpk', 'wb+') as fs:
			#		pk.dump(norms_logs_mat, fs)

			### plot rl_vals **g_num**
			#if len(rl_vals_logs) > 1:
			#	fig, ax = plt.subplots(figsize=(8, 6))
			#	ax.clear()
			#	for g in range(ganist.g_num):
			#		ax.plot(itrs_logs, rl_vals_logs_mat[:, g], 
			#			label='g_%d' % g, c=global_color_set[g])
			#	ax.grid(True, which='both', linestyle='dotted')
			#	ax.set_title('RL Q Values')
			#	ax.set_xlabel('Iterations')
			#	ax.set_ylabel('Values')
			#	ax.legend(loc=0)
			#	fig.savefig(log_path+'/rl_q_vals.png', dpi=300)
			#	plt.close(fig)
			
			### plot rl_pvals **g_num**
			#if len(rl_pvals_logs) > 1:
			#	fig, ax = plt.subplots(figsize=(8, 6))
			#	ax.clear()
			#	for g in range(ganist.g_num):
			#		ax.plot(itrs_logs, rl_pvals_logs_mat[:, g], 
			#			label='g_%d' % g, c=global_color_set[g])
			#	ax.grid(True, which='both', linestyle='dotted')
			#	ax.set_title('RL Policy')
			#	ax.set_xlabel('Iterations')
			#	ax.set_ylabel('Values')
			#	ax.legend(loc=0)
			#	fig.savefig(log_path+'/rl_policy.png', dpi=300)
			#	plt.close(fig)
			#	### save pval_logs
			#	with open(log_path+'/rl_pvals.cpk', 'wb+') as fs:
			#		pk.dump([itrs_logs, rl_pvals_logs_mat], fs)

			### plot en_accs **g_num**
			#if len(en_acc_logs) > 1:
			#	fig, ax = plt.subplots(figsize=(8, 6))
			#	ax.clear()
			#	for g in range(ganist.g_num):
			#		ax.plot(itrs_logs, en_acc_logs_mat[:, g], 
			#			label='g_%d' % g, c=global_color_set[g])
			#	ax.grid(True, which='both', linestyle='dotted')
			#	ax.set_title('Encoder Accuracy')
			#	ax.set_xlabel('Iterations')
			#	ax.set_ylabel('Values')
			#	ax.legend(loc=0)
			#	fig.savefig(log_path+'/encoder_acc.png', dpi=300)
			#	plt.close(fig)

			### plot layer stats
			#if len(g_sim_list) > 1:
			#	plot_layer_stats(g_sim_list, itrs_logs, 'g_sim', log_path)
			#if len(g_nz_list) > 1:
			#	plot_layer_stats(g_nz_list, itrs_logs, 'g_nz', log_path, beta_conf=0)
			#if len(d_sim_list) > 1:
			#	plot_layer_stats(d_sim_list, itrs_logs, 'd_sim', log_path)
			#if len(d_nz_list) > 1:
			#	plot_layer_stats(d_nz_list, itrs_logs, 'd_nz', log_path, beta_conf=0)

			### save best model
			if fid_dist < fid_best:
				fid_best = fid_dist
				fid_best_itr = itr_total
				ganist.save(log_path_snap+'/model_best.h5')
				with open(log_path+'/model_best_itr.txt', 'w+') as fs:
					print >> fs, '{}'.format(fid_best_itr)

			if itr_total == max_itr_total:
				break

		### discriminator update
		if d_update_flag is True:
			batch_sum, batch_g_data = ganist.step(batch_data, 
				batch_size=None, gen_update=False, run_count=itr_total)
			ganist.write_sum(batch_sum, itr_total)
			d_itr += 1
			itr_total += 1
			d_update_flag = False if d_itr % d_updates == 0 else True
			fetch_batch = True
		
		### generator updates: g_updates times for each d_updates of discriminator
		elif g_updates > 0:
			batch_sum, batch_g_data = ganist.step(batch_data, 
				batch_size=None, gen_update=True, run_count=itr_total)
			ganist.write_sum(batch_sum, itr_total)
			g_itr += 1
			itr_total += 1
			d_update_flag = True if g_itr % g_updates == 0 else False
		
		### only happens if g_updates = 0
		else:
			d_update_flag = True

'''
Sample sample_size data points from ganist.
'''
def sample_ganist(ganist, sample_size, sampler=None, batch_size=64, 
	z_data=None, zi_data=None, z_im=None, output_type='rec'):
	sampler = sampler if sampler is not None else ganist.step
	if output_type == 'l0':
		g_samples = np.zeros([sample_size, 32, 32, 3])
	elif output_type == 'l1':
		g_samples = np.zeros([sample_size, 64, 64, 3])
	else:
		g_samples = np.zeros([sample_size, 128, 128, 3])
	
	for batch_start in range(0, sample_size, batch_size):
		batch_end = batch_start + batch_size
		batch_len = g_samples[batch_start:batch_end, ...].shape[0]
		batch_z = z_data[batch_start:batch_end, ...] if z_data is not None else None
		batch_zi = zi_data[batch_start:batch_end, ...] if zi_data is not None else None
		batch_im = z_im[batch_start:batch_end, ...] if z_im is not None else None
		g_samples[batch_start:batch_end, ...] = \
			sampler(batch_im, batch_len, 
				gen_only=True, z_data=batch_z, zi_data=batch_zi, output_type=output_type)
	return g_samples

'''
Calculate FID score between two set of image Inception features.
Shape (N, 2048)
'''
def compute_fid(feat1, feat2):
	num = feat1.shape[0]
	m1 = np.mean(feat1, axis=0)
	m2 = np.mean(feat2, axis=0)
	cov1 = (feat1 - m1).T.dot(feat1 - m1) / (num - 1.)
	cov2 = (feat2 - m2).T.dot(feat2 - m2) / (num - 1.)
	csqrt = scipy.linalg.sqrtm(cov1.dot(cov2))
	fid2 = np.sum(np.square(m1 - m2)) + np.trace(cov1 + cov2 - 2.*csqrt)
	return np.sqrt(fid2)

'''
Extract inception final pool features from pretrained inception v3 model on imagenet
'''
def extract_inception_feat(sess, feat_layer, im_layer, im_data):
	data_size = im_data.shape[0]
	batch_size = 64
	feat_size = feat_layer.get_shape().as_list()[-1]
	im_feat = np.zeros((data_size, feat_size))
	### forward on inception v3
	widgets = ["InceptionV3", Percentage(), Bar(), ETA()]
	pbar = ProgressBar(maxval=data_size, widgets=widgets)
	pbar.start()
	for batch_start in range(0, data_size, batch_size):
		pbar.update(batch_start)
		batch_end = batch_start + batch_size
		pe = sess.run(feat_layer, {im_layer: im_data[batch_start:batch_end, ...]})
		im_feat[batch_start:batch_end, ...] = pe.reshape((-1, feat_size))
	return im_feat

def blur_images(imgs, sigma, blur_type='gauss'):
	if sigma==0:
		return imgs
	if blur_type == 'avg':
		return TFutil.get().blur_images(imgs, sigma)
	### kernel
	t = np.linspace(-20, 20, 41)
	#t = np.linspace(-20, 20, 81) ## for 128x128 images
	bump = np.exp(0.5 * -t**2/sigma**2)
	bump /= np.trapz(bump) # normalize the integral to 1
	kernel = bump[:, np.newaxis] * bump[np.newaxis, :]
	imgs_blur = np.array(imgs)
	for i in range(imgs.shape[0]):
		imgs_blur[i, ...] = signal.fftconvolve(imgs[i, ...], kernel[:, :, np.newaxis], mode='same')
	return imgs_blur

def apply_lap_pyramid(im, batch_size=64):
	im_size, h, w, c = im.shape
	### build op
	im_layer = tf.placeholder(tf_ganist.tf_dtype, [None, h, w, c])
	pyramid = tf_ganist.tf_make_lap_pyramid(im_layer)
	reconst = tf_ganist.tf_reconst_lap_pyramid(pyramid)
	### apply op
	im_reconst = np.zeros((im_size, h, w, c))
	for batch_start in range(0, im_size, batch_size):
		batch_end = min(batch_start + batch_size, im_size)
		im_batch = im[batch_start:batch_end]
		im_reconst[batch_start:batch_end, ...] = sess.run(reconst, {im_layer: im_batch})
	return im_reconst

class TFutil:
	__instance = None
	@staticmethod
	def get():
		if TFutil.__instance == None:
			raise Exception('TFutil is not initialized.')
		return TFutil.__instance
	
	def __init__(self, sess):
		if TFutil.__instance != None:
			raise Exception('TFutil is a singleton class.')
		else:
			TFutil.__instance = self
			self.sess = sess
			self.blur_dict = dict()
			self.im_dict = dict()
			self.upsample_dict = dict()
	
	def blur_images(self, imgs, ksize, batch_size=64):
		if ksize == 1 or ksize == 0:
			return imgs
		imgs_size = imgs.shape[0]
		imgs_blur = np.array(imgs)
		if imgs.shape[1:] not in self.im_dict:
			self.im_dict[imgs.shape[1:]] = \
				tf.placeholder(tf.float32, (None,)+imgs.shape[1:])
		im_layer = self.im_dict[imgs.shape[1:]]
		if ksize not in self.blur_dict:
			self.blur_dict[ksize] = tf.nn.avg_pool(im_layer, 
				ksize=[1, ksize, ksize, 1], strides=[1, 1, 1, 1], padding='SAME')
		blur_layer = self.blur_dict[ksize]
		for batch_start in range(0, imgs_size, batch_size):
			batch_end = min(batch_start+batch_size, imgs_size)
			imgs_blur[batch_start:batch_end, ...] = self.sess.run(blur_layer, 
				feed_dict={im_layer: imgs[batch_start:batch_end, ...]})
		return imgs_blur

	def upsample(self, imgs, times=1, batch_size=64):
		imgs_size, h, w, c = imgs.shape
		imgs_us = np.zeros((imgs_size, h*2**times, w*2**times, c))
		im_key = imgs.shape[1:]
		if im_key not in self.im_dict:
			self.im_dict[im_key] = \
				tf.placeholder(tf.float32, (None,)+imgs.shape[1:])
		im_layer = self.im_dict[im_key]
		
		us_key = imgs.shape[1:]+(times,)
		if us_key not in self.upsample_dict:
			im_x = im_layer
			for i in range(times):
				im_x = tf_ganist.tf_upsample(im_x)
			self.upsample_dict[us_key] = im_x
		us_layer = self.upsample_dict[us_key]
		
		for batch_start in range(0, imgs_size, batch_size):
			batch_end = min(batch_start+batch_size, imgs_size)
			imgs_us[batch_start:batch_end, ...] = self.sess.run(us_layer, 
				feed_dict={im_layer: imgs[batch_start:batch_end, ...]})
		return imgs_us

'''
Evaluate fid on data
'''
def eval_fid(sess, im_r, im_g, blur=0):
	### extract real images (at least data_size)
	feat_r = extract_inception_feat(sess, inception_feat_layer, inception_im_layer, blur_images(im_r, blur))
	### extract fake images
	feat_g = extract_inception_feat(sess, inception_feat_layer, inception_im_layer, blur_images(im_g, blur))
	### compute fid
	return compute_fid(feat_r, feat_g)

def eval_fid_levels(sess, im_r, im_g, blur_levels):
	fid_list = list()
	for b in blur_levels:
		print('>>> Computing FID Level {}'.format(b))
		fid_list.append(eval_fid(sess, im_r, im_g, b))
	return fid_list

'''
Run discriminator of ganist on the given im_data, return logits and gradient norms. **g_num**
'''
def run_ganist_disc(ganist, im_data, sampler=None, batch_size=64, z_data=None):
	sampler = sampler if sampler is not None else ganist.step
	sample_size = im_data.shape[0]
	logits = np.zeros(sample_size)
	grad_norms = np.zeros(sample_size)
	for batch_start in range(0, sample_size, batch_size):
		batch_end = batch_start + batch_size
		batch_z = z_data[batch_start:batch_end, ...] if z_data is not None else None
		batch_im = im_data[batch_start:batch_end, ...]
		batch_logits, batch_grad_norms = sampler(batch_im, None, dis_only=True, z_data=batch_z)
		logits[batch_start:batch_end] = batch_logits
		grad_norms[batch_start:batch_end] = batch_grad_norms
	return logits, grad_norms

'''
Evaluate encoder logits on the given dataset.
'''
def eval_ganist_en(ganist, im_data, batch_size=64):
	sample_size = im_data.shape[0]
	en_logits = np.zeros([sample_size, ganist.g_num])
	for batch_start in range(0, sample_size, batch_size):
		batch_end = batch_start + batch_size
		batch_im = im_data[batch_start:batch_end, ...]
		en_logits[batch_start:batch_end, ...] = \
			ganist.step(batch_im, None, en_only=True)
	return en_logits

'''
Evaluate encoder accuracy on the given dataset.
'''
def eval_en_acc(ganist, im_data, im_label, batch_size=64):
	en_logits = eval_ganist_en(ganist, im_data, batch_size)
	acc = np.mean((np.argmax(en_logits, axis=1) - im_label) == 0)
	return acc

'''
Returns the energy distance of a trained GANist, and draws block images of GAN samples
'''
def eval_ganist(ganist, im_data, draw_path=None, sampler=None):
	### sample and batch size
	sample_size = 1000
	batch_size = 64
	draw_size = 5
	sampler = sampler if sampler is not None else ganist.step
	
	### collect real and gen samples **mt**
	r_samples = im_data[0:sample_size, ...]
	g_samples = sample_ganist(ganist, sample_size, sampler=sampler)

	### draw block image of gen samples
	if draw_path is not None:
		g_samples = g_samples.reshape((-1,) + im_data.shape[1:])
		im_block_draw(g_samples, draw_size, draw_path+'.png', border=True)
		sample_pyramid(ganist, draw_path+'_pyramid.png', 10)

	### get network stats
	net_stats = ganist.step(None, None, stats_only=True)

	### fid
	fid = eval_fid(ganist.sess, r_samples, g_samples)
	#fid = 0

	return fid, net_stats

'''
Runs eval_modes and store the results in pathname
'''
def mode_analysis(mnet, im_data, pathname, labels=None, draw_list=None, draw_name='gen'):
	mode_num, mode_count, mode_vars = eval_modes(mnet, im_data, labels, draw_list, draw_name)
	with open(pathname, 'wb+') as fs:
		pk.dump([mode_num, mode_count, mode_vars], fs)
	return mode_num, mode_count, mode_vars

'''
Returns #modes in stacked mnist, #im per modes, and average distance per mode.
Images im_data shape is (N, 28, 28, ch)
>>> dataset sensitive: class_size, chennels predictions, knn
'''
def eval_modes(mnet, im_data, labels=None, draw_list=None, draw_name='gen'):
	batch_size = 64
	mode_threshold = 0
	pr_threshold = 0.8
	data_size = im_data.shape[0]
	channels = im_data.shape[-1]
	### **cifar**
	#class_size = 10**channels
	class_size = 2
	knn = 6 * channels
	im_class_ids = dict((i, list()) for i in range(class_size))
	print '>>> Mode Eval Started'
	widgets = ["Eval_modes", Percentage(), Bar(), ETA()]
	pbar = ProgressBar(maxval=data_size, widgets=widgets)
	pbar.start()
	if labels is None:
		### classify images into modes
		for batch_start in range(0, data_size, batch_size):
			pbar.update(batch_start)
			batch_end = batch_start + batch_size
			batch_len = batch_size if batch_end < data_size else data_size - batch_start
			preds = np.zeros(batch_len)
			### channels predictions (nan if less than pr_threshold)
			'''
			for ch in range(channels):
				batch_data = im_data[batch_start:batch_end, ..., ch][..., np.newaxis]
				preds_pr = mnet.step(batch_data, pred_only=True)
				preds_id = np.argmax(preds_pr, axis=1)
				preds[np.max(preds_pr, axis=1) < pr_threshold] = np.nan
				preds += 10**ch * preds_id
			'''
			### prediction **cifar**
			batch_data = im_data[batch_start:batch_end, ...]
			preds_pr = mnet.step(batch_data, pred_only=True)
			preds_id = np.argmax(preds_pr, axis=1)
			preds[np.max(preds_pr, axis=1) < pr_threshold] = np.nan
			preds += preds_id
			
			### put each image id into predicted class list
			for i, c in enumerate(preds):
				if not np.isnan(c):
					im_class_ids[c].append(batch_start+i)
	else:
		### put each image id into predicted class list
		print 'labs>>> ', labels[0:10]
		for i, c in enumerate(labels):
			im_class_ids[c].append(i)

	### draw samples from modes
	if draw_list is not None:
		print '>>> Mode Draw Started'
		for c in draw_list:
			l = im_class_ids[c]
			if len(l) >= 25:
				im_block_draw(im_data[l, ...], 5, 
					log_path_draw+'/'+draw_name+'_class_%d.png' % c)

	### analyze modes
	print '>>> Mode Var Started'
	mode_count = np.zeros(class_size) 
	mode_vars = np.zeros(class_size)
	widgets = ["Var_modes", Percentage(), Bar(), ETA()]
	pbar = ProgressBar(maxval=class_size, widgets=widgets)
	pbar.start()
	for c, l in im_class_ids.items():
		pbar.update(c)
		mode_count[c] = len(l)
		mode_vars[c] = eval_mode_var(im_data[l, ...], knn) if len(l) > knn else 0.0
	print '>>> mode count: ', mode_count
	return np.sum(mode_count > mode_threshold), mode_count, mode_vars

'''
Return average pairwise iso-distance of im_data
'''
def eval_mode_var(im_data, n_neighbors, n_jobs=12):
	max_size = 1000
	im_data = im_data[0:max_size, ...]
	### preprocess images
	eval_data = im_data.reshape((im_data.shape[0], -1))
	### calculate isomap
	nns = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', n_jobs=n_jobs)
	nns.fit(eval_data)
	kng = kneighbors_graph(nns, n_neighbors, mode='distance', n_jobs=n_jobs)
	### calculate shortest path matrix
	d_mat = graph_shortest_path(kng, method='auto', directed=False)
	### calculate variance
	d_tri = np.tril(d_mat)
	count = np.sum(d_tri > 0)
	print '>>> Mode zero distance counts: ', 2. * count / (d_mat.shape[0]**2 - d_mat.shape[0])
	#d_var = 2.0 * np.sum(d_tri ** 2) / count
	d_var = 2.0 * np.sum(d_tri ** 2) / max_size
	return d_var

'''
Calculate and plot the percentage of samples classified with high confidence
'''
def eval_sample_quality(mnet, im_data, pathname):
	batch_size = 64
	th_size = 100
	data_size = im_data.shape[0]
	channels = im_data.shape[-1]

	### classify images into modes
	preds_pr = np.zeros((data_size, mnet.num_class))
	for batch_start in range(0, data_size, batch_size):
		batch_end = batch_start + batch_size
		batch_len = batch_size if batch_end < data_size else data_size - batch_start
		### cifar prediction **cifar**
		batch_data = im_data[batch_start:batch_end, ...]
		preds_pr[batch_start:batch_end, ...] = mnet.step(batch_data, pred_only=True)
	
	### compute percentage of high threshold samples
	threshold_list = np.arange(th_size + 1) * 1. / th_size
	high_conf = np.zeros(th_size+1)
	for i, th in enumerate(threshold_list):
		lower_sum = np.sum(np.max(preds_pr, axis=1) < th)
		high_conf[i] = 1. - 1. * lower_sum / data_size

	### plot sample quality ratio **g_num**
	fig, ax = plt.subplots(figsize=(8, 6))
	ax.clear()
	ax.plot(threshold_list, high_conf)
	ax.grid(True, which='both', linestyle='dotted')
	ax.set_title('Sample Quality')
	ax.set_xlabel('Confidence')
	ax.set_ylabel('Sample Ratio')
	fig.savefig(pathname+'.png', dpi=300)
	plt.close(fig)
	### save pval_logs
	with open(pathname+'.cpk', 'wb+') as fs:
		pk.dump([threshold_list, high_conf], fs)

'''
Draw gan specific manifold samples **g_num**
'''
def gset_sample_draw(ganist, block_size):
	sample_size = block_size ** 2
	for g in range(ganist.g_num):
		z_data = g * np.ones(sample_size, dtype=np.int32)
		samples = sample_ganist(ganist, sample_size, z_data=z_data)
		im_block_draw(samples, block_size, log_path_draw+'/g_%d_manifold' % g, ganist=ganist)

def train_mnist_net(mnet, im_data, labels, eval_im_data=None, eval_labels=None):
	### dataset definition
	train_size = im_data.shape[0]

	### logs initi
	eval_logs = list()
	itrs_logs = list()

	### training configs
	max_itr_total = 5e5
	batch_size = 64
	eval_step = 100

	### training inits
	itr_total = 0
	epoch = 0
	widgets = ["Mnist_net", Percentage(), Bar(), ETA()]
	pbar = ProgressBar(maxval=max_itr_total, widgets=widgets)
	pbar.start()

	order = np.arange(train_size)
	while itr_total < max_itr_total:
		### get a rgb stacked mnist dataset
		np.random.shuffle(order)
		im_data_sh = im_data[order, ...]
		labels_sh = labels[order, ...]
		epoch += 1
		print ">>> Epoch %d started..." % epoch
		### train one epoch
		for batch_start in range(0, train_size, batch_size):
			if itr_total >= max_itr_total:
				break
			pbar.update(itr_total)
			batch_end = batch_start + batch_size
			### fetch batch data
			batch_data = im_data_sh[batch_start:batch_end, ...]
			batch_labels = labels_sh[batch_start:batch_end, ...]
			### train one step
			batch_preds, batch_acc, batch_sum = mnet.step(batch_data, batch_labels)
			mnet.write_sum(batch_sum, itr_total)
			itr_total += 1

			if itr_total % eval_step == 0 and eval_im_data is not None:
				eval_loss, eval_acc = eval_mnist_net(mnet, eval_im_data, eval_labels, batch_size)
				train_loss, train_acc = eval_mnist_net(mnet, im_data_sh[0:10000, ...], 
					labels_sh[0:10000], batch_size)
				eval_logs.append([eval_loss, eval_acc, train_loss, train_acc])
				itrs_logs.append(itr_total)
				print '>>> train_acc: ', train_acc
				print '>>> eval_acc: ', eval_acc

			if itr_total >= max_itr_total:
				break

		### save network every epoch
		mnet.save(c_log_path_snap+'/model_%d.h5' % itr_total)

		### plot mnet evaluation plot every epoch
		if len(eval_logs) < 2:
			continue
		eval_logs_mat = np.array(eval_logs)
		eval_logs_names = ['eval_loss', 'eval_acc', 'train_loss', 'train_acc']
		plot_time_mat(eval_logs_mat, eval_logs_names, 1, c_log_path, itrs=itrs_logs)
	return eval_loss, eval_acc

def eval_mnist_net(mnet, im_data, labels, batch_size):
	eval_size = im_data.shape[0]
	eval_sum = 0.0
	eval_loss = 0.0
	eval_count = 0.0

	### eval on all im_data
	for batch_start in range(0, eval_size, batch_size):
		batch_end = batch_start + batch_size
		### fetch batch data
		batch_data = im_data[batch_start:batch_end, ...]
		batch_labels = labels[batch_start:batch_end, ...]
		### eval one step
		batch_preds, batch_acc, batch_loss = mnet.step(batch_data, batch_labels, train=False)
		eval_sum += 1.0 * batch_acc * batch_data.shape[0]
		eval_loss += 1.0 * batch_loss * batch_data.shape[0]
		eval_count += batch_data.shape[0]

	return eval_loss / eval_count, eval_sum / eval_count

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

def eval_fft_layer(val, dft_size=None):
	val_agg = np.sum(val, axis=-1)
	if dft_size is None:
		val_ft = np.fft.fftn(val_agg)
		val_ft_s = np.flip(np.fft.fftshift(val_ft), 0)
	else:
		freqs = 1. * np.arange(dft_size) / dft_size
		val_ft_s, val_ft = compute_dft(val_agg, freqs)
	fft_power = np.abs(val_ft_s)**2
	return fft_power

'''
Computes and plots the fft of each conv layer in ganist.
dft_size: if not None, will apply dft without assuming continuous signal, using a base freq of 1/dft_size.
'''
def eval_fft(ganist, save_dir, dft_size=5):
	fft_vars = list()
	d_vars, g_vars = ganist.get_vars_array()
	fig = plt.figure(0, figsize=(8,6))
	for val, name in d_vars+g_vars:
		if 'kernel' in name and 'conv' in name:
			scopes = name.split('/')
			net_name = next(v for v in scopes if 'net' in  v)
			conv_name = next(v for v in scopes if 'conv' in  v)
			save_path = '{}/{}_{}'.format(save_dir, net_name, conv_name)
			layer_name = '{}_{}'.format(net_name, conv_name)
			print(name)
			print(val.shape)
			print(save_path)
			fft_mat = np.zeros((val.shape[3], val.shape[0], val.shape[1]))
			for i in range(val.shape[3]):
				fft_mat[i, ...] = eval_fft_layer(val[:, :, :, i], dft_size)
			print '>>> FFT MIN: ', fft_mat.min()
			print '>>> FFT MAX: ', fft_mat.max()
			fft_mean = np.mean(fft_mat, axis=0)
			### plot mean fft
			fig.clf()
			ax = fig.add_subplot(1,1,1)
			pa = ax.imshow(np.log(fft_mean), cmap=plt.get_cmap('hot'), vmin=-6, vmax=6)
			ax.set_title(layer_name)
			if dft_size is not None:
				ticks = [-(dft_size//2), 0, dft_size//2]
				ticks_loc = [0, dft_size//2, dft_size-dft_size%2]
				ax.set_xticks(ticks_loc)
				ax.set_xticklabels(ticks)
				ax.set_yticks(ticks_loc)
				ax.set_yticklabels(ticks[::-1])
			fig.colorbar(pa)
			fig.savefig(save_path+'.png', dpi=300)
			fft_vars.append((fft_mean, layer_name))
	with open('{}/eval_fft.cpk'.format(save_dir), 'wb+') as fs:
		pk.dump(fft_vars, fs)

if __name__ == '__main__':
	'''
	Script Setup
	'''
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('-l', '--log-path', dest='log_path', required=True, help='log directory to store logs.')
	arg_parser.add_argument('-e', '--eval', dest='eval_int', required=True, help='eval intervals.')
	arg_parser.add_argument('-s', '--seed', dest='seed', default=0, help='random seed.')
	args = arg_parser.parse_args()
	log_path = args.log_path
	eval_int = int(args.eval_int)
	run_seed = int(args.seed)
	np.random.seed(run_seed)
	tf.set_random_seed(run_seed)
	import tf_ganist
	import mnist_net
	import vae_ganist

	### global colormap set
	global_cmap = mat_cm.get_cmap('tab20')
	global_color_locs = np.arange(20) / 20.
	global_color_set = global_cmap(global_color_locs)

	### init setup
	mnist_stack_size = 1
	c_log_path = log_path+'/classifier'
	log_path_snap = log_path+'/snapshots'
	c_log_path_snap = c_log_path+'/snapshots'
	log_path_draw = log_path+'/draws'
	log_path_sum = log_path+'/sums'
	c_log_path_sum = c_log_path+'/sums'
	log_path_sample = log_path+'/samples/'

	log_path_vae = log_path+'/vae'
	log_path_draw_vae = log_path_vae+'/draws'
	log_path_snap_vae = log_path_vae+'/snapshots'
	log_path_sum_vae = log_path_vae+'/sums'

	os.system('mkdir -p '+log_path_snap)
	os.system('mkdir -p '+c_log_path_snap)
	os.system('mkdir -p '+log_path_draw)
	os.system('mkdir -p '+log_path_sum)
	os.system('mkdir -p '+c_log_path_sum)
	os.system('mkdir -p '+log_path_sample)
	os.system('mkdir -p '+log_path_draw_vae)
	os.system('mkdir -p '+log_path_snap_vae)
	os.system('mkdir -p '+log_path_sum_vae)

	### read and process data
	sample_size = 1000

	'''
	DATASET LOADING AND DRAWING
	'''
	### mnist dataset
	'''
	train_data, val_data, test_data = read_mnist(data_path)
	train_labs = train_data[1]
	train_imgs = im_process(train_data[0])
	val_labs = val_data[1]
	val_imgs = im_process(val_data[0])
	test_labs = test_data[1]
	test_imgs = im_process(test_data[0])
	all_labs = np.concatenate([train_labs, val_labs, test_labs], axis=0)
	all_imgs = np.concatenate([train_imgs, val_imgs, test_imgs], axis=0)
	'''
	### cifar dataset **cifar**
	'''
	cifar_batch_path= '/media/evl/Public/Mahyar/cifar_10/data_batch_%d'
	cifar_test_path= '/media/evl/Public/Mahyar/cifar_10/test_batch'
	cifar_data_list = list()
	cifar_labs_list = list()
	for i in range(1, 6):
		data, labs = read_cifar(cifar_batch_path % i)
		cifar_data_list.append(data)
		cifar_labs_list.append(labs)
	cifar_test_data, cifar_test_labs = read_cifar(cifar_test_path)
	train_labs = np.concatenate(cifar_labs_list, axis=0)
	train_imgs = np.concatenate(cifar_data_list, axis=0)
	all_labs = np.concatenate(cifar_labs_list+[cifar_test_labs], axis=0)
	all_imgs = np.concatenate(cifar_data_list+[cifar_test_data], axis=0)
	test_labs = val_labs = cifar_test_labs
	test_imgs = val_imgs = cifar_test_data
	'''

	###stl10 dataset
	'''
	stl_data_path_train = '/media/evl/Public/Mahyar/Data/stl10_binary/train_X.bin'
	stl_lab_path_train = '/media/evl/Public/Mahyar/Data/stl10_binary/train_y.bin'
	stl_data_path_test = '/media/evl/Public/Mahyar/Data/stl10_binary/test_X.bin'
	stl_lab_path_test = '/media/evl/Public/Mahyar/Data/stl10_binary/test_y.bin'
	train_imgs, train_labs = read_stl(stl_data_path_train, stl_lab_path_train)
	test_imgs, test_labs = read_stl(stl_data_path_test, stl_lab_path_test)
	val_labs = test_labs
	val_imgs = test_imgs
	all_labs = np.concatenate([train_labs, test_labs], axis=0)
	all_imgs = np.concatenate([train_imgs, test_imgs], axis=0)
	'''

	### lsun dataset
	'''
	data_size_train = 20000
	data_size_val = 300
	lsun_bed_path_train = '/media/evl/Public/Mahyar/Data/lsun/bedroom_train_imgs'
	lsun_bed_path_val = '/media/evl/Public/Mahyar/Data/lsun/bedroom_val_imgs'
	lsun_bridge_path_train = '/media/evl/Public/Mahyar/Data/lsun/bridge_train_imgs'
	lsun_bridge_path_val = '/media/evl/Public/Mahyar/Data/lsun/bridge_val_imgs'
	lsun_church_path_train = '/media/evl/Public/Mahyar/Data/lsun/church_outdoor_train_imgs'
	lsun_church_path_val = '/media/evl/Public/Mahyar/Data/lsun/church_outdoor_val_imgs'
	
	train_imgs_bed = read_lsun(lsun_bed_path_train, data_size_train)
	val_imgs_bed = read_lsun(lsun_bed_path_val, data_size_val)

	train_imgs_bridge = read_lsun(lsun_bridge_path_train, data_size_train)
	val_imgs_bridge = read_lsun(lsun_bridge_path_val, data_size_val)

	train_imgs_church = read_lsun(lsun_church_path_train, data_size_train)
	val_imgs_church = read_lsun(lsun_church_path_val, data_size_val)

	train_imgs = np.concatenate([train_imgs_bed, train_imgs_bridge, train_imgs_church], axis=0)
	train_labs = np.concatenate([0 * np.ones(data_size_train, dtype=np.int32),
								1 * np.ones(data_size_train, dtype=np.int32), 
								2 * np.ones(data_size_train, dtype=np.int32)], axis=0)
	val_imgs = np.concatenate([val_imgs_bed, val_imgs_bridge, val_imgs_church], axis=0)
	val_labs = np.concatenate([0 * np.ones(data_size_val, dtype=np.int32),
								1 * np.ones(data_size_val, dtype=np.int32), 
								2 * np.ones(data_size_val, dtype=np.int32)], axis=0)
	test_imgs = val_imgs
	test_labs = val_labs
	all_labs = np.concatenate([train_labs, test_labs], axis=0)
	all_imgs = np.concatenate([train_imgs, test_imgs], axis=0)

	print '>>> lsun train mean: ', np.mean(train_imgs, axis=(0,1,2))
	print '>>> lsun train std: ', np.std(train_imgs, axis=(0,1,2))
	'''
	### celeba lsun
	#data_size_train = 50000
	#data_size_val = 300
	#lsun_bed_path_train = '/media/evl/Public/Mahyar/Data/lsun/bedroom_train_imgs'
	#lsun_bed_path_val = '/media/evl/Public/Mahyar/Data/lsun/bedroom_val_imgs'
	#lsun_bridge_path_train = '/media/evl/Public/Mahyar/Data/lsun/bridge_train_imgs'
	#lsun_bridge_path_val = '/media/evl/Public/Mahyar/Data/lsun/bridge_val_imgs'
	#lsun_church_path_train = '/media/evl/Public/Mahyar/Data/lsun/church_outdoor_train_imgs'
	#lsun_church_path_val = '/media/evl/Public/Mahyar/Data/lsun/church_outdoor_val_imgs'
	#celeba_train = '/media/evl/Public/Mahyar/Data/celeba/img_align_celeba'
	#celeba_val = '/media/evl/Public/Mahyar/Data/celeba/img_align_celeba_val'
	
	#train_imgs_bed = read_lsun(lsun_bed_path_train, data_size_train)
	#val_imgs_bed = read_lsun(lsun_bed_path_val, data_size_val)

	#train_imgs_celeba = read_lsun(celeba_train, data_size_train)
	#val_imgs_celeba = read_lsun(celeba_val, data_size_val)

	#train_imgs = train_imgs_celeba
	#val_imgs = val_imgs_celeba
	
	#train_imgs = np.concatenate([train_imgs_bed, train_imgs_celeba], axis=0)
	#train_labs = np.concatenate([0 * np.ones(data_size_train, dtype=np.int32),
	#							1 * np.ones(data_size_train, dtype=np.int32)], axis=0)
	#val_imgs = np.concatenate([val_imgs_bed, val_imgs_celeba], axis=0)
	#val_labs = np.concatenate([0 * np.ones(data_size_val, dtype=np.int32),
	#							1 * np.ones(data_size_val, dtype=np.int32)], axis=0)
	#test_imgs = val_imgs
	#test_labs = val_labs
	#all_labs = np.concatenate([train_labs, test_labs], axis=0)
	#all_imgs = np.concatenate([train_imgs, test_imgs], axis=0)

	#print '>>> lsun train mean: ', np.mean(train_imgs, axis=(0,1,2))
	#print '>>> lsun train std: ', np.std(train_imgs, axis=(0,1,2))

	#all_imgs = train_imgs
	#all_labs = train_labs = None

	### read art dataset
	'''
	art_path = '/media/evl/Public/Mahyar/Data/art_images_annotated'
	all_imgs, all_labs = read_art(art_path)
	train_imgs = all_imgs
	train_labs = all_labs
	'''

	#all_imgs_stack, all_labs_stack = get_stack_mnist(all_imgs, all_labs, stack_size=mnist_stack_size)
	#im_block_draw(all_imgs_stack, 10, log_path_draw+'/true_samples.png', border=True)
	
	### read celeba 128
	im_dir = '/media/evl/Public/Mahyar/Data/celeba/img_align_celeba/'
	im_size = 128
	dataset_size = 2000
	im_paths = readim_path_from_dir(im_dir)
	np.random.shuffle(im_paths)
	im_data = readim_from_path(im_paths[:dataset_size], im_size, center_crop=(121, 89))
	all_imgs_stack = im_data
	train_imgs = im_data[sample_size:, ...]
	train_labs = None
	print all_imgs_stack.shape
	im_block_draw(all_imgs_stack[:25], 5, log_path+'/true_samples.png', border=True)

	'''
	TENSORFLOW SETUP
	'''
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
	config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
	sess = tf.Session(config=config)
	### init TFutil
	tfutil = TFutil(sess)
	### create a ganist instance
	ganist = tf_ganist.Ganist(sess, log_path_sum)
	### create mnist classifier
	#mnet = mnist_net.MnistNet(sess, c_log_path_sum)
	### init variables
	sess.run(tf.global_variables_initializer())
	### save network initially
	#ganist.save(log_path_snap+'/model_0_0.h5')
	with open(log_path+'/vars_count_log.txt', 'w+') as fs:
		print >>fs, '>>> g_vars: %d --- d_vars: %d' \
			% (ganist.g_vars_count, ganist.d_vars_count)
	### draw filtered real samples (blurred)
	#im_block_draw(ganist.step(all_imgs_stack[:25], 25, filter_only=True), 5, 
	#	log_path_draw+'/real_samples_lp.png', border=True)

	'''
	INCEPTION SETUP
	'''
	fid_im_size = 128
	inception_dir = '/media/evl/Public/Mahyar/Data/models/research/slim'
	ckpt_path = '/media/evl/Public/Mahyar/Data/inception_v3_model/kaggle/inception_v3.ckpt'
	sys.path.insert(0, inception_dir)

	#from inception.slim import slim
	import nets.inception_v3 as inception
	from nets.inception_v3 import inception_v3_arg_scope

	### images should be N*299*299*3 of values (-1,1)
	images_pl = tf.placeholder(tf_ganist.tf_dtype, [None, fid_im_size, fid_im_size, 3], name='input_proc_images')
	images_pl_re = tf.image.resize_bilinear(images_pl, [299, 299], align_corners=False)

	### build model
	with tf.contrib.slim.arg_scope(inception_v3_arg_scope()):
		logits, endpoints = inception.inception_v3(images_pl_re, is_training=False, num_classes=1001)
	
	### load trained model
	variables_to_restore = tf.contrib.slim.get_variables_to_restore(include=['InceptionV3'])
	saver = tf.train.Saver(variables_to_restore)
	saver.restore(sess, ckpt_path)

	inception_im_layer = images_pl
	inception_feat_layer = endpoints['AvgPool_1a']

	#fid_test = eval_fid(sess, train_imgs[:5000], train_imgs[5000:10000])
	#print '>>> FID TEST: ', fid_test

	'''
	GAN SETUP SECTION
	'''
	### train ganist
	train_ganist(ganist, train_imgs, train_labs)

	### load ganist
	load_path = log_path_snap+'/model_best.h5'
	#load_path = '/media/evl/Public/Mahyar/ganist_lsun_logs/layer_stats/23_logs_gandm_or_celeba128cc/run_{}/snapshots/model_best.h5'
	ganist.load(load_path.format(run_seed))

	'''
	GAN DATA EVAL
	'''
	#eval_fft(ganist, log_path_draw)
	gan_model = ganist#vae
	sampler = ganist.step#vae.step
	### sample gen data and draw **mt**
	g_samples = sample_ganist(gan_model, sample_size, sampler=sampler)
	print '>>> g_samples shape: ', g_samples.shape
	im_block_draw(g_samples, 5, log_path+'/gen_samples.png', border=True)
	im_separate_draw(g_samples[:1000], log_path_sample)
	sample_pyramid(ganist, log_path+'/gen_samples_pyramid.png')
	#sys.exit(0)

	'''
	Read data from ImageNet and BigGan
	'''
	#im_dir = '/media/evl/Public/Mahyar/Data/image_net/train_128'
	#im_size = 128
	#im_data = read_imagenet(im_dir, 20, im_size=128)
	##im_data_re = im_data[:, :10, ...].reshape((-1, im_size, im_size, 3))
	#im_data_re = im_data.reshape((-1, im_size, im_size, 3))
	#np.random.shuffle(im_data_re)
	#all_imgs_stack = im_data_re
	#print all_imgs_stack.shape
	#
	##biggan_dir = '/media/evl/Public/Mahyar/Data/image_net/biggan/all_class_samples'
	#biggan_dir = '/media/evl/Public/Mahyar/Data/image_net/biggan/per_class_samples'
	#im_size = 128
	##g_samples = read_lsun(biggan_dir, sample_size, im_size)
	#g_im_data = read_imagenet(biggan_dir, 20, im_size=im_size).reshape((-1, im_size, im_size, 3))
	#np.random.shuffle(g_im_data)
	#g_samples = g_im_data[:sample_size, ...]
	#print g_samples.shape

	'''
	Read data from LSUN and StyleGAN
	'''
	#im_dir = '/media/evl/Public/Mahyar/Data/lsun/cat/'
	#im_size = 256
	#im_paths = readim_path_from_dir(im_dir)
	#np.random.shuffle(im_paths)
	#im_data = readim_from_path(im_paths[:sample_size*2], im_size)
	#all_imgs_stack = im_data
	#print all_imgs_stack.shape
	#
	#stylegan_dir = '/media/evl/Public/Mahyar/Data/stylegan/cat/'
	#im_size = 256
	#g_im_paths = readim_path_from_dir(stylegan_dir)
	#np.random.shuffle(g_im_paths)
	#g_samples = readim_from_path(g_im_paths[:sample_size], im_size)
	#print g_samples.shape

	'''
	Read data from CelebA 128 and ProgGAN
	'''
	#im_dir = '/media/evl/Public/Mahyar/Data/celeba/img_align_celeba/'
	#im_size = 128
	#im_paths = readim_path_from_dir(im_dir)
	#np.random.shuffle(im_paths)
	#im_data = readim_from_path(im_paths[:sample_size*2], im_size)
	#all_imgs_stack = im_data
	#print all_imgs_stack.shape
	
	#prog_gan_dir = '/media/evl/Public/Mahyar/Data/stylegan/celeba_1024/'
	#im_size = 128
	#g_im_paths = readim_path_from_dir(prog_gan_dir)
	#np.random.shuffle(g_im_paths)
	#g_samples = readim_from_path(g_im_paths[:sample_size], im_size)
	#print g_samples.shape
	#im_separate_draw(g_samples[:1000], log_path_sample)

	#g_samples = apply_lap_pyramid(all_imgs_stack[sample_size:2*sample_size])
	#im_block_draw(g_samples, 5, log_path+'/lap_samples.png', border=True)

	'''
	Multi Level FID
	'''
	blur_levels = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
	#blur_levels = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
	### draw blurred images
	blur_im_list = list()
	blur_draw_size = 10
	for b in blur_levels:
		blur_im_list.append(blur_images(all_imgs_stack[:blur_draw_size], b))
	blur_im = np.stack(blur_im_list, axis=0)
	block_draw(blur_im, log_path+'/blur_im_samples.png', border=True)
	### compute multi level fid (second line for real data fid levels)
	fid_list = eval_fid_levels(sess, g_samples, all_imgs_stack[:sample_size], blur_levels)
	#fid_list = eval_fid_levels(sess, all_imgs_stack[:sample_size], all_imgs_stack[sample_size:2*sample_size], blur_levels)
	### plot fid_levels
	fig, ax = plt.subplots(figsize=(8, 6))
	ax.clear()
	ax.plot(blur_levels, fid_list)
	ax.grid(True, which='both', linestyle='dotted')
	ax.set_title('FID levels')
	ax.set_xlabel('Filter sigma')
	#ax.set_xlabel('Filter Size')
	ax.set_ylabel('FID')
	ax.set_xticks(blur_levels)
	#ax.legend(loc=0)
	fig.savefig(log_path+'/fid_levels.png', dpi=300)
	plt.close(fig)
	### save fids
	with open(log_path+'/fid_levels.cpk', 'wb+') as fs:
		pk.dump([blur_levels, fid_list], fs)

	sess.close()


