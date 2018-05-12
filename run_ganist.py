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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0, 1" for multiple

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
### >>> dataset sensitive: stack_size
mnist_stack_size = 1
c_log_path = log_path+'/classifier'
log_path_snap = log_path+'/snapshots'
c_log_path_snap = c_log_path+'/snapshots'
log_path_draw = log_path+'/draws'
log_path_sum = log_path+'/sums'
c_log_path_sum = c_log_path+'/sums'

log_path_vae = log_path+'/vae'
log_path_draw_vae = log_path_vae+'/draws'
log_path_snap_vae = log_path_vae+'/snapshots'
log_path_sum_vae = log_path_vae+'/sums'

os.system('mkdir -p '+log_path_snap)
os.system('mkdir -p '+c_log_path_snap)
os.system('mkdir -p '+log_path_draw)
os.system('mkdir -p '+log_path_sum)
os.system('mkdir -p '+c_log_path_sum)
os.system('mkdir -p '+log_path_draw_vae)
os.system('mkdir -p '+log_path_snap_vae)
os.system('mkdir -p '+log_path_sum_vae)

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
def gset_block_draw(ganist, sample_size, path, en_color=True):
	im_draw = np.zeros([ganist.g_num, sample_size]+ganist.data_dim)
	im_size = ganist.data_dim[0]
	for g in range(ganist.g_num):
		z_data = g * np.ones(sample_size, dtype=np.int32)
		im_draw[g, ...] = sample_ganist(ganist, sample_size, z_data=z_data)
	#im_draw = (im_draw + 1.0) / 2.0
	if en_color:
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
		im_draw = im_draw_color.reshape([top_g_count, sample_size]+ganist.data_dim[:-1]+[3])
	if en_color is True:
		en_block_draw(ganist, im_draw, path)
	else:
		block_draw(im_draw, path)

'''
Draws sample_size**2 randomly selected images from im_data.
If im_labels is provided: selects sample_size images for each im_label and puts in columns.
If ganist is provided: classifies selected images and adds color border.
im_data must have shape (imb, imh, imw, imc) with values in [-1,1].
'''
def im_block_draw(im_data, sample_size, path, im_labels=None, ganist=None):
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
	block_draw(im_draw_color.reshape([cols, rows, imh, imw, 3]), path)

'''
Adds a color border to im_data corresponding to its im_label.
im_data must have shape (imb, imh, imw, imc) with values in [-1,1].
'''
def im_color_borders(im_data, im_labels, max_label=None, color_map=None):
	fh = fw = 4
	imb, imh, imw, imc = im_data.shape
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
	rgb_colors_t = np.tile(rgb_colors.reshape((imb, 1, 1, 3)), (1, imh, imw, 1))

	### create mask
	box_mask = np.ones((imh, imw))
	box_mask[fh+1:imh-fh, fw+1:imw-fw] = 0.
	box_mask_t = np.tile(box_mask.reshape((1, imh, imw, 1)), (imb, 1, 1, 3))
	box_mask_inv = np.abs(box_mask_t - 1.)

	### apply mask
	im_data_border = im_data_t * box_mask_inv + rgb_colors_t * box_mask_t
	return im_data_border

'''
im_data should be a (columns, rows, imh, imw, imc).
im_data values should be in [0, 1].
If c is not 3 then draws first channel only.
'''
def block_draw(im_data, path, separate_channels=False):
	cols, rows, imh, imw, imc = im_data.shape
	### block shape
	im_draw = im_data.reshape([cols, imh*rows, imw, imc])
	im_draw = np.concatenate([im_draw[i, ...] for i in range(im_draw.shape[0])], axis=1)
	im_draw = (im_draw + 1.0) / 2.0
	### plots
	fig = plt.figure(0)
	fig.clf()
	if not separate_channels or im_draw.shape[-1] != 3:
		ax = fig.add_subplot(1, 1, 1)
		if im_draw.shape[-1] == 1:
			ims = ax.imshow(im_draw.reshape(im_draw.shape[:-1]))
		else:
			ims = ax.imshow(im_draw)
		ax.set_axis_off()
		#fig.colorbar(ims)
		fig.savefig(path, dpi=300)
	else:
		im_tmp = np.zeros(im_draw.shape)
		ax = fig.add_subplot(1, 3, 1)
		im_tmp[..., 0] = im_draw[..., 0]
		ax.set_axis_off()
		ax.imshow(im_tmp)

		ax = fig.add_subplot(1, 3, 2)
		im_tmp[...] = 0.0
		im_tmp[..., 1] = im_draw[..., 1]
		ax.set_axis_off()
		ax.imshow(im_tmp)

		ax = fig.add_subplot(1, 3, 3)
		im_tmp[...] = 0.0
		im_tmp[..., 2] = im_draw[..., 2]
		ax.set_axis_off()
		ax.imshow(im_tmp)

		fig.subplots_adjust(wspace=0, hspace=0)
		fig.savefig(path, dpi=300)

'''
Train Ganist
'''
def train_ganist(ganist, im_data, labels=None):
	### dataset definition
	train_size = im_data.shape[0]

	### training configs
	max_itr_total = 5e5
	d_updates = 5
	g_updates = 1
	batch_size = 32
	eval_step = eval_int
	draw_step = eval_int

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

	### training inits
	d_itr = 0
	g_itr = 0
	itr_total = 0
	epoch = 0
	d_update_flag = True
	widgets = ["Ganist", Percentage(), Bar(), ETA()]
	pbar = ProgressBar(maxval=max_itr_total, widgets=widgets)
	pbar.start()

	while itr_total < max_itr_total:
		### get a rgb stacked mnist dataset
		### >>> dataset sensitive: stack_size
		train_dataset, train_labs = get_stack_mnist(im_data, 
			labels=labels, stack_size=mnist_stack_size)
		epoch += 1
		print ">>> Epoch %d started..." % epoch

		### train one epoch
		for batch_start in range(0, train_size, batch_size):
			if itr_total >= max_itr_total:
				break
			pbar.update(itr_total)
			batch_end = batch_start + batch_size
			### fetch batch data
			batch_data = train_dataset[batch_start:batch_end, ...]
			fetch_batch = False
			while fetch_batch is False:
				### evaluate energy distance between real and gen distributions
				if itr_total % eval_step == 0:
					draw_path = log_path_draw+'/gen_sample_%d' % itr_total if itr_total % draw_step == 0 \
						else None
					e_dist, fid_dist, net_stats = eval_ganist(ganist, train_dataset, draw_path)
					#e_dist = 0 if e_dist < 0 else np.sqrt(e_dist)
					eval_logs.append([e_dist, fid_dist])
					stats_logs.append(net_stats)
					### log norms every epoch
					d_sample_size = 100
					_, grad_norms = run_ganist_disc(ganist, 
						train_dataset[0:d_sample_size, ...], batch_size=256)
					norms_logs.append([np.max(grad_norms), np.mean(grad_norms), np.std(grad_norms)])
					itrs_logs.append(itr_total)

					### log rl vals and pvals **g_num**
					rl_vals_logs.append(list(ganist.g_rl_vals))
					rl_pvals_logs.append(list(ganist.g_rl_pvals))
					#z_pr = np.exp(ganist.pg_temp * ganist.g_rl_pvals)
					#z_pr = z_pr / np.sum(z_pr)
					#rl_pvals_logs.append(list(z_pr))

					### en_accuracy plots **g_num**
					acc_array = np.zeros(ganist.g_num)
					sample_size = 1000
					for g in range(ganist.g_num):
						z = g * np.ones(sample_size)
						z = z.astype(np.int32)
						g_samples = sample_ganist(ganist, sample_size, z_data=z)
						acc_array[g] = eval_en_acc(ganist, g_samples, z)
					en_acc_logs.append(list(acc_array))

					### draw real samples en classified **g_num**
					d_sample_size = 1000
					#im_true_color = im_color_borders(train_dataset[:d_sample_size], 
					#	train_labs[:d_sample_size], max_label=9)
					#im_block_draw(im_true_color, 10, draw_path+'_t.png', 
					#	im_labels=train_labs[:d_sample_size])
					im_block_draw(train_dataset[:d_sample_size], 10, draw_path+'_t.png', 
						im_labels=train_labs[:d_sample_size], ganist=ganist)

					### en_preds
					'''
					en_preds = np.argmax(en_logits, axis=1)
					en_order = np.argsort(en_preds)
					preds_order = en_preds[en_order]
					logits_order = en_logits[en_order]
					labels_order = train_labs[en_order]
					start = 0
					preds_list = list()
					for i, v in enumerate(labels_order):
						if i == len(labels_order)-1 or preds_order[i] < preds_order[i+1]:
							preds_list.append(list(labels_order[start:i+1]))
							start = i+1

					print '>>> EN_LABELS:'
					for l in preds_list:
						print l
					'''

				### discriminator update
				if d_update_flag is True:
					batch_sum, batch_g_data = ganist.step(batch_data, batch_size=None, gen_update=False)
					ganist.write_sum(batch_sum, itr_total)
					d_itr += 1
					itr_total += 1
					d_update_flag = False if d_itr % d_updates == 0 else True
					fetch_batch = True
				### generator updates: g_updates times for each d_updates of discriminator
				elif g_updates > 0:
					batch_sum, batch_g_data = ganist.step(batch_data, batch_size=None, gen_update=True)
					ganist.write_sum(batch_sum, itr_total)
					g_itr += 1
					itr_total += 1
					d_update_flag = True if g_itr % g_updates == 0 else False

				if itr_total >= max_itr_total:
					break

		### save network every epoch
		ganist.save(log_path_snap+'/model_%d_%d.h5' % (g_itr, itr_total))

		### plot ganist evaluation plot every epoch **g_num**
		if len(eval_logs) < 2:
			continue
		eval_logs_mat = np.array(eval_logs)
		stats_logs_mat = np.array(stats_logs)
		norms_logs_mat = np.array(norms_logs)
		rl_vals_logs_mat = np.array(rl_vals_logs)
		rl_pvals_logs_mat = np.array(rl_pvals_logs)
		en_acc_logs_mat = np.array(en_acc_logs)

		eval_logs_names = ['fid_dist', 'fid_dist']
		stats_logs_names = ['nan_vars_ratio', 'inf_vars_ratio', 'tiny_vars_ratio', 
							'big_vars_ratio']
		plot_time_mat(eval_logs_mat, eval_logs_names, 1, log_path, itrs=itrs_logs)
		plot_time_mat(stats_logs_mat, stats_logs_names, 1, log_path, itrs=itrs_logs)
		
		### plot norms
		fig, ax = plt.subplots(figsize=(8, 6))
		ax.clear()
		ax.plot(itrs_logs, norms_logs_mat[:,0], color='r', label='max_norm')
		ax.plot(itrs_logs, norms_logs_mat[:,1], color='b', label='mean_norm')
		ax.plot(itrs_logs, norms_logs_mat[:,1]+norms_logs_mat[:,2], color='b', linestyle='--')
		ax.plot(itrs_logs, norms_logs_mat[:,1]-norms_logs_mat[:,2], color='b', linestyle='--')
		ax.grid(True, which='both', linestyle='dotted')
		ax.set_title('Norm Grads')
		ax.set_xlabel('Iterations')
		ax.set_ylabel('Values')
		ax.legend(loc=0)
		fig.savefig(log_path+'/norm_grads.png', dpi=300)
		plt.close(fig)
		
		### plot rl_vals **g_num**
		fig, ax = plt.subplots(figsize=(8, 6))
		ax.clear()
		for g in range(ganist.g_num):
			ax.plot(itrs_logs, rl_vals_logs_mat[:, g], label='g_%d' % g, c=global_color_set[g])
		ax.grid(True, which='both', linestyle='dotted')
		ax.set_title('RL Q Values')
		ax.set_xlabel('Iterations')
		ax.set_ylabel('Values')
		ax.legend(loc=0)
		fig.savefig(log_path+'/rl_q_vals.png', dpi=300)
		plt.close(fig)
		
		### plot rl_pvals **g_num**
		fig, ax = plt.subplots(figsize=(8, 6))
		ax.clear()
		for g in range(ganist.g_num):
			ax.plot(itrs_logs, rl_pvals_logs_mat[:, g], label='g_%d' % g, c=global_color_set[g])
		ax.grid(True, which='both', linestyle='dotted')
		ax.set_title('RL Policy')
		ax.set_xlabel('Iterations')
		ax.set_ylabel('Values')
		ax.legend(loc=0)
		fig.savefig(log_path+'/rl_policy.png', dpi=300)
		plt.close(fig)

		### plot en_accs **g_num**
		fig, ax = plt.subplots(figsize=(8, 6))
		ax.clear()
		for g in range(ganist.g_num):
			ax.plot(itrs_logs, en_acc_logs_mat[:, g], label='g_%d' % g, c=global_color_set[g])
		ax.grid(True, which='both', linestyle='dotted')
		ax.set_title('Encoder Accuracy')
		ax.set_xlabel('Iterations')
		ax.set_ylabel('Values')
		ax.legend(loc=0)
		fig.savefig(log_path+'/encoder_acc.png', dpi=300)
		plt.close(fig)

	### save norm_logs
	with open(log_path+'/norm_grads.cpk', 'wb+') as fs:
		pk.dump(norms_logs_mat, fs)

	### save pval_logs
	with open(log_path+'/rl_pvals.cpk', 'wb+') as fs:
		pk.dump([itrs_logs, rl_pvals_logs_mat], fs)

	### save eval_logs
	with open(log_path+'/eval_logs.cpk', 'wb+') as fs:
		pk.dump([itrs_logs, eval_logs_mat], fs)

'''
Train VAE Ganist
'''
def train_vae(vae, im_data):
	### dataset definition
	train_size = im_data.shape[0]

	### training configs
	max_itr_total = 3e5
	batch_size = 32
	eval_step = eval_int
	draw_step = eval_int

	### logs initi
	eval_logs = list()
	stats_logs = list()
	itrs_logs = list()

	### training inits
	itr_total = 0
	epoch = 0
	widgets = ["VAEGanist", Percentage(), Bar(), ETA()]
	pbar = ProgressBar(maxval=max_itr_total, widgets=widgets)
	pbar.start()

	while itr_total < max_itr_total:
		### get a rgb stacked mnist dataset
		### >>> dataset sensitive: stack_size
		train_dataset, _ = get_stack_mnist(im_data, stack_size=mnist_stack_size)
		epoch += 1
		print ">>> Epoch %d started..." % epoch
		### train one epoch
		for batch_start in range(0, train_size, batch_size):
			if itr_total >= max_itr_total:
				break
			pbar.update(itr_total)
			batch_end = batch_start + batch_size
			### fetch batch data
			batch_data = train_dataset[batch_start:batch_end, ...]
			fetch_batch = False

			### evaluate energy distance between real and gen distributions
			if itr_total % eval_step == 0:
				draw_path = log_path_draw_vae+'/vae_sample_%d' % itr_total if itr_total % draw_step == 0 else None
				e_dist, e_norm, net_stats = eval_ganist(vae, train_dataset, draw_path, vae.step_vae)
				e_dist = 0 if e_dist < 0 else np.sqrt(e_dist)
				eval_logs.append([e_dist, e_dist/np.sqrt(2.0*e_norm)])
				stats_logs.append(net_stats)
				itrs_logs.append(itr_total)

			batch_sum, batch_g_data = vae.step_vae(batch_data, batch_size=None, update=True)
			vae.write_sum(batch_sum, itr_total)
			itr_total += 1

		### save network every epoch
		vae.save(log_path_snap_vae+'/model_%d.h5' % itr_total)

		### plot vae evaluation plot every epoch
		if len(eval_logs) < 2:
			continue
		eval_logs_mat = np.array(eval_logs)
		stats_logs_mat = np.array(stats_logs)
		eval_logs_names = ['energy_distance', 'energy_distance_norm']
		stats_logs_names = ['nan_vars_ratio', 'inf_vars_ratio', 'tiny_vars_ratio', 
							'big_vars_ratio', 'vars_count']
		plot_time_mat(eval_logs_mat, eval_logs_names, 1, log_path_vae, itrs=itrs_logs)
		plot_time_mat(stats_logs_mat, stats_logs_names, 1, log_path_vae, itrs=itrs_logs)

'''
Sample sample_size data points from ganist.
'''
def sample_ganist(ganist, sample_size, sampler=None, batch_size=64, 
	z_data=None, zi_data=None, z_im=None):
	sampler = sampler if sampler is not None else ganist.step
	g_samples = np.zeros([sample_size] + ganist.data_dim)
	for batch_start in range(0, sample_size, batch_size):
		batch_end = batch_start + batch_size
		batch_len = g_samples[batch_start:batch_end, ...].shape[0]
		batch_z = z_data[batch_start:batch_end, ...] if z_data is not None else None
		batch_zi = zi_data[batch_start:batch_end, ...] if zi_data is not None else None
		batch_im = z_im[batch_start:batch_end, ...] if z_im is not None else None
		g_samples[batch_start:batch_end, ...] = \
			sampler(batch_im, batch_len, gen_only=True, z_data=batch_z, zi_data=batch_zi)
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
	im_feat = np.zeros((data_size, 2048))
	### forward on inception v3
	widgets = ["InceptionV3", Percentage(), Bar(), ETA()]
	pbar = ProgressBar(maxval=data_size, widgets=widgets)
	pbar.start()
	for batch_start in range(0, data_size, batch_size):
		pbar.update(batch_start)
		batch_end = batch_start + batch_size
		pe = sess.run(feat_layer, {im_layer: im_data[batch_start:batch_end, ...]})
		im_feat[batch_start:batch_end, ...] = pe.reshape((-1, 2048))
	return im_feat

'''
Evaluate fid on data
'''
def eval_fid(sess, im_r, im_g):
	### extract real images (at least data_size)
	feat_r = extract_inception_feat(sess, inception_feat_layer, inception_im_layer, im_r)
	### extract fake images
	feat_g = extract_inception_feat(sess, inception_feat_layer, inception_im_layer, im_g)
	### compute fid
	return compute_fid(feat_r, feat_g)

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
	sample_size = 10000
	batch_size = 64
	draw_size = 10
	sampler = sampler if sampler is not None else ganist.step
	
	### collect real and gen samples **mt**
	r_samples = im_data[0:sample_size, ...]
	g_samples = sample_ganist(ganist, sample_size, sampler=sampler,
		z_im=im_data[-sample_size:, ...])
	
	### calculate energy distance
	#rr_score = np.mean(np.sqrt(np.sum(np.square( \
	#	r_samples[0:sample_size//2, ...] - r_samples[sample_size//2:, ...]), axis=1)))
	#gg_score = np.mean(np.sqrt(np.sum(np.square( \
	#	g_samples[0:sample_size//2, ...] - g_samples[sample_size//2:, ...]), axis=1)))
	#rg_score = np.mean(np.sqrt(np.sum(np.square( \
	#	r_samples[0:sample_size//2, ...] - g_samples[0:sample_size//2, ...]), axis=1)))

	### draw block image of gen samples
	if draw_path is not None:
		g_samples = g_samples.reshape((-1,) + im_data.shape[1:])
		### manifold interpolation drawing mode **mt** **g_num**
		'''
		gr_samples = im_data[-sample_size:, ...]
		gr_flip = np.array(gr_samples)
		for batch_start in range(0, sample_size, batch_size):
			batch_end = batch_start + batch_size
			gr_flip[batch_start:batch_end, ...] = np.flip(gr_flip[batch_start:batch_end, ...], axis=0)
		draw_samples = np.concatenate([g_samples, gr_samples, gr_flip], axis=3)
		im_block_draw(draw_samples, draw_size, draw_path)
		'''
		### **g_num**
		im_block_draw(g_samples, draw_size, draw_path+'.png', ganist=ganist)
		gset_block_draw(ganist, 10, draw_path+'_gset.png', en_color=True)

	### get network stats
	net_stats = ganist.step(None, None, stats_only=True)

	### fid
	fid = eval_fid(ganist.sess, r_samples, g_samples)

	return fid, fid, net_stats

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
	class_size = 10
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
	d_var = 2.0 * np.sum(d_tri ** 2) / count
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
	threshold_list = np.arange(th_size + 1) * 1. / th_size
	high_conf = np.zeros(th_size+1)
	for i, th in enumerate(threshold_list):
		isnan_sum = 0.
		for batch_start in range(0, data_size, batch_size):
			batch_end = batch_start + batch_size
			batch_len = batch_size if batch_end < data_size else data_size - batch_start
			preds = np.zeros(batch_len)
			### mnist channels predictions (nan if less than th)
			'''
			for ch in range(channels):
				batch_data = im_data[batch_start:batch_end, ..., ch][..., np.newaxis]
				preds_pr = mnet.step(batch_data, pred_only=True)
				preds_id = np.argmax(preds_pr, axis=1)
				preds[np.max(preds_pr, axis=1) < th] = np.nan
				preds += 10**ch * preds_id
			'''
			### cifar prediction **cifar**
			batch_data = im_data[batch_start:batch_end, ...]
			preds_pr = mnet.step(batch_data, pred_only=True)
			preds_id = np.argmax(preds_pr, axis=1)
			preds[np.max(preds_pr, axis=1) < th] = np.nan
			preds += preds_id

			isnan_sum += np.sum(np.isnan(preds))
		high_conf[i] = 1. - 1. * isnan_sum / data_size

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
	max_itr_total = 1e5
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

if __name__ == '__main__':
	### read and process data **cifar**
	### >>> dataset sensitive
	data_path = '/media/evl/Public/Mahyar/Data/mnist.pkl.gz'
	#stack_mnist_path = '/media/evl/Public/Mahyar/stack_mnist_350k.cpk'
	#stack_mnist_mode_path = '/media/evl/Public/Mahyar/mode_analysis_stack_mnist_350k.cpk'
	stack_mnist_path = '/media/evl/Public/Mahyar/mnist_70k.cpk'
	stack_mnist_mode_path = '/media/evl/Public/Mahyar/mode_analysis_mnist_70k.cpk'
	#class_net_path = '/media/evl/Public/Mahyar/Data/mnist_classifier/snapshots/model_100000.h5'
	class_net_path = '/media/evl/Public/Mahyar/Data/cifar_classifier/snapshots/model_100000.h5'
	#ganist_path = '/media/evl/Public/Mahyar/ganist_logs/logs_monet_126_with_pvals_saving/run_%d/snapshots/model_83333_500000.h5'
	#ganist_path = 'logs_c1_egreedy/snapshots/model_16628_99772.h5'
	sample_size = 10000
	#sample_size = 350000

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

	### draw true stacked mnist images
	### >>> dataset sensitive
	all_imgs_stack, all_labs_stack = get_stack_mnist(all_imgs, all_labs, stack_size=mnist_stack_size)
	print all_imgs_stack.shape
	print all_labs_stack.shape

	im_block_draw(all_imgs_stack, 10, log_path_draw+'/true_samples.png')
	
	'''
	TENSORFLOW SETUP
	'''
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
	config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
	sess = tf.Session(config=config)
	### create mnist classifier
	mnet = mnist_net.MnistNet(sess, c_log_path_sum)
	### create a ganist instance
	ganist = tf_ganist.Ganist(sess, log_path_sum)
	### create a vaeganist instance
	#vae = vae_ganist.VAEGanist(sess, log_path_sum_vae)
	### init variables
	sess.run(tf.global_variables_initializer())
	### save network initially
	ganist.save(log_path_snap+'/model_0_0.h5')
	with open(log_path+'/vars_count_log.txt', 'w+') as fs:
		print >>fs, '>>> g_vars: %d --- d_vars: %d --- e_vars: %d' \
			% (ganist.g_vars_count, ganist.d_vars_count, ganist.e_vars_count)

	'''
	INCEPTION SETUP
	'''
	inception_dir = '/media/evl/Public/Mahyar/Data/models/research/slim'
	ckpt_path = '/media/evl/Public/Mahyar/Data/inception_v3_model/kaggle/inception_v3.ckpt'
	sys.path.insert(0, inception_dir)

	#from inception.slim import slim
	import nets.inception_v3 as inception
	from nets.inception_v3 import inception_v3_arg_scope

	### images should be N*299*299*3 of values (-1,1)
	images_pl = tf.placeholder(tf_ganist.tf_dtype, [None, ganist.data_dim[0], ganist.data_dim[1], 3], name='input_proc_images')
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

	#fid_test = eval_fid(sess, train_imgs[:5000], test_imgs[:5000])
	#print '>>> FID TEST: ', fid_test

	'''
	CLASSIFIER SETUP SECTION
	'''
	### train mnist classifier
	#val_loss, val_acc = train_mnist_net(mnet, train_imgs, train_labs, val_imgs, val_labs)
	#print ">>> validation loss: ", val_loss
	#print ">>> validation accuracy: ", val_acc

	### load mnist classifier
	#mnet.load(class_net_path)

	### test mnist classifier
	#test_loss, test_acc = eval_mnist_net(mnet, test_imgs, test_labs, batch_size=64)
	#print ">>> test loss: ", test_loss
	#print ">>> test accuracy: ", test_acc

	'''
	GAN SETUP SECTION
	'''

	### train ganist
	train_ganist(ganist, all_imgs, all_labs)

	### load ganist **g_num**
	#ganist.load(ganist_path % run_seed)
	### gset draws: run sample_draw before block_draw_top to load learned gset prior
	gset_sample_draw(ganist, 10)
	gset_block_draw_top(ganist, 10, log_path+'/gset_top_samples.png')
	#sys.exit(0)

	'''
	VAE GANIST SETUP SECTION
	'''
	### train the vae part
	#train_vae(vae, all_imgs)

	### load the vae part
	#vae.load_vae(ganist_path % run_seed)

	### train the ganist part
	#train_ganist(vae, all_imgs)

	### load the whole vaeganist
	#vae.load(ganist_path % run_seed)

	'''
	REAL DATASET CREATE OR LOAD AND EVAL
	'''
	### create stack mnist dataset of all_imgs_size*factor
	factor = sample_size // all_imgs_stack.shape[0]
	mod = sample_size % all_imgs_stack.shape[0]
	if mod > 0:
		ims, labs = get_stack_mnist(all_imgs, all_labs, stack_size=mnist_stack_size)
		ims_mod = ims[:mod, ...]
		labs_mod = labs[:mod, ...]
	if factor > 0:
		r_samples = np.zeros((factor,)+all_imgs_stack.shape)
		r_labs = np.zeros((factor,)+all_labs_stack.shape)
		for i in range(factor):
			ims, labs = get_stack_mnist(all_imgs, all_labs, stack_size=mnist_stack_size)
			r_samples[i, ...] = ims[...]
			r_labs[i, ...] = labs[...]
		r_samples = r_samples.reshape((-1,)+all_imgs_stack.shape[1:])
		r_labs = r_labs.flatten()
		if mod > 0:
			r_samples = np.concatenate((r_samples, ims_mod), axis=0)
			r_labs = np.concatenate((r_labs, labs_mod), axis=0)
	else:
		r_samples = ims_mod
		r_labs = labs_mod

	print '>>> r_samples shape: ', r_samples.shape
	print '>>> r_labs shape: ', r_labs.shape
	#with open(log_path+'/stack_mnist_dataset.cpk', 'wb') as fs:
	#	pk.dump([r_samples, r_labs], fs)
	### OR load stack mnist dataset
	#with open(stack_mnist_path, 'rb') as fs:
	#	r_samples, r_labs = pk.load(fs)
	### mode eval true data
	#eval_sample_quality(mnet, test_imgs[:sample_size, ...], log_path+'/sample_quality_test')
	#mode_num, mode_count, mode_vars = mode_analysis(mnet, r_samples,
	#	log_path+'/mode_analysis_true.cpk', labels=r_labs)
	### mode eval real data
	#eval_sample_quality(mnet, r_samples, log_path+'/sample_quality_real')
	#mode_num, mode_count, mode_vars = mode_analysis(mnet, r_samples, 
	#	log_path+'/mode_analysis_real.cpk')

	### OR load mode eval real data
	#with open(stack_mnist_mode_path, 'rb') as fs:
	#	mode_num, mode_count, mode_vars = pk.load(fs)

	#pr = 1.0 * mode_count / np.sum(mode_count)
	#print ">>> real_mode_num: ", mode_num
	#print ">>> real_mode_count_std: ", np.std(mode_count)
	#print ">>> real_mode_var ", np.mean(mode_vars)

	'''
	VAE DATA EVAL
	'''
	'''
	gan_model = vae
	sampler = vae.step_vae
	### sample gen data and draw **mt**
	g_samples = sample_ganist(gan_model, sample_size, sampler=sampler,
		z_im=r_samples[0:sample_size, ...])
	im_block_draw(g_samples, 10, log_path_draw_vae+'/vae_samples.png')
	
	### mode eval gen data
	### >>> dataset sensitive: draw_list
	mode_num, mode_count, mode_vars = mode_analysis(mnet, g_samples, 
		log_path_vae+'/mode_analysis_gen.cpk')#, draw_list=range(1000), draw_name='gen')
	pg = 1.0 * mode_count / np.sum(mode_count)
	print ">>> gen_mode_num: ", mode_num
	print ">>> gen_mode_count_std: ", np.std(mode_count)
	print ">>> gen_mode_var: ", np.mean(mode_vars)

	### KL and JSD computation
	kl_g = np.sum(pg*np.log(1e-6 + pg / (pr+1e-6)))
	kl_p = np.sum(pr*np.log(1e-6 + pr / (pg+1e-6)))
	jsd = (np.sum(pg*np.log(1e-6 + 2 * pg / (pg+pr+1e-6))) + \
	np.sum(pr*np.log(1e-6 + 2 * pr / (pg+pr+1e-6)))) / 2.0
	print ">>> KL(g||p): ", kl_g
	print ">>> KL(p||g): ", kl_p
	print ">>> JSD(g||p): ", jsd
	'''
	'''
	GAN DATA EVAL
	'''
	'''
	gan_model = ganist#vae
	sampler = ganist.step#vae.step
	### sample gen data and draw **mt**
	g_samples = sample_ganist(gan_model, sample_size, sampler=sampler,
		z_im=r_samples[0:sample_size, ...])
	im_block_draw(g_samples, 10, log_path_draw+'/gen_samples.png')
	
	### mode eval gen data
	### >>> dataset sensitive: draw_list
	eval_sample_quality(mnet, g_samples, log_path+'/sample_quality_gen')
	mode_num, mode_count, mode_vars = mode_analysis(mnet, g_samples, 
		log_path+'/mode_analysis_gen.cpk')#, draw_list=range(1000), draw_name='gen')
	pg = 1.0 * mode_count / np.sum(mode_count)
	print ">>> gen_mode_num: ", mode_num
	print ">>> gen_mode_count_std: ", np.std(mode_count)
	print ">>> gen_mode_var: ", np.mean(mode_vars)

	### KL and JSD computation
	kl_g = np.sum(pg*np.log(1e-6 + pg / (pr+1e-6)))
	kl_p = np.sum(pr*np.log(1e-6 + pr / (pg+1e-6)))
	jsd = (np.sum(pg*np.log(1e-6 + 2 * pg / (pg+pr+1e-6))) + \
	np.sum(pr*np.log(1e-6 + 2 * pr / (pg+pr+1e-6)))) / 2.0
	print ">>> KL(g||p): ", kl_g
	print ">>> KL(p||g): ", kl_p
	print ">>> JSD(g||p): ", jsd
	'''
	sess.close()

