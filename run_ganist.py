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
import tf_ganist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from progressbar import ETA, Bar, Percentage, ProgressBar
import argparse
print matplotlib.get_backend()
import cPickle as pk
import gzip
from skimage.transform import resize


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-l', '--log-path', dest='log_path', required=True, help='log directory to store logs.')
args = arg_parser.parse_args()
log_path = args.log_path
log_path_snap = log_path+'/snapshots'
log_path_draw = log_path+'/draws'
log_path_sum = log_path+'/sums'
os.system('mkdir -p '+log_path_snap)
os.system('mkdir -p '+log_path_draw)
os.system('mkdir -p '+log_path_sum)

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
	im_data = im_data.reshape([50000, 28, 28, 1])
	### resize
	#im_data_re = np.zeros((im_data.shape[0], im_size, im_size, 1))
	#for i in range(im_data.shape[0]):
	#	im_data_re[i, ...] = resize(im_data[i, ...], (im_size, im_size), preserve_range=True)
	im_data_re = im_data

	### rescale
	im_data_re = im_data_re * 2.0 - 1.0
	return im_data_re

'''
Stacks images randomly on RGB channels, im_data shape must be (N, d, d, 1).
'''
def get_stack_mnist(im_data):
	### copy channels
	im_data_r = np.copy(im_data)
	im_data_g = np.copy(im_data)
	im_data_b = np.copy(im_data)

	### shuffle
	np.random.shuffle(im_data_r)
	np.random.shuffle(im_data_g)
	np.random.shuffle(im_data_b)

	### stack shuffled channels
	im_data_stacked = np.concatenate((im_data_r, im_data_g, im_data_b), axis=3)
	return im_data_stacked

def plot_time_series(name, vals, fignum, save_path, color='b', ytype='linear'):
	plt.figure(fignum, figsize=(8, 6))
	plt.clf()
	plt.plot(vals, color=color)
	plt.grid(True, which='both', linestyle='dotted')
	plt.title(name)
	plt.xlabel('Iterations')
	plt.ylabel('Values')
	if ytype=='log':
		plt.yscale('log')
	plt.savefig(save_path)

def plot_time_mat(mat, mat_names, fignum, save_path, ytype=None):
	for n in range(mat.shape[1]):
		fig_name = mat_names[n]
		if not ytype:
			ytype = 'log' if 'param' in fig_name else 'linear'
		plot_time_series(fig_name, mat[:,n], fignum, save_path+'/'+fig_name+'.png', ytype=ytype)

'''
Draws a draw_size*draw_size block image by randomly selecting from im_data.
Assumes im_data range of (-1,1) and shape (N, d, d, 3).
'''
def im_block_draw(im_data, draw_size, path):
	plt.figure(0)
	plt.clf()
	sample_size = im_data.shape[0]
	im_size = im_data.shape[1]
	im_channel = im_data.shape[3]
	draw_ids = np.random.choice(sample_size, size=draw_size**2, replace=False)
	im_draw = im_data[draw_ids, ...].reshape([draw_size, im_size*draw_size, im_size, im_channel])
	im_draw = np.concatenate([im_draw[i, ...] for i in range(im_draw.shape[0])], axis=1)
	im_draw = (im_draw + 1.0) / 2.0
	plt.imshow(im_draw)
	plt.savefig(path)

'''
Train Ganist
'''
def train_ganist(ganist, im_data):
	### dataset definition
	train_size = im_data.shape[0]

	### baby gan training configs
	max_itr_total = 1e5
	g_max_itr = 2e4
	d_updates = 5
	g_updates = 1
	batch_size = 64
	eval_step = 100

	### logs initi
	g_logs = list()
	d_r_logs = list()
	d_g_logs = list()
	eval_logs = list()

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
		train_dataset = get_stack_mnist(im_data)
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
				### discriminator update
				if d_update_flag is True:
					batch_sum, batch_g_data = ganist.step(batch_data, batch_size, gen_update=False)
					ganist.write_sum(batch_sum, itr_total)
					d_itr += 1
					itr_total += 1
					d_update_flag = False if d_itr % d_updates == 0 else True
					fetch_batch = True
				else:
				### generator updates: g_updates times for each d_updates of discriminator
					batch_sum, batch_g_data = ganist.step(batch_data, batch_size, gen_update=True)
					ganist.write_sum(batch_sum, itr_total)
					g_itr += 1
					itr_total += 1
					d_update_flag = True if g_itr % g_updates == 0 else False
				
				### evaluate energy distance between real and gen distributions
				if itr_total % eval_step == 0:
					e_dist, e_norm = eval_ganist(ganist, train_dataset, log_path_draw+'/%d.png' % itr_total)
					e_dist = 0 if e_dist < 0 else np.sqrt(e_dist)
					eval_logs.append([e_dist, e_dist/np.sqrt(2.0*e_norm)])

				if itr_total >= max_itr_total:
					break

		### save network every epoch
		ganist.save(log_path_snap+'/model_%d_%d.h5' % (g_itr, itr_total))

		### plot ganist evaluation plot every epoch
		eval_logs_mat = np.array(eval_logs)
		eval_logs_names = ['energy_distance', 'energy_distance_norm']
		plot_time_mat(eval_logs_mat, eval_logs_names, 1, log_path)

def eval_ganist(ganisy, im_data, draw_path=None):
	### sample and batch size
	sample_size = 1024
	batch_size = 64
	draw_size = 10
	
	### collect real and gen samples
	r_samples = im_data[0:sample_size, ...].reshape((sample_size, -1))
	g_samples = np.zeros(r_samples.shape)
	for batch_start in range(0, sample_size, batch_size):
		batch_end = batch_start + batch_size
		batch_len = g_samples[batch_start:batch_end, ...].shape[0]
		g_samples[batch_start:batch_end, ...] = \
			ganist.step(None, batch_len, gen_only=True).reshape((batch_len, -1))
	
	### calculate energy distance
	rr_score = np.mean(np.sqrt(np.sum(np.square( \
		r_samples[0:sample_size//2, ...] - r_samples[sample_size//2:, ...]), axis=1)))
	gg_score = np.mean(np.sqrt(np.sum(np.square( \
		g_samples[0:sample_size//2, ...] - g_samples[sample_size//2:, ...]), axis=1)))
	rg_score = np.mean(np.sqrt(np.sum(np.square( \
		r_samples[0:sample_size//2, ...] - g_samples[0:sample_size//2, ...]), axis=1)))

	### draw block image of gen samples
	if draw_path is not None:
		g_samples = g_samples.reshape((-1,) + im_data.shape[1:])
		im_block_draw(g_samples, draw_size, draw_path)

	return 2*rg_score - rr_score - gg_score, rg_score

if __name__ == '__main__':
	### read and process data
	data_path = '/home/mahyar/Downloads/mnist.pkl.gz'
	train_data, val_data, test_data = read_mnist(data_path)
	train_labs = train_data[1]
	train_imgs = im_process(train_data[0])

	### get a ganist instance
	ganist = tf_ganist.Ganist(log_path_sum)

	### draw true images
	train_imgs_stack = get_stack_mnist(train_imgs)
	im_block_draw(train_imgs_stack, 10, log_path_draw+'/true_samples.png')

	### train ganist
	train_ganist(ganist, train_imgs)