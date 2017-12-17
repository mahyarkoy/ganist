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
import os
from progressbar import ETA, Bar, Percentage, ProgressBar
import argparse
print matplotlib.get_backend()
import cPickle as pk
import gzip
from skimage.transform import resize
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.utils.graph import graph_shortest_path

np.random.seed(13)
tf.set_random_seed(13)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0, 1" for multiple

import tf_ganist
import mnist_net

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-l', '--log-path', dest='log_path', required=True, help='log directory to store logs.')
args = arg_parser.parse_args()
log_path = args.log_path
c_log_path = log_path+'/classifier'
log_path_snap = log_path+'/snapshots'
c_log_path_snap = c_log_path+'/snapshots'
log_path_draw = log_path+'/draws'
log_path_sum = log_path+'/sums'
c_log_path_sum = c_log_path+'/sums'
os.system('mkdir -p '+log_path_snap)
os.system('mkdir -p '+c_log_path_snap)
os.system('mkdir -p '+log_path_draw)
os.system('mkdir -p '+log_path_sum)
os.system('mkdir -p '+c_log_path_sum)

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
	im_data_re = im_data

	### rescale
	im_data_re = im_data_re * 2.0 - 1.0
	return im_data_re

'''
Stacks images randomly on RGB channels, im_data shape must be (N, d, d, 1).
'''
def get_stack_mnist(im_data, labels=None):
	order = np.arange(im_data.shape[0])
	
	np.random.shuffle(order)
	im_data_r = im_data[order, ...]
	labs_r = labels[order] if labels is not None else None

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

def get_stack_mnist_legacy(im_data):
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

def plot_time_series(name, vals, fignum, save_path, color='b', ytype='linear', itrs=None):
	plt.figure(fignum, figsize=(8, 6))
	plt.clf()
	if itrs is None:
		plt.plot(vals, color=color)	
	else:
		plt.plot(itrs, vals, color=color)
	plt.grid(True, which='both', linestyle='dotted')
	plt.title(name)
	plt.xlabel('Iterations')
	plt.ylabel('Values')
	if ytype=='log':
		plt.yscale('log')
	plt.savefig(save_path)

def plot_time_mat(mat, mat_names, fignum, save_path, ytype=None, itrs=None):
	for n in range(mat.shape[1]):
		fig_name = mat_names[n]
		if not ytype:
			ytype = 'log' if 'param' in fig_name else 'linear'
		plot_time_series(fig_name, mat[:,n], fignum, save_path+'/'+fig_name+'.png', ytype=ytype, itrs=itrs)

'''
Draws a draw_size*draw_size block image by randomly selecting from im_data.
Assumes im_data range of (-1,1) and shape (N, d, d, 3).
'''
def im_block_draw(im_data, draw_size, path, separate_channels=True):
	sample_size = im_data.shape[0]
	im_size = im_data.shape[1]
	im_channel = im_data.shape[3]
	### choses images and puts them into a block shape
	draw_ids = np.random.choice(sample_size, size=draw_size**2, replace=False)
	im_draw = im_data[draw_ids, ...].reshape([draw_size, im_size*draw_size, im_size, im_channel])
	im_draw = np.concatenate([im_draw[i, ...] for i in range(im_draw.shape[0])], axis=1)
	im_draw = (im_draw + 1.0) / 2.0
	### plots
	fig = plt.figure(0)
	fig.clf()
	if not separate_channels:
		fig.imshow(im_draw)
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
def train_ganist(ganist, im_data):
	### dataset definition
	train_size = im_data.shape[0]

	### baby gan training configs
	max_itr_total = 1e6
	g_max_itr = 2e4
	d_updates = 5
	g_updates = 1
	batch_size = 4
	eval_step = 100
	draw_step = 100

	### logs initi
	g_logs = list()
	d_r_logs = list()
	d_g_logs = list()
	eval_logs = list()
	itrs_logs = list()

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
		train_dataset = get_stack_mnist_legacy(im_data)
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
					batch_sum, batch_g_data = ganist.step(batch_data, batch_size=None, gen_update=False)
					ganist.write_sum(batch_sum, itr_total)
					d_itr += 1
					itr_total += 1
					d_update_flag = False if d_itr % d_updates == 0 else True
					fetch_batch = True
				else:
				### generator updates: g_updates times for each d_updates of discriminator
					batch_sum, batch_g_data = ganist.step(batch_data, batch_size=None, gen_update=True)
					ganist.write_sum(batch_sum, itr_total)
					g_itr += 1
					itr_total += 1
					d_update_flag = True if g_itr % g_updates == 0 else False
				
				### evaluate energy distance between real and gen distributions
				if itr_total % eval_step == 0:
					draw_path = log_path_draw+'/gen_sample_%d.png' % itr_total if itr_total % draw_step == 0 else None
					e_dist, e_norm = eval_ganist(ganist, train_dataset, draw_path)
					e_dist = 0 if e_dist < 0 else np.sqrt(e_dist)
					eval_logs.append([e_dist, e_dist/np.sqrt(2.0*e_norm)])
					itrs_logs.append(itr_total)

				if itr_total >= max_itr_total:
					break

		### save network every epoch
		ganist.save(log_path_snap+'/model_%d_%d.h5' % (g_itr, itr_total))

		### plot ganist evaluation plot every epoch
		if len(eval_logs) < 2:
			continue
		eval_logs_mat = np.array(eval_logs)
		eval_logs_names = ['energy_distance', 'energy_distance_norm']
		plot_time_mat(eval_logs_mat, eval_logs_names, 1, log_path, itrs=itrs_logs)

'''
Sample sample_size data points from ganist.
'''
def sample_ganist(ganist, sample_size, batch_size=64):
	g_samples = np.zeros([sample_size] + ganist.data_dim)
	for batch_start in range(0, sample_size, batch_size):
		batch_end = batch_start + batch_size
		batch_len = g_samples[batch_start:batch_end, ...].shape[0]
		g_samples[batch_start:batch_end, ...] = \
			ganist.step(None, batch_len, gen_only=True)
	return g_samples

'''
Returns the energy distance of a trained GANist, and draws block images of GAN samples
'''
def eval_ganist(ganist, im_data, draw_path=None):
	### sample and batch size
	sample_size = 1024
	batch_size = 64
	draw_size = 10
	
	### collect real and gen samples
	r_samples = im_data[0:sample_size, ...].reshape((sample_size, -1))
	g_samples = sample_ganist(ganist, sample_size).reshape((sample_size, -1))
	
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
Images im_data shape is (N, 28, 28, 3)
'''
def eval_modes(mnet, im_data, labels=None, draw_list=None, draw_name='gen'):
	batch_size = 64
	mode_threshold = 0
	data_size = im_data.shape[0]
	im_class_ids = dict((i, list()) for i in range(1000))
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
			### red channel
			batch_data = im_data[batch_start:batch_end, ..., 0][..., np.newaxis]
			preds += 1 * np.argmax(mnet.step(batch_data, pred_only=True), axis=1)
			### green channel
			batch_data = im_data[batch_start:batch_end, ..., 1][..., np.newaxis]
			preds += 10 * np.argmax(mnet.step(batch_data, pred_only=True), axis=1)
			### blue channel
			batch_data = im_data[batch_start:batch_end, ..., 2][..., np.newaxis]
			preds += 100 * np.argmax(mnet.step(batch_data, pred_only=True), axis=1)
			### put each image id into predicted class list
			for i, c in enumerate(preds):
				im_class_ids[c].append(batch_start+i)
	else:
		### put each image id into predicted class list
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
	mode_count = np.zeros(1000) 
	mode_vars = np.zeros(1000)
	widgets = ["Var_modes", Percentage(), Bar(), ETA()]
	pbar = ProgressBar(maxval=1000, widgets=widgets)
	pbar.start()
	for c, l in im_class_ids.items():
		pbar.update(c)
		mode_count[c] = len(l)
		mode_vars[c] = eval_mode_var(im_data[l, ...], 18) if len(l) > 18 else 0.0
	return np.sum(mode_count > mode_threshold), mode_count, mode_vars

'''
Return average pairwise iso-distance of im_data
'''
def eval_mode_var(im_data, n_neighbors, n_jobs=12):
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
	d_var = 2.0 * np.sum(d_tri ** 2) / count
	return d_var

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
				eval_logs.append([eval_loss, eval_acc])
				itrs_logs.append(itr_total)

			if itr_total >= max_itr_total:
				break

		### save network every epoch
		mnet.save(c_log_path_snap+'/model_%d.h5' % itr_total)

		### plot mnet evaluation plot every epoch
		if len(eval_logs) < 2:
			continue
		eval_logs_mat = np.array(eval_logs)
		eval_logs_names = ['eval_loss', 'eval_acc']
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
	### read and process data
	data_path = '/media/evl/Public/Mahyar/Data/mnist.pkl.gz'
	stack_mnist_path = '/media/evl/Public/Mahyar/stack_mnist_350k.cpk'
	stack_mnist_mode_path = '/media/evl/Public/Mahyar/mode_analysis_stack_mnist_350k.cpk'
	mnist_net_path = '/media/evl/Public/Mahyar/Data/mnist_classifier/snapshots/model_100000.h5'
	ganist_path = '/media/evl/Public/Mahyar/ganist_logs/logs_monet_3/snapshots/model_333333_2000000.h5'
	sample_size = 350000

	'''
	DATASET LOADING AND DRAWING
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

	### draw true stacked mnist images
	all_imgs_stack, all_labs_stack = get_stack_mnist(all_imgs, all_labs)
		
	#all_imgs_stack = get_stack_mnist_legacy(train_imgs)
	im_block_draw(all_imgs_stack, 10, log_path_draw+'/true_samples.png')
	
	'''
	TENSORFLOW SETUP
	'''
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
	config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
	sess = tf.Session(config=config)
	### create mnist classifier
	mnet = mnist_net.MnistNet(sess, c_log_path_sum)
	### get a ganist instance
	ganist = tf_ganist.Ganist(sess, log_path_sum)
	### init variables
	sess.run(tf.global_variables_initializer())

	'''
	CLASSIFIER SETUP SECTION
	'''
	### train mnist classifier
	#val_loss, val_acc = train_mnist_net(mnet, train_imgs, train_labs, val_imgs, val_labs)
	#print ">>> validation loss: ", val_loss
	#print ">>> validation accuracy: ", val_acc

	### load mnist classifier
	mnet.load(mnist_net_path)

	### test mnist classifier
	test_loss, test_acc = eval_mnist_net(mnet, test_imgs, test_labs, batch_size=64)
	print ">>> test loss: ", test_loss
	print ">>> test accuracy: ", test_acc

	'''
	GAN SETUP SECTION
	'''

	### train ganist
	#train_ganist(ganist, train_imgs)

	### load ganist
	ganist.load(ganist_path)

	'''
	EVALUATION SECTION
	'''
	### create stack mnist dataset of all_imgs_size*factor
	'''
	factor = 5
	r_samples = np.zeros((factor,)+all_imgs_stack.shape)
	r_labs = np.zeros((factor,)+all_labs_stack.shape)
	for i in range(factor):
		ims, labs = get_stack_mnist(all_imgs, all_labs)
		r_samples[i, ...] = ims[...]
		r_labs[i, ...] = labs[...]
	r_samples = r_samples.reshape((-1,)+all_imgs_stack.shape[1:])
	r_labs = r_labs.flatten()

	print '>>> r_samples shape: ', r_samples.shape
	print '>>> r_labs shape: ', r_labs.shape
	with open(log_path+'/stack_mnist_dataset.cpk', 'wb') as fs:
		pk.dump([r_samples, r_labs], fs)
	### OR load stack mnist dataset
	with open(stack_mnist_path, 'rb') as fs:
		r_samples, r_labs = pk.load(fs)
	### mode eval real data
	mode_num, mode_count, mode_vars = mode_analysis(mnet, r_samples, 
		log_path+'/mode_analysis_real.cpk', r_labs)
	'''
	
	### OR load mode eval real data
	with open(stack_mnist_mode_path, 'rb') as fs:
		mode_num, mode_count, mode_vars = pk.load(fs)

	pr = 1.0 * mode_count / sample_size
	print ">>> real_mode_num: ", mode_num
	print ">>> real_mode_count_std: ", np.std(mode_count)
	print ">>> real_mode_var ", np.mean(mode_vars)

	### sample gen data
	g_samples = sample_ganist(ganist, sample_size)
	im_block_draw(g_samples, 10, log_path_draw+'/gen_samples.png')
	### mode eval gen data
	mode_num, mode_count, mode_vars = mode_analysis(mnet, g_samples, 
		log_path+'/mode_analysis_gen.cpk')#, draw_list=range(1000), draw_name='gen')
	pg = 1.0 * mode_count / sample_size
	print ">>> gen_mode_num: ", mode_num
	print ">>> gen_mode_count: ", np.std(mode_count)
	print ">>> gen_mode_var: ", np.mean(mode_vars)

	### KL and JSD computation
	kl_g = np.sum(pg*np.log(pg / pr))
	kl_p = np.sum(pr*np.log(pr / pg))
	jsd = (np.sum(pg*np.log(2 * pg / (pg+pr))) + np.sum(pr*np.log(2 * pr / (pg+pr)))) / 2.0
	print ">>> KL(g||p): ", kl_g
	print ">>> KL(p||g): ", kl_p
	print ">>> JSD(g||p): ", jsd

	sess.close()

