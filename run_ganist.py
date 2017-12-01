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
import tf_baby_gan
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from progressbar import ETA, Bar, Percentage, ProgressBar
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm
import matplotlib.tri as mtri
from sklearn.neighbors.kde import KernelDensity
import argparse
print matplotlib.get_backend()

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-l', '--log-path', dest='log_path', required=True, help='log directory to store logs.')
args = arg_parser.parse_args()
log_path = args.log_path
log_path_png = log_path+'/fields'
log_path_snap = log_path+'/snapshots'
log_path_manifold = log_path+'/manifolds'
os.system('mkdir -p '+log_path_png)
os.system('mkdir -p '+log_path_snap)
os.system('mkdir -p '+log_path_manifold)

'''
Training Baby GAN
'''
def train_baby_gan(baby, centers, stds, ratios=None):
	### dataset definition
	data_dim = len(centers[0])
	train_size = 51200

	### baby gan training configs
	epochs = 100
	d_updates = 1
	g_updates = 1
	batch_size = 512

	### logs initi
	g_logs = list()
	d_r_logs = list()
	d_g_logs = list()
	eval_logs = list()

	### training inits
	d_itr = 0
	g_itr = 0
	itr_total = 0
	g_max_itr = 2e4
	widgets = ["baby_gan", Percentage(), Bar(), ETA()]
	pbar = ProgressBar(maxval=g_max_itr, widgets=widgets)
	pbar.start()
	#train_dataset, train_gt = \
	#	generate_normal_data(train_size, centers, stds, ratios)

	while g_itr < g_max_itr:
		#np.random.shuffle(train_dataset)
		train_dataset, train_gt = \
			generate_circle_data(train_size)
		#	generate_normal_data(train_size, centers, stds, ratios)
		
		for batch_start in range(0, train_size, batch_size):
			if g_itr >= g_max_itr:
				break
			pbar.update(g_itr)
			batch_end = batch_start + batch_size
			batch_data = train_dataset[batch_start:batch_end, 0] if data_dim == 1 else train_dataset[batch_start:batch_end, :]
			batch_data = batch_data.reshape((batch_data.shape[0], data_dim))
			### discriminator update
			logs, batch_g_data = baby.step(batch_data, batch_size, gen_update=False)
			if data_dim == 1:
				g_data = baby.step(None, field_sample_size, gen_only=True)
				d_data = train_dataset[0:field_sample_size, 0]
				d_data = d_data.reshape((d_data.shape[0], data_dim))
			else:
				g_data = baby.step(None, field_sample_size, gen_only=True)
				d_data = train_dataset[0:field_sample_size, :]
				d_data = d_data.reshape((d_data.shape[0], data_dim))
			### logging dis results
			g_logs.append(logs[0])
			d_r_logs.append(logs[1])
			d_g_logs.append(logs[2])

			### calculate and plot field of decision for dis update
			field_params = None
			if d_draw > 0 and d_itr % d_draw == 0:
				if data_dim == 1:
					field_params = baby_gan_field_1d(baby, -fov, fov, batch_size*10)
					plot_field_1d(field_params, (d_data, batch_data), (g_data, batch_g_data), 0,
						log_path_png+'/field_%06d.png' % itr_total, 'DIS_%d_%d_%d' % (d_itr%d_updates, g_itr, itr_total))    
				else:
					field_params = baby_gan_field_2d(baby, -fov, fov, -fov, fov, batch_size*10)
					plot_field_2d(field_params, fov, (d_data, batch_data), (g_data, batch_g_data), 0,
						log_path_png+'/field_%06d.png' % itr_total, 'DIS_%d_%d_%d' % (d_itr%d_updates, g_itr, itr_total))
			d_itr += 1
			itr_total += 1
			
			### generator updates: g_updates times for each d_updates of discriminator
			if d_itr % d_updates == 0:
				for gn in range(g_updates):
					### evaluate energy distance between real and gen distributions
					e_dist, e_norm = eval_baby_gan(baby, centers, stds, ratios)
					e_dist = 0 if e_dist < 0 else np.sqrt(e_dist)
					eval_logs.append([e_dist, e_dist/np.sqrt(2.0*e_norm)])

					### generator update
					logs, batch_g_data = baby.step(batch_data, batch_size, gen_update=True)
					g_data = baby.step(None, field_sample_size, gen_only=True)
					g_logs.append(logs[0])
					d_r_logs.append(logs[1])
					d_g_logs.append(logs[2])
					if g_draw > 0 and g_itr % g_draw == 0:
						if data_dim == 1:
							if field_params is None:
								field_params = baby_gan_field_1d(baby, -fov, fov, batch_size*10)
							plot_field_1d(field_params, (d_data, batch_data), (g_data, batch_g_data), 0,
								log_path_png+'/field_%06d.png' % itr_total, 'GEN_%d_%d_%d' % (gn, g_itr, itr_total))
						else:
							if field_params is None:
								field_params = baby_gan_field_2d(baby, -fov, fov, -fov, fov, batch_size*10)
							plot_field_2d(field_params, fov, (d_data, batch_data), (g_data, batch_g_data), 0,
								log_path_png+'/field_%06d.png' % itr_total, 'GEN_%d_%d_%d' % (gn, g_itr, itr_total))
					### draw manifold of generator data
					if g_manifold > 0 and g_itr % g_manifold == 0:
						plot_manifold(baby, 200, 0, log_path_manifold+'/manifold_%06d.png' % itr_total, 'GEN_%d_%d_%d' % (gn, g_itr, itr_total))
					g_itr += 1
					itr_total += 1
					if g_itr >= g_max_itr:
						break
				#_, dis_confs, trace = baby.gen_consolidate(count=50)
				#print '>>> CONFS: ', dis_confs
				#print '>>> TRACE: ', trace
				#baby.reset_network('d_')
	baby.save(log_path_snap+'/model_%d_%d.h5' % (g_itr, itr_total))

	### plot baby gan progress logs
	g_logs_mat = np.array(g_logs)
	d_r_logs_mat = np.array(d_r_logs)
	d_g_logs_mat = np.array(d_g_logs)
	eval_logs_mat = np.array(eval_logs)
	g_logs_names = ['g_loss', 'g_logit_diff', 'g_out_diff', 'g_param_diff']
	d_r_logs_names = ['d_loss', 'd_param_diff', 'd_r_loss', 'r_logit_data', 'd_r_logit_diff', 'd_r_param_diff']
	d_g_logs_names = ['d_g_loss', 'g_logit_data', 'd_g_logit_diff', 'd_g_param_diff']
	eval_logs_names = ['energy_distance', 'energy_distance_norm']

	plot_time_mat(g_logs_mat, g_logs_names, 1, log_path)
	plot_time_mat(d_r_logs_mat, d_r_logs_names, 1, log_path)
	plot_time_mat(d_g_logs_mat, d_g_logs_names, 1, log_path)
	plot_time_mat(eval_logs_mat, eval_logs_names, 1, log_path)

def eval_baby_gan(baby, centers, stds, ratios=None):
	### dataset definition
	data_dim = len(centers[0])
	sample_size = 10000
	r_samples, gt = \
		generate_circle_data(sample_size)
		#generate_normal_data(sample_size, centers, stds, ratios)

	g_samples = baby.step(None, sample_size, gen_only=True)
	if data_dim > 1:
		rr_score = np.mean(np.sqrt(np.sum(np.square(r_samples[0:sample_size//2, ...] - r_samples[sample_size//2:, ...]), axis=1)))
		gg_score = np.mean(np.sqrt(np.sum(np.square(g_samples[0:sample_size//2, ...] - g_samples[sample_size//2:, ...]), axis=1)))
		rg_score = np.mean(np.sqrt(np.sum(np.square(r_samples[0:sample_size//2, ...] - g_samples[0:sample_size//2, ...]), axis=1)))
	else:
		rr_score = np.mean(np.abs(r_samples[0:sample_size//2] - r_samples[sample_size//2:]))
		gg_score = np.mean(np.abs(g_samples[0:sample_size//2] - g_samples[sample_size//2:]))
		rg_score = np.mean(np.abs(r_samples[0:sample_size//2] - g_samples[0:sample_size//2]))
	return 2*rg_score - rr_score - gg_score, rg_score


if __name__ == '__main__':
	centers = [[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
	stds = [[0.02, 0.02], [0.02, 0.02], [0.02, 0.02], [0.02, 0.02]]
	#centers = [[-0.5, 0.0], [0.5, 0.0], [0.0, 0.5], [0.0, -0.5]]
	#stds = [[0.01, 0.01], [0.01, 0.01], [0.01, 0.01], [0.01, 0.01]]
	#ratios = [0.2, 0.2, 0.4, 0.2]
	ratios = None
	data_dim = len(centers[0])
	#baby = baby_gan.BabyGAN(data_dim)
	baby = tf_baby_gan.TFBabyGAN(data_dim)

	train_baby_gan(baby, centers, stds, ratios)

	e_dist, e_norm = eval_baby_gan(baby, centers, stds)
	with open(log_path+'/txt_logs.txt', 'w+') as fs:
		e_dist = 0 if e_dist < 0 else np.sqrt(e_dist)
		print >>fs, '>>> energy_distance: %f, energy_coef: %f' % (e_dist, e_dist/np.sqrt(2.0*e_norm))