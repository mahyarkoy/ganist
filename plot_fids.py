#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 16:41:15 2019

@author: mahyar
"""

import numpy as np
import matplotlib.pyplot as plt
import cPickle as pk
import matplotlib.cm as matcm
import os


### global colormap set
global_cmap = matcm.get_cmap('tab10')
global_color_locs = np.arange(10) / 10.
global_color_set = global_cmap(global_color_locs)

fid_paths = [
	#'/media/evl/Public/Mahyar/ganist_lsun_logs/layer_stats/fid_levels/logs_fidlevels_celeba128cc_gauss41/run_%d/fid_levels.cpk',
	'/media/evl/Public/Mahyar/ganist_lap_logs/4_logs_wganbn_lap3_celeba128cc_fid50_gwrong_realonly/run_%d/fid_levels_r.cpk',
	#'/media/evl/Public/Mahyar/ganist_lsun_logs/layer_stats/35_logs_lap3_reconst_celeba128cc/run_%d/fid_levels.cpk',
	#'/media/evl/Public/Mahyar/ganist_lsun_logs/layer_stats/33_logs_gansd_celeba128cc/run_%d/fid_levels.cpk',
	#'/media/evl/Public/Mahyar/ganist_lsun_logs/layer_stats/23_logs_gandm_or_celeba128cc/run_%d/fid_levels.cpk',
	'/media/evl/Public/Mahyar/ganist_lap_logs/5_logs_wganbn_celeba128cc_fid50/run_%d/fid_levels.cpk',
	#'/media/evl/Public/Mahyar/ganist_lap_logs/10_logs_wganbn_conv3_celeba128cc_fid50/run_%d/fid_levels.cpk'
	#'/media/evl/Public/Mahyar/ganist_lap_logs/0_logs_wganbn_lap3_celeba128cc/fid50/run_%d/fid_levels.cpk',
	'/media/evl/Public/Mahyar/ganist_lap_logs/9_logs_wganbn_lap3_gd344_celeba128cc_fid50/run_%d/fid_levels.cpk',
	'/media/evl/Public/Mahyar/ganist_lap_logs/11_logs_wganbn_split4_celeba128_fid50/run_%d/fid_levels.cpk'
	#'/media/evl/Public/Mahyar/ganist_lap_logs/8_logs_wganbn_lap3_gd334_fshift_celeba128cc_fid50/run_%d/fid_levels.cpk'
	#'/media/evl/Public/Mahyar/ganist_lap_logs/9_logs_wganbn_lap3_gd344_celeba128cc_fid50/run_%d/fid_levels.cpk'
	#'/media/evl/Public/Mahyar/ganist_lap_logs/2_logs_wganbn_lap3_celeba128cc_popos/fid50/run_%d/fid_levels.cpk'
	#'/media/evl/Public/Mahyar/ganist_lap_logs/temp/logs_wganbn_lap3_celeba128cc_newshuffle/run_0/fid_levels.cpk'
	#'/media/evl/Public/Mahyar/ganist_lsun_logs/layer_stats/23_logs_gandm_or_celeba128cc/run_%d/fid_levels.cpk',
	#'/media/evl/Public/Mahyar/ganist_lsun_logs/layer_stats/24_logs_gandm_ords4_celeba128cc/run_%d/fid_levels.cpk',
	#'/media/evl/Public/Mahyar/ganist_lsun_logs/layer_stats/temp/logs_gandm_ordsus4_celeba128cc/run_0/fid_levels.cpk'
	#'/media/evl/Public/Mahyar/ganist_lsun_logs/layer_stats/fid_levels/logs_fidlevels_proggan_celeba128cc_gauss41/run_%d/fid_levels.cpk'
	#'/media/evl/Public/Mahyar/ganist_lsun_logs/layer_stats/23_logs_gandm_or_celeba128cc/run_%d/fid_levels.cpk',
	#'/media/evl/Public/Mahyar/ganist_lsun_logs/layer_stats/27_logs_gan_celeba128cc_frz5e4_butconv3/run_%d/fid_levels.cpk',
	#'/media/evl/Public/Mahyar/ganist_lsun_logs/layer_stats/29_logs_gan_celeba128cc_frz5e4_butconv3fco/run_%d/fid_levels.cpk',
	#'/media/evl/Public/Mahyar/ganist_lsun_logs/layer_stats/30_logs_gan_celeba128cc_frz5e4_butconv3_newopt/run_%d/fid_levels.cpk'
	#'/media/evl/Public/Mahyar/ganist_lsun_logs/layer_stats/fid_levels/10_fidlevels_gauss_logs/run_%d/fid_levels.cpk'
	#'/media/evl/Public/Mahyar/ganist_lsun_logs/layer_stats/fid_levels/3_fidlevels_logs/run_%d/fid_levels.cpk',
	#'/media/evl/Public/Mahyar/ganist_lsun_logs/layer_stats/fid_levels/7_fidlevels_logs/run_%d/fid_levels.cpk',
	#'/media/evl/Public/Mahyar/ganist_lsun_logs/layer_stats/fid_levels/8_fidlevels_logs/run_%d/fid_levels.cpk',
	#'/media/evl/Public/Mahyar/ganist_lsun_logs/layer_stats/fid_levels/9_fidlevels_logs/run_%d/fid_levels.cpk'
	]

def plot_fid_levels(ax, pathname, pname, pcolor):
	paths = list()
	### collect existing filnames for this pathname
	for i in range(10):
		try:
			p = pathname % i
		except:
			p = pathname
			paths.append(p)
			break
		if not os.path.exists(p):
			continue
		paths.append(p)
	### read the fid_levels
	print '>>> paths: ', paths
	fids = list()
	for p in paths:
		with open(p, 'rb') as fs:
			blur_levels, fid_list = pk.load(fs)
			fids.append(fid_list)
	fid_mat = np.array(fids)
	fid_mean = np.mean(fid_mat, axis=0)
	fid_std = np.std(fid_mat, axis=0)
	### plot fid means with std
	#blur_levels[0] = 1 ## for the average pooling blurring
	blur_levels = np.array(blur_levels)
	ax.plot(blur_levels, fid_mean, color=pcolor, label=pname)
	ax.plot(blur_levels, fid_mean+fid_std, linestyle='--', linewidth=0.5, color=pcolor)
	ax.plot(blur_levels, fid_mean-fid_std, linestyle='--', linewidth=0.5, color=pcolor)
	ax.set_xticks(blur_levels)
	
if __name__ == '__main__':
	### prepare plot
	fig = plt.figure(0, figsize=(8,6))
	ax = fig.add_subplot(1,1,1)
	ax.grid(True, which='both', linestyle='dotted')
	ax.set_xlabel('Filter Std')
	ax.set_ylabel('FID')
	#ax.set_yscale('log')
	ax.set_title('CelebA 128: WGAN-BN')

	### plot
	pnames = ['real', 'wganbn', 'lap3_gd344', 'split4']
	pcolors = [0, 1, 2, 3] ## add 0 for real
	for i, p in enumerate(fid_paths):
		plot_fid_levels(ax, p, pnames[i], global_color_set[pcolors[i]])
	
	ax.legend(loc=0)
	fig.savefig('/media/evl/Public/Mahyar/ganist_lap_logs/plots/fids50_wganbn_lap3gd344_split4_celeba128cc.pdf')