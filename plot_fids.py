#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 16:41:15 2019

@author: mahyar
"""

import numpy as np
import matplotlib.pyplot as plt
#import cPickle as pk
import pickle as pk
import matplotlib.cm as matcm
import os

SMALL_SIZE = 14
MEDIUM_SIZE = 14
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


### global colormap set
global_cmap = matcm.get_cmap('tab10')
global_color_locs = np.arange(10) / 10.
global_color_set_alt = global_cmap(global_color_locs)

global_cmap = matcm.get_cmap('gnuplot2')
global_color_locs = np.arange(20) / 20.
global_color_set = global_cmap(global_color_locs)
global_color_set[0] = global_color_set_alt[0]

def adjust_hls(color, amount):
	import matplotlib.colors as mc
	import colorsys
	try:
		c = mc.cnames[color]
	except:
		c = color
	c = colorsys.rgb_to_hls(*mc.to_rgb(c))
	return colorsys.hls_to_rgb(
		max(0, min(1, amount[0] * c[0])), 
		max(0, min(1, amount[1] * c[1])), 
		max(0, min(1, amount[2] * c[2])))

evl_path = '/dresden/users/mk1391/evl/'
log_dir = 'logs_miss_details_iclr/'
fid_paths = [
	#'/media/evl/Public/Mahyar/ganist_lsun_logs/layer_stats/fid_levels/logs_fidlevels_celeba128cc_gauss41/run_%d/fid_levels.cpk',
	#'/media/evl/Public/Mahyar/ganist_lap_logs/4_logs_wganbn_lap3_celeba128cc_fid50_gwrong_realonly/run_%d/fid_levels_r.cpk',
	#evl_path+'ganist_lap_logs/logs_wganbn_cub128bb/run_%d/fid_levels_r.cpk'
	evl_path+'ganist_lap_logs/41_logs_wganbn_celeba128cc_hpfid_constrad8/run_%d/fid_levels_r.cpk',
	evl_path+'ganist_lap_logs/logs_true_fid_levels_noisy_celeba128cc_f8/fid_levels_snr20_low0.00_high1.00.cpk',
	evl_path+'ganist_lap_logs/logs_true_fid_levels_noisy_celeba128cc_f8/fid_levels_snr10_low0.00_high1.00.cpk',
	evl_path+'ganist_lap_logs/logs_true_fid_levels_noisy_celeba128cc_f8/fid_levels_snr5_low0.00_high1.00.cpk'
	#evl_path+'ganist_lap_logs/45_logs_wganbn_sceleba128cc_hpfid_constrad8/run_%d/fid_levels.cpk',
	#evl_path+'ganist_lap_logs/47_logs_wganbn_gshift_sceleba128cc_hpfid_constrad8/run_%d/fid_levels.cpk'
	#evl_path+'ganist_lap_logs/41_logs_wganbn_celeba128cc_hpfid_constrad8/run_%d/fid_levels.cpk',
	#evl_path+'/dresden/users/mk1391/evl/pggan_logs/logs_bedroom128cc_sh/results_gdsmall_sbedroom_0/run_%d/fid_levels.cpk'
	#evl_path+'ganist_lap_logs/45_logs_wganbn_sceleba128cc_hpfid_constrad8/run_%d/fid_levels.cpk'
	#evl_path+'ganist_lap_logs/44_logs_wganbn_bedroom128cc_hpfid_constrad8/run_%d/fid_levels_r.cpk',
	#evl_path+'ganist_lap_logs/44_logs_wganbn_bedroom128cc_hpfid_constrad8/run_%d/fid_levels.cpk'
	#evl_path+'ganist_lap_logs/46_logs_wgan_sbedroom128cc_hpfid/run_%d/fid_levels.cpk',
	#evl_path+'ganist_lap_logs/49_logs_wgan_gshift_sbedroom128cc_hpfid/run_%d/fid_levels.cpk'
	#evl_path+'ganist_lap_logs/43_logs_wganbn_cub128bb_hpfid_constrad8/run_%d/fid_levels_r.cpk',
	#evl_path+'ganist_lap_logs/43_logs_wganbn_cub128bb_hpfid_constrad8/run_%d/fid_levels.cpk'
	#evl_path+'ganist_lap_logs/48_logs_wganbn_fsg16_celeba128cc_hpfid_constrad8/run_%d/fid_levels.cpk',
	#evl_path+'ganist_lap_logs/logs_wganbn_fsg4_celeba128cc_hpfid/run_%d/fid_levels.cpk',
	#evl_path+'ganist_lap_logs/logs_wganbn_fsg16_noshift_celeba128cc_hpfid/run_%d/fid_levels.cpk'
	#evl_path+'ganist_lap_logs/52_logs_wgan_fsg16_branch_celeba128cc/run_%d/fid_levels.cpk'
	#evl_path+'pggan_logs/logs_bedroom128cc/logs_pggan_bedroom128cc_hpfid/run_%d/fid_levels.cpk'
	#evl_path+'pggan_logs/logs_bedroom128cc_sh/logs_pggan_sbedroom128cc_hpfid/run_%d/fid_levels.cpk',
	#evl_path+'pggan_logs/logs_bedroom128cc_sh/logs_pggan_outsh_sbedroom128cc_hpfid/run_%d/fid_levels.cpk'
	#evl_path+'pggan_logs/logs_celeba128cc/logs_pggan_celeba128cc_hpfid_constrad8/run_%d/fid_levels.cpk',
	#evl_path+'pggan_logs/logs_celeba128cc_sh/logs_pggan_sceleba128cc_hpfid_constrad8/run_%d/fid_levels.cpk',
	#evl_path+'pggan_logs/logs_celeba128cc_sh/logs_pggan_outsh_sceleba128cc_hpfid/run_%d/fid_levels.cpk'
	#evl_path+'pggan_logs/logs_celeba128cc/logs_pggan_fsg16_celeba128cc_hpfid/run_%d/fid_levels.cpk',
	#evl_path+'pggan_logs/logs_celeba128cc/logs_pggan_fsg16_noshift_celeba128cc_hpfid/run_%d/fid_levels.cpk'
	#evl_path+'pggan_logs/logs_celeba128cc/logs_pggan_fsg_out_share_celeba128cc_hpfid/run_%d/fid_levels.cpk'
	#evl_path+'stylegan2_logs/logs_celeba128cc/logs_stylegan2_small_celeba128cc_hpfid_valtrue/run_%d/fid_levels.cpk',
	#evl_path+'stylegan2_logs/logs_celeba128cc/logs_stylegan2_small_fsg16_celeba128cc_hpfid/run_%d/fid_levels.cpk',
	#evl_path+'stylegan2_logs/logs_celeba128cc/logs_stylegan2_small_fsg_finalstylemix_celeba128cc_hpfid/run_%d/fid_levels.cpk',
	#evl_path+'stylegan2_logs/logs_celeba128cc/logs_stylegan2_small_fsg_noshift_celeba128cc_hpfid/run_%d/fid_levels.cpk' 
	#evl_path+'stylegan2_logs/logs_celeba128cc/logs_stylegan2_small_fsg_nostylemix_celeba128cc_hpfid/run_%d/fid_levels.cpk'
	#evl_path+'stylegan2_logs/logs_sceleba128cc/logs_stylegan2_small_sceleba128cc_hpfid_valtrue/run_%d/fid_levels.cpk'
	#evl_path+'stylegan2_logs/logs_sceleba128cc/logs_stylegan2_small_outsh_sceleba128cc_hpfid/run_%d/fid_levels.cpk'
	#evl_path+'stylegan2_logs/logs_bedroom128cc/logs_stylegan2_small_bedroom128cc_hpfid_valtrue/run_%d/fid_levels.cpk'
	#evl_path+'stylegan2_logs/logs_sbedroom128cc/logs_stylegan2_small_sbedroom128cc_hpfid_valtrue/run_%d/fid_levels.cpk'
	#evl_path+'stylegan2_logs/logs_sbedroom128cc/logs_stylegan2_small_outsh_sbedroom128cc_hpfid/run_%d/fid_levels.cpk'
	#'/media/evl/Public/Mahyar/ganist_lap_logs/logs_wganbn_bedroom128cc/run_%d/fid_levels_r.cpk',
	#'/media/evl/Public/Mahyar/ganist_lap_logs/logs_wganbn_bedroom128cc/run_%d/fid_levels.cpk'
	#'/media/evl/Public/Mahyar/ganist_lsun_logs/layer_stats/35_logs_lap3_reconst_celeba128cc/run_%d/fid_levels.cpk',
	#'/media/evl/Public/Mahyar/ganist_lsun_logs/layer_stats/33_logs_gansd_celeba128cc/run_%d/fid_levels.cpk',
	#'/media/evl/Public/Mahyar/ganist_lsun_logs/layer_stats/23_logs_gandm_or_celeba128cc/run_%d/fid_levels.cpk',
	#'/media/evl/Public/Mahyar/ganist_lap_logs/5_logs_wganbn_celeba128cc_fid50/run_%d/fid_levels.cpk',
	#'/media/evl/Public/Mahyar/ganist_lap_logs/13_logs_wganbn_celeba128cc_fssetup/run_%d/fid_levels.cpk',
	#'/media/evl/Public/Mahyar/ganist_lap_logs/10_logs_wganbn_conv3_celeba128cc_fid50/run_%d/fid_levels.cpk'
	#'/media/evl/Public/Mahyar/ganist_lap_logs/0_logs_wganbn_lap3_celeba128cc/fid50/run_%d/fid_levels.cpk',
	#'/media/evl/Public/Mahyar/ganist_lap_logs/9_logs_wganbn_lap3_gd344_celeba128cc_fid50/run_%d/fid_levels.cpk',
	#'/media/evl/Public/Mahyar/ganist_lap_logs/11_logs_wganbn_split4_celeba128_fid50/run_%d/fid_levels.cpk'
	#'/media/evl/Public/Mahyar/ganist_lap_logs/14_logs_wganbn_celeba128cc_fssetup_fshift/run_%d/fid_levels.cpk',
	#'/media/evl/Public/Mahyar/ganist_lap_logs/8_logs_wganbn_lap3_gd334_fshift_celeba128cc_fid50/run_%d/fid_levels.cpk'
	#'/media/evl/Public/Mahyar/ganist_lap_logs/7_logs_wganbn_lap3_gd334_celeba128cc_fid50/run_%d/fid_levels.cpk',
	#'/media/evl/Public/Mahyar/ganist_lap_logs/23_logs_wganbn_lap3_gd334_winsinc4_celeba128cc/run_%d/fid_levels.cpk'
	#'/media/evl/Public/Mahyar/ganist_lap_logs/9_logs_wganbn_lap3_gd344_celeba128cc_fid50/run_%d/fid_levels.cpk',
	#'/media/evl/Public/Mahyar/ganist_lap_logs/15_logs_wganbn_lap3_winsinc4_celeba128cc/run_%d/fid_levels.cpk',
	#'/media/evl/Public/Mahyar/ganist_lap_logs/16_logs_wganbn_imsize32_celeba128cc/run_%d/fid_levels.cpk',
	#'/media/evl/Public/Mahyar/ganist_lap_logs/17_logs_wganbn_imsize32_d32d128_celeba128cc/run_%d/fid_levels.cpk',
	#'/media/evl/Public/Mahyar/ganist_lap_logs/22_logs_wganbn_gshift_celeba128cc_fshift/run_%d/fid_levels.cpk'
	#'/media/evl/Public/Mahyar/ganist_lap_logs/24_logs_wganbn_gshift_gnoshift_celeba128cc_fshift/run_%d/fid_levels.cpk'
	#'/media/evl/Public/Mahyar/ganist_lap_logs/25_logs_fsm_wganbn_8g64_d128_celeba128cc/run_%d/fid_levels.cpk',
	#'/media/evl/Public/Mahyar/ganist_lap_logs/28_logs_fsm_wganbn_8g64_g128_d128_dlp32_celeba128cc/run_%d/fid_levels.cpk'
	#'/media/evl/Public/Mahyar/ganist_lap_logs/30_logs_wganbn_gshift_gnoshift_2d128_celeba128cc_fshift/run_%d/fid_levels.cpk'
	#'/media/evl/Public/Mahyar/pggan_logs/logs_celeba128cc/logs_pggan_gdsmall_%d/fid_levels.cpk',
	#'/media/evl/Public/Mahyar/pggan_logs/logs_celeba128cc/logs_pggan_fsg16_%d/fid_levels.cpk'
	#'/media/evl/Public/Mahyar/pggan_logs/logs_celeba128cc_sh/logs_pggan_celeba128cc_sh_%d/fid_levels.cpk'
	#'/media/evl/Public/Mahyar/ganist_lap_logs/31_logs_fsm16_wganbn_8g64_gd128_celeba128cc/run_%d/fid_levels.cpk'
	#'/media/evl/Public/Mahyar/pggan_logs/logs_bedroom128cc/logs_pggan_gdsmall_%d/fid_levels.cpk'
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
	print('>>> paths: {}'.format(paths))
	fids = list()
	for p in paths:
		with open(p, 'rb') as fs:
			try:
				blur_levels, fid_list = pk.load(fs)
			except:
				blur_levels, fid_list = pk.load(fs, encoding='latin1')
			fids.append(fid_list)
	fid_mat = np.array(fids)**2
	fid_mean = np.mean(fid_mat, axis=0)
	fid_std = np.std(fid_mat, axis=0)
	### plot fid means with std
	#blur_levels[0] = 1 ## for the average pooling blurring
	blur_levels = np.array(blur_levels)
	ax.plot(fid_mean, color=pcolor, label=pname)
	ax.plot(fid_mean+fid_std, linestyle='--', linewidth=0.5, color=pcolor)
	ax.plot(fid_mean-fid_std, linestyle='--', linewidth=0.5, color=pcolor)
	ax.set_xticks(range(len(blur_levels)))

	### cut off frequency conversion
	blur_cutoff = [0.] + [1./(np.pi*2*s) for s in blur_levels if s > 0]
	
	### for regular SD labels
	ax.set_xticklabels(map('{:.2f}'.format, blur_cutoff))
	
	### for fractions: in terms of the cutoff radius of first filter which is M/(2*pi)
	#frac_labels = ['0', '1', r'$\frac{7}{8}$', r'$\frac{6}{8}$', r'$\frac{5}{8}$', 
	#	r'$\frac{4}{8}$', r'$\frac{3}{8}$', r'$\frac{2}{8}$', r'$\frac{1}{8}$']
	#ax.set_xticklabels(frac_labels)
	print(blur_levels)
	print(f'mean is {fid_mean[0]} and std is {fid_std[0]}')
	
if __name__ == '__main__':
	### prepare plot
	fig = plt.figure(0, figsize=(8,6))
	ax = fig.add_subplot(1,1,1)
	ax.grid(True, which='both', linestyle='dotted')
	ax.set_xlabel(r'High-pass Cut-off Frequency')
	ax.set_ylabel('FID')
	#ax.set_yscale('log')
	ax.set_title('FID HP Levels: CelebA 128')

	### plot
	pnames = ['True', 'SNR=20', 'SNR=10', 'SNR=5']
	pcolor_ids = [0, -15, -11, -9] ## add 0 for real, pggan 6 and 4, wgan 1 and 5, stylegan2 3 and 7
	pcolors_adjust = [[1, 1, 1], [1, 1.6, 0.8], [1, 1.3, 0.8], [1, 1, 0.8]]
	pcolors = [adjust_hls(global_color_set[i], amount) for i, amount in zip(pcolor_ids, pcolors_adjust)]
	for i, pcolor in enumerate(pcolors):
		p = fid_paths[i]
		plot_fid_levels(ax, p, pnames[i], pcolor)
	
	ax.legend(loc=0)
	log_path = os.path.join(log_dir, 'fids_font_fix/fids_hp_true_celeba128cc_noisy_low0.00_high1.00.pdf')
	#log_path = os.path.join(log_dir, 'fids_font_fix/_fids_hp_stylegan2_vs_fsg16_noshift_celeba128cc.pdf')
	#fig.savefig('/media/evl/Public/Mahyar/ganist_lap_logs/plots/fids50_wganbn_celeba128cc.pdf')
	#fig.savefig('/home/mahyar/miss_details_images/temp/fids50_true_cub128bb.pdf')
	fig.savefig(log_path)




