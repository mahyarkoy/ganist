import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as matcm
import cPickle as pk
import os

### global colormap set
global_cmap = matcm.get_cmap('tab20')
global_color_locs = np.arange(20) / 20.
global_color_set = global_cmap(global_color_locs)

def read_mode_analysis(pathname, sample_quality=False):
	modes_list = list()
	counts_list = list()
	vars_list = list()
	p_list = list()
	paths = list()
	high_conf_list = list()

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

	if sample_quality is True:
		for p in paths:
			with open(p, 'rb') as fs:
				th_set, high_conf = pk.load(fs)
				high_conf_list.append(high_conf)
		return th_set, np.array(high_conf_list)

	for p in paths:
		with open(p, 'rb') as fs:
			mode_num, mode_count, mode_vars = pk.load(fs)
			sample_size = np.sum(mode_count)
			mode_p = 1. * mode_count / sample_size
			modes_list.append(mode_num)
			counts_list.append(mode_count)
			vars_list.append(mode_vars)
			p_list.append(mode_p)

	return np.array(modes_list), np.array(counts_list), np.array(vars_list), np.array(p_list)

'''
pr and pg must have same shape: (N, 1000)
return kl_p, kl_g, and jsd of shape (N,)
'''
def compute_stats(pr, pg):
	kl_p = np.sum(pr*np.log(1e-6 + pr / (pg+1e-6)), axis=1)
	kl_g = np.sum(pg*np.log(1e-6 + pg / (pr+1e-6)), axis=1)
	jsd = (np.sum(pg*np.log(1e-6 + 2. * pg / (pg+pr+1e-6)), axis=1) + \
		np.sum(pr*np.log(1e-6 + 2. * pr / (pg+pr+1e-6)), axis=1)) / 2.
	return kl_p, kl_g, jsd

def plot_quality(ax, inds, vals, names):
	b = 0
	cmap = matcm.get_cmap('tab10')
	c = [0., 0.1, 0.2, 0.3, 0.4]
	c = [a*0.1 for a in range(10)] if len(c) < len(names) else c
	for v, n in zip(vals, names):
		mean_v = np.mean(v, axis=0)
		std_v = np.std(v, axis=0)
		ax.plot(inds, mean_v, c=cmap(c[b]), label=n)
		ax.plot(inds, mean_v+std_v, linestyle='--', linewidth=0.5, c=cmap(c[b]))
		ax.plot(inds, mean_v-std_v, linestyle='--', linewidth=0.5, c=cmap(c[b]))
		b += 1

def plot_analysis_bars(ax, vals, names):
	### plot mean values bar plots
	ind = np.arange(vals[0].shape[1])
	w = 0.2
	b = 0
	cmap = matcm.get_cmap('tab10')
	c = [0., 0.1, 0.2, 0.3, 0.4]
	c = [a*0.1 for a in range(10)] if len(c) < len(names) else c 
	for v, n in zip(vals, names):
		mean_v = np.mean(v, axis=0)
		std_v = np.std(v, axis=0)
		ax.bar(ind+b*w, mean_v, width=w, label=n, color=cmap(c[b]), align='center')
		ax.errorbar(ind+b*w, mean_v, yerr=std_v, fmt='none', ecolor='k', capsize=2, capthick=1., elinewidth=1.)
		b += 1
	ax.set_xticks(ind+(b-1)*w / 2.0)
	ax.set_xticklabels(ind)

def plot_analysis(ax, vals, name, window_size=1):
	k = 1. * np.ones(window_size) / window_size
	mean_vals = np.mean(vals, axis=0)
	std_vals = np.std(vals, axis=0)

	### plot mean values
	sm_mean_vals = mean_vals
	if window_size > 1:
		sm_mean_vals = np.convolve(mean_vals, k, 'same')
		sm_mean_vals[0:window_size/2+1] = None
		sm_mean_vals[-window_size/2:] = None
	cp = ax.plot(sm_mean_vals, label=name)

	### plot mean +- std values
	sm_std_vals = std_vals
	if window_size > 1:
		sm_std_vals = np.convolve(std_vals, k, 'same')
		sm_std_vals[0:window_size/2+1] = None
		sm_std_vals[-window_size/2:] = None
	ax.plot(sm_mean_vals+sm_std_vals, linestyle='--', linewidth=0.5, color=cp[0].get_color())
	ax.plot(np.clip(sm_mean_vals-sm_std_vals, 0., None), linestyle='--', linewidth=0.5, color=cp[0].get_color())

def setup_plot_ax(fignum, x_axis, y_axis, title, yscale='linear', figsize=(10,5)):
	fig = plt.figure(fignum, figsize=figsize)
	ax = fig.add_subplot(1,1,1)
	ax.grid(True, which='both', linestyle='dotted')
	ax.set_xlabel(x_axis)
	ax.set_ylabel(y_axis)
	ax.set_yscale(yscale)
	ax.set_title(title)
	#ax.xaxis.set_ticks(np.arange(0, 1000, 1))
	return ax, fig

if __name__ == '__main__':
	#true_path = '/media/evl/Public/Mahyar/mode_analysis_stack_mnist_350k.cpk'
	#true_path = 'logs_c3_cifar/mode_analysis_true.cpk'
	true_path = '/media/evl/Public/Mahyar/mode_analysis_mnist_70k.cpk'
	#true_path = '/media/evl/Public/Mahyar/ganist_logs/logs_monet_18/run_%d/mode_analysis_real.cpk'
	paths = [#'/media/evl/Public/Mahyar/ganist_logs/logs_monet_14_c8/run_%d/mode_analysis_real.cpk',
				#'/media/evl/Public/Mahyar/mode_analysis_mnist_70k_c8.cpk',
				'/media/evl/Public/Mahyar/ganist_logs/logs_monet_real_mnist/run_%d/mode_analysis_real.cpk',
				#'/media/evl/Public/Mahyar/vae_logs/logs_2/run_%d/vae/mode_analysis_gen.cpk',
				'/media/evl/Public/Mahyar/ganist_logs/logs_monet_128/run_%d/mode_analysis_gen.cpk',
				#'/media/evl/Public/Mahyar/ganist_logs/logs_monet_125/run_%d/mode_analysis_gen.cpk',
				#'/media/evl/Public/Mahyar/ganist_logs/logs_monet_52/run_%d/mode_analysis_gen.cpk',
				'/media/evl/Public/Mahyar/ganist_logs/logs_monet_129/run_%d/mode_analysis_gen.cpk']
				#'logs_c3_cifar/mode_analysis_gen.cpk']
	
	names = ['Real',
				#'sisley_2', 
				#'monet_12', 
				'DMGAN-PL', 
				#'monet_98',
				'GAN-GP']
	
	sq_names = ['/media/evl/Public/Mahyar/ganist_logs/logs_monet_real_mnist/run_%d/sample_quality_real.cpk',
				'/media/evl/Public/Mahyar/ganist_logs/logs_monet_128/run_%d/sample_quality_gen.cpk',
				#'/media/evl/Public/Mahyar/ganist_logs/logs_monet_126/run_%d/sample_quality_gen.cpk',
				#'/media/evl/Public/Mahyar/ganist_logs/logs_monet_126/run_%d/sample_quality_gen.cpk',
				#'/media/evl/Public/Mahyar/ganist_logs/logs_monet_126/run_%d/sample_quality_gen.cpk',
				'/media/evl/Public/Mahyar/ganist_logs/logs_monet_129/run_%d/sample_quality_gen.cpk']

	log_path = '/media/evl/Public/Mahyar/ganist_logs/plots'
	#log_path = 'plots'

	ax_p, fig_p = setup_plot_ax(0, 'Modes', 'Probability', 'Probability over Modes', yscale='log')
	ax_vars, fig_vars = setup_plot_ax(1, 'Modes', 'MSD', 'Average Distance over Modes')
	ax_sq, fig_sq = setup_plot_ax(2, 'Confidence', 'Sample Ratio', 'Sample Quality', figsize=(8,6))
	pr_logs = list()
	vars_logs = list()
	sq_logs = list()

	### true modes plotting
	modes_r, counts_r, vars_r, p_r = read_mode_analysis(true_path)
	#pr_logs.append(p_r)
	#vars_logs.append(vars_r)
	
	#plot_analysis(ax_p, p_r, 'true')
	#plot_analysis(ax_vars, vars_r, 'true')
	
	### gen modes plotting
	for p, n, sqn in zip(paths, names, sq_names):
		modes_g, counts_g, vars_g, p_g = read_mode_analysis(p)
		kl_p, kl_g, jsd = compute_stats(np.mean(p_r, axis=0), p_g)
		pr_logs.append(p_g)
		vars_logs.append(vars_g)
		#plot_analysis(ax_p, p_g, n)
		#plot_analysis(ax_vars, vars_g, n)
		print 'KL(p||g) for %s: %f std %f' % (n, np.mean(kl_p), np.std(kl_p))
		print 'KL(g||p) for %s: %f std %f' % (n, np.mean(kl_g), np.std(kl_g))
		print 'JSD(p||g) for %s: %f std %f' % (n, np.mean(jsd), np.std(jsd))
		print np.mean(modes_g, axis=0)
		print modes_g

		inds, high_conf = read_mode_analysis(sqn, sample_quality = True)
		sq_logs.append(high_conf)

	#for p, v, n in zip(pr_logs, vars_logs, ['true']+names):
	#	plot_analysis(ax_p, p, n)
	#	plot_analysis(ax_vars, v, n)
	#plot_analysis_bars(ax_p, pr_logs, ['True']+names)
	#plot_analysis_bars(ax_vars, vars_logs, ['True']+names)
	plot_analysis_bars(ax_p, pr_logs, names)
	plot_analysis_bars(ax_vars, vars_logs, names)
	plot_quality(ax_sq, inds, sq_logs, names)

	### save figures
	ax_p.legend(loc=0)
	ax_vars.legend(loc=0)
	ax_sq.legend(loc=0)
	#fig_p.savefig(log_path+'/pr_modes_'+'_'.join(names)+'.png', dpi=300)
	#fig_vars.savefig(log_path+'/vars_modes_'+'_'.join(names)+'.png', dpi=300)
	#fig_sq.savefig(log_path+'/sample_quality_'+'_'.join(names)+'.png', dpi=300)
	fig_p.savefig(log_path+'/pr_modes_'+'_'.join(names)+'.pdf')
	fig_vars.savefig(log_path+'/vars_modes_'+'_'.join(names)+'.pdf')
	fig_sq.savefig(log_path+'/sample_quality_'+'_'.join(names)+'.pdf')

	### pvals plot
	### plot rl_pvals **g_num**
	pval_path = '/media/evl/Public/Mahyar/ganist_logs/logs_monet_126_with_pvals_saving/run_%d/rl_pvals.cpk' % 4
	with open(pval_path, 'rb') as fs:
		pvals_mat = pk.load(fs)

	pr_g = np.exp(pvals_mat)
	pr_g = pr_g / (np.sum(pr_g, axis=1).reshape([-1, 1]))
	pvals_mat = pr_g

	fig, ax = plt.subplots(figsize=(8, 6))
	ax.clear()
	#print pvals_mat[:,13]
	itrs_logs = np.arange(pvals_mat.shape[0]) * 1000
	for g in range(pvals_mat.shape[-1]):
		#g = 13
		ax.plot(itrs_logs, pvals_mat[:, g], label='g_%d' % g, c=global_color_set[g])
	#ax.plot(itrs_logs, pvals_mat[:, 13], label='g_%d' % g, c=global_color_set[13])
	ax.grid(True, which='both', linestyle='dotted')
	ax.set_title('RL Policy')
	ax.set_xlabel('Iterations')
	ax.set_ylabel('Probability')
	#ax.legend(loc=0)
	#fig.savefig(log_path+'/rl_policy.png', dpi=300)
	fig.savefig(log_path+'/rl_policy.pdf')
	plt.close(fig)