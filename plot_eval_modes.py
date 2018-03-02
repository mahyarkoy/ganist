import numpy as np
import matplotlib.pyplot as plt
import cPickle as pk
import os

def read_mode_analysis(pathname):
	modes_list = list()
	counts_list = list()
	vars_list = list()
	p_list = list()
	paths = list()
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

def setup_plot_ax(fignum, x_axis, y_axis, title, yscale='linear'):
	fig = plt.figure(fignum)
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
	true_path = '/media/evl/Public/Mahyar/mode_analysis_mnist_70k.cpk'
	#true_path = '/media/evl/Public/Mahyar/ganist_logs/logs_monet_18/run_%d/mode_analysis_real.cpk'
	paths = [#'/media/evl/Public/Mahyar/ganist_logs/logs_monet_14_c8/run_%d/mode_analysis_real.cpk',
				'/media/evl/Public/Mahyar/mode_analysis_mnist_70k_c8.cpk',
				#'/media/evl/Public/Mahyar/vae_logs/logs_2/run_%d/vae/mode_analysis_gen.cpk',
				#'/media/evl/Public/Mahyar/ganist_logs/logs_monet_72/run_%d/mode_analysis_gen.cpk',
				'/media/evl/Public/Mahyar/ganist_logs/logs_monet_73/run_%d/mode_analysis_gen.cpk',
				'/media/evl/Public/Mahyar/ganist_logs/logs_monet_74/run_%d/mode_analysis_gen.cpk',
				'/media/evl/Public/Mahyar/ganist_logs/logs_sisley_2/run_%d/mode_analysis_gen.cpk']
	names = ['real',
				#'monet_72', 
				'monet_73', 
				'monet_74', 
				'sisley_2']
	log_path = '/media/evl/Public/Mahyar/ganist_logs/plots'

	ax_p, fig_p = setup_plot_ax(0, 'Modes', 'Probability', 'Probability over Modes', yscale='log')
	ax_vars, fig_vars = setup_plot_ax(1, 'Modes', 'Variance', 'Average Distance over Modes')

	### real modes plotting
	modes_r, counts_r, vars_r, p_r = read_mode_analysis(true_path)
	plot_analysis(ax_p, p_r, 'true')
	plot_analysis(ax_vars, vars_r, 'true')
	
	### gen modes plotting
	for p, n in zip(paths, names):
		modes_g, counts_g, vars_g, p_g = read_mode_analysis(p)
		kl_p, kl_g, jsd = compute_stats(np.mean(p_r, axis=0), p_g)
		plot_analysis(ax_p, p_g, n)
		plot_analysis(ax_vars, vars_g, n)
		print 'KL(p||g) for %s: %f std %f' % (n, np.mean(kl_p), np.std(kl_p))
		print 'KL(g||p) for %s: %f std %f' % (n, np.mean(kl_g), np.std(kl_g))
		print 'JSD(p||g) for %s: %f std %f' % (n, np.mean(jsd), np.std(jsd))
		print np.mean(modes_g, axis=0)
		print modes_g

	### save figures
	ax_p.legend(loc=0)
	ax_vars.legend(loc=0)
	fig_p.savefig(log_path+'/pr_modes_'+'_'.join(names)+'.png', dpi=300)
	fig_vars.savefig(log_path+'/vars_modes_'+'_'.join(names)+'.png', dpi=300)