import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import matplotlib.cm as cm
import os

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

SMALL_SIZE = 23
MEDIUM_SIZE = 23
BIGGER_SIZE = 23

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

root_path = '/dresden/users/mk1391/evl/'
exp_paths = {
    'true_celeba128': [os.path.join(root_path, f'ganist_lap_logs/41_logs_wganbn_celeba128cc_hpfid_constrad8/run_{i}/fid_levels_r.cpk')
        for i in [0, 1, 2]],
    'true_bedrooms128': [os.path.join(root_path, f'ganist_lap_logs/44_logs_wganbn_bedroom128cc_hpfid_constrad8/run_{i}/fid_levels_r.cpk')
        for i in [0, 1, 2]],

    'wgan_celeba128': [os.path.join(root_path, f'ganist_lap_logs/41_logs_wganbn_celeba128cc_hpfid_constrad8/run_{i}/fid_levels.cpk')
        for i in [0, 1, 2]],
    'wgan_bedrooms128': [os.path.join(root_path, f'ganist_lap_logs/44_logs_wganbn_bedroom128cc_hpfid_constrad8/run_{i}/fid_levels.cpk')
        for i in [0, 1, 2]],
    'wgan_sceleba128': [os.path.join(root_path, f'ganist_lap_logs/45_logs_wganbn_sceleba128cc_hpfid_constrad8/run_{i}/fid_levels.cpk')
        for i in [0, 1, 2]],
    'wgan_sbedrooms128': [os.path.join(root_path, f'ganist_lap_logs/46_logs_wgan_sbedroom128cc_hpfid/run_{i}/fid_levels.cpk')
        for i in [0, 1, 2]],
    'wgan_fsg_sceleba128': [os.path.join(root_path, f'ganist_lap_logs/47_logs_wganbn_gshift_sceleba128cc_hpfid_constrad8/run_{i}/fid_levels.cpk')
        for i in [0, 1, 2]],
    'wgan_fsg_sbedrooms128': [os.path.join(root_path, f'ganist_lap_logs/49_logs_wgan_gshift_sbedroom128cc_hpfid/run_{i}/fid_levels.cpk')
        for i in [0, 1, 2]],
    'wgan_fsg16_celeba128': [os.path.join(root_path, f'ganist_lap_logs/48_logs_wganbn_fsg16_celeba128cc_hpfid_constrad8/run_{i}/fid_levels.cpk')
        for i in [0, 1, 2]],
    'wgan_fsg16_noshift_celeba128': [os.path.join(root_path, f'ganist_lap_logs/logs_wganbn_fsg16_noshift_celeba128cc_hpfid/run_{i}/fid_levels.cpk')
        for i in [0, 1, 2]],
    
    'pggan_celeba128': [os.path.join(root_path, f'pggan_logs/logs_celeba128cc/logs_pggan_celeba128cc_hpfid_constrad8/run_{i}/fid_levels.cpk')
        for i in [0, 1]],
    'pggan_bedrooms128': [os.path.join(root_path, f'pggan_logs/logs_bedroom128cc/logs_pggan_bedroom128cc_hpfid/run_{i}/fid_levels.cpk')
        for i in [0, 1]],
    'pggan_sceleba128': [os.path.join(root_path, f'pggan_logs/logs_celeba128cc_sh/logs_pggan_sceleba128cc_hpfid_constrad8/run_{i}/fid_levels.cpk')
        for i in [0, 1]],
    'pggan_sbedrooms128': [os.path.join(root_path, f'pggan_logs/logs_bedroom128cc_sh/logs_pggan_sbedroom128cc_hpfid/run_{i}/fid_levels.cpk')
        for i in [0, 1]],
    'pggan_fsg_sceleba128': [os.path.join(root_path, f'pggan_logs/logs_celeba128cc_sh/logs_pggan_outsh_sceleba128cc_hpfid/run_{i}/fid_levels.cpk')
        for i in [0, 1]],
    'pggan_fsg_sbedrooms128': [os.path.join(root_path, f'pggan_logs/logs_bedroom128cc_sh/logs_pggan_outsh_sbedroom128cc_hpfid/run_{i}/fid_levels.cpk')
        for i in [0, 1]],
    'pggan_fsg16_celeba128': [os.path.join(root_path, f'pggan_logs/logs_celeba128cc/logs_pggan_fsg16_celeba128cc_hpfid/run_{i}/fid_levels.cpk')
        for i in [0, 1]],
    'pggan_fsg16_noshift_celeba128': [os.path.join(root_path, f'pggan_logs/logs_celeba128cc/logs_pggan_fsg16_noshift_celeba128cc_hpfid/run_{i}/fid_levels.cpk')
        for i in [0, 1]],

    'stylegan2_celeba128':  [os.path.join(root_path, f'stylegan2_logs/logs_celeba128cc/logs_stylegan2_small_celeba128cc_hpfid_valtrue/run_{i}/fid_levels.cpk')
        for i in [0, 1]],
    'stylegan2_bedrooms128': [os.path.join(root_path, f'stylegan2_logs/logs_bedroom128cc/logs_stylegan2_small_bedroom128cc_hpfid_valtrue/run_{i}/fid_levels.cpk')
        for i in [0, 1]],
    'stylegan2_sceleba128': [os.path.join(root_path, f'stylegan2_logs/logs_sceleba128cc/logs_stylegan2_small_sceleba128cc_hpfid_valtrue/run_{i}/fid_levels.cpk')
        for i in [0, 1]],
    'stylegan2_sbedrooms128': [os.path.join(root_path, f'stylegan2_logs/logs_sbedroom128cc/logs_stylegan2_small_sbedroom128cc_hpfid_valtrue/run_{i}/fid_levels.cpk')
        for i in [0, 1]],
    'stylegan2_fsg_sceleba128': [os.path.join(root_path, f'stylegan2_logs/logs_sceleba128cc/logs_stylegan2_small_outsh_sceleba128cc_hpfid/run_{i}/fid_levels.cpk')
        for i in [0, 1]],
    'stylegan2_fsg_sbedrooms128': [os.path.join(root_path, f'stylegan2_logs/logs_sbedroom128cc/logs_stylegan2_small_outsh_sbedroom128cc_hpfid/run_{i}/fid_levels.cpk')
        for i in [0, 1]],
    'stylegan2_fsg16_celeba128': [os.path.join(root_path, f'stylegan2_logs/logs_celeba128cc/logs_stylegan2_small_fsg_finalstylemix_celeba128cc_hpfid/run_{i}/fid_levels.cpk')
        for i in [0, 1]],
    'stylegan2_fsg16_noshift_celeba128': [os.path.join(root_path, f'stylegan2_logs/logs_celeba128cc/logs_stylegan2_small_fsg_noshift_celeba128cc_hpfid/run_{i}/fid_levels.cpk')
        for i in [1, 2]],
    
    'lowband_noisy_snr5_celeba128': [os.path.join(root_path, f'ganist_lap_logs/logs_true_fid_levels_noisy_celeba128cc_f8/fid_levels_snr5_low0.00_high0.12.cpk')],
    'lowband_noisy_snr10_celeba128': [os.path.join(root_path, f'ganist_lap_logs/logs_true_fid_levels_noisy_celeba128cc_f8/fid_levels_snr10_low0.00_high0.12.cpk')],
    'lowband_noisy_snr20_celeba128': [os.path.join(root_path, f'ganist_lap_logs/logs_true_fid_levels_noisy_celeba128cc_f8/fid_levels_snr20_low0.00_high0.12.cpk')],

    'highband_noisy_snr5_celeba128': [os.path.join(root_path, f'ganist_lap_logs/logs_true_fid_levels_noisy_celeba128cc_f8/fid_levels_snr5_low0.38_high1.00.cpk')],
    'highband_noisy_snr10_celeba128': [os.path.join(root_path, f'ganist_lap_logs/logs_true_fid_levels_noisy_celeba128cc_f8/fid_levels_snr10_low0.38_high1.00.cpk')],
    'highband_noisy_snr20_celeba128': [os.path.join(root_path, f'ganist_lap_logs/logs_true_fid_levels_noisy_celeba128cc_f8/fid_levels_snr20_low0.38_high1.00.cpk')],

    'fullband_noisy_snr5_celeba128': [os.path.join(root_path, f'ganist_lap_logs/logs_true_fid_levels_noisy_celeba128cc_f8/fid_levels_snr5_low0.00_high1.00.cpk')],
    'fullband_noisy_snr10_celeba128': [os.path.join(root_path, f'ganist_lap_logs/logs_true_fid_levels_noisy_celeba128cc_f8/fid_levels_snr10_low0.00_high1.00.cpk')],
    'fullband_noisy_snr20_celeba128': [os.path.join(root_path, f'ganist_lap_logs/logs_true_fid_levels_noisy_celeba128cc_f8/fid_levels_snr20_low0.00_high1.00.cpk')]    
}

plot_configs = {
    'fids_lowband_noisy_celeba128': {
        'title': None,
        'names': ['True', 'SNR=20', 'SNR=10', 'SNR=5'],
        'paths': [exp_paths['true_celeba128'], exp_paths['lowband_noisy_snr20_celeba128'], exp_paths['lowband_noisy_snr10_celeba128'], exp_paths['lowband_noisy_snr5_celeba128']],
        'colors': [cm.get_cmap('tab10')(0.), cm.get_cmap('plasma')(1./5), cm.get_cmap('plasma')(2./5), cm.get_cmap('plasma')(3./5)],
        'markers': [None, 'o', 'v', 's'],
        'xlabel': 'High-pass Cut-off Frequency',
        'ylabel': 'FID'
    },

    'fids_highband_noisy_celeba128': {
        'title': None,
        'names': ['True', 'SNR=20', 'SNR=10', 'SNR=5'],
        'paths': [exp_paths['true_celeba128'], exp_paths['highband_noisy_snr20_celeba128'], exp_paths['highband_noisy_snr10_celeba128'], exp_paths['highband_noisy_snr5_celeba128']],
        'colors': [cm.get_cmap('tab10')(0.), cm.get_cmap('plasma')(1./5), cm.get_cmap('plasma')(2./5), cm.get_cmap('plasma')(3./5)],
        'markers': [None, 'o', 'v', 's'],
        'xlabel': 'High-pass Cut-off Frequency',
        'ylabel': 'FID'
    },

    'fids_fullband_noisy_celeba128': {
        'title': None,
        'names': ['True', 'SNR=20', 'SNR=10', 'SNR=5'],
        'paths': [exp_paths['true_celeba128'], exp_paths['fullband_noisy_snr20_celeba128'], exp_paths['fullband_noisy_snr10_celeba128'], exp_paths['fullband_noisy_snr5_celeba128']],
        'colors': [cm.get_cmap('tab10')(0.), cm.get_cmap('plasma')(1./5), cm.get_cmap('plasma')(2./5), cm.get_cmap('plasma')(3./5)],
        'markers': [None, 'o', 'v', 's'],
        'xlabel': 'High-pass Cut-off Frequency',
        'ylabel': 'FID'
    },

    'fids_wgan_celeba128': {
        'title': None,
        'names': ['True', 'WGAN-GP'],
        'paths': [exp_paths['true_celeba128'], exp_paths['wgan_celeba128']],
        'colors': [cm.get_cmap('tab10')(0), cm.get_cmap('plasma')(1./5)],
        'markers': [None, 'o'],
        'xlabel': 'High-pass Cut-off Frequency',
        'ylabel': 'FID'
    },

    'fids_pggan_celeba128': {
        'title': None,
        'names': ['True', 'PG-GAN'],
        'paths': [exp_paths['true_celeba128'], exp_paths['pggan_celeba128']],
        'colors': [cm.get_cmap('tab10')(0), cm.get_cmap('plasma')(1./5)],
        'markers': [None, 'o'],
        'xlabel': 'High-pass Cut-off Frequency',
        'ylabel': 'FID'
    },

    'fids_stylegan2_celeba128': {
        'title': None,
        'names': ['True', 'StyleGAN2'],
        'paths': [exp_paths['true_celeba128'], exp_paths['stylegan2_celeba128']],
        'colors': [cm.get_cmap('tab10')(0), cm.get_cmap('plasma')(1./5)],
        'markers': [None, 'o'],
        'xlabel': 'High-pass Cut-off Frequency',
        'ylabel': 'FID'
    },

    'fids_wgan_bedrooms128': {
        'title': None,
        'names': ['True', 'WGAN-GP'],
        'paths': [exp_paths['true_bedrooms128'], exp_paths['wgan_bedrooms128']],
        'colors': [cm.get_cmap('tab10')(0), cm.get_cmap('plasma')(1./5)],
        'markers': [None, 'o'],
        'xlabel': 'High-pass Cut-off Frequency',
        'ylabel': 'FID'
    },

    'fids_pggan_bedrooms128': {
        'title': None,
        'names': ['True', 'PG-GAN'],
        'paths': [exp_paths['true_bedrooms128'], exp_paths['pggan_bedrooms128']],
        'colors': [cm.get_cmap('tab10')(0), cm.get_cmap('plasma')(1./5)],
        'markers': [None, 'o'],
        'xlabel': 'High-pass Cut-off Frequency',
        'ylabel': 'FID'
    },

    'fids_stylegan2_bedrooms128': {
        'title': None,
        'names': ['True', 'StyleGAN2'],
        'paths': [exp_paths['true_bedrooms128'], exp_paths['stylegan2_bedrooms128']],
        'colors': [cm.get_cmap('tab10')(0), cm.get_cmap('plasma')(1./5)],
        'markers': [None, 'o'],
        'xlabel': 'High-pass Cut-off Frequency',
        'ylabel': 'FID'
    },

    'fids_wgan_fsg16_celeba128': {
        'title': None,
        'names': ['True', 'WGAN-GP', 'FSG', 'FSG-noshift'],
        'paths': [exp_paths['true_celeba128'], exp_paths['wgan_celeba128'], exp_paths['wgan_fsg16_celeba128'], exp_paths['wgan_fsg16_noshift_celeba128']],
        'colors': [cm.get_cmap('tab10')(0), cm.get_cmap('plasma')(1./5), cm.get_cmap('plasma')(2./5), cm.get_cmap('plasma')(3./5)],
        'markers': [None, 'o', 'v', 's'],
        'xlabel': 'High-pass Cut-off Frequency',
        'ylabel': 'FID'
    },

    'fids_pggan_fsg16_celeba128': {
        'title': None,
        'names': ['True', 'PG-GAN', 'FSG', 'FSG-noshift'],
        'paths': [exp_paths['true_celeba128'], exp_paths['pggan_celeba128'], exp_paths['pggan_fsg16_celeba128'], exp_paths['pggan_fsg16_noshift_celeba128']],
        'colors': [cm.get_cmap('tab10')(0), cm.get_cmap('plasma')(1./5), cm.get_cmap('plasma')(2./5), cm.get_cmap('plasma')(3./5)],
        'markers': [None, 'o', 'v', 's'],
        'xlabel': 'High-pass Cut-off Frequency',
        'ylabel': 'FID'
    },

    'fids_stylegan2_fsg16_celeba128': {
        'title': None,
        'names': ['True', 'StyleGAN2', 'FSG', 'FSG-noshift'],
        'paths': [exp_paths['true_celeba128'], exp_paths['stylegan2_celeba128'], exp_paths['stylegan2_fsg16_celeba128'], exp_paths['stylegan2_fsg16_noshift_celeba128']],
        'colors': [cm.get_cmap('tab10')(0), cm.get_cmap('plasma')(1./5), cm.get_cmap('plasma')(2./5), cm.get_cmap('plasma')(3./5)],
        'markers': [None, 'o', 'v', 's'],
        'xlabel': 'High-pass Cut-off Frequency',
        'ylabel': 'FID'
    }
}

def plot_fid_levels(ax, paths, name, color, marker):
    ### Read the fid_levels
    print(f'\n>>> Reading FID Levels from paths: {paths}')
    fids = list()
    for p in paths:
        with open(p, 'rb') as fs:
            try:
                filter_levels, fid_list = pk.load(fs)
            except:
                filter_levels, fid_list = pk.load(fs, encoding='latin1')
            fids.append(fid_list)
    
    fid_mat = np.array(fids)**2
    fid_mean = np.mean(fid_mat, axis=0)
    fid_std = np.std(fid_mat, axis=0)
    
    ### Plot fid means with std
    filter_levels = np.array(filter_levels)
    ax.plot(fid_mean, color=color, marker=marker, label=name)
    ax.plot(fid_mean+fid_std, linestyle='--', linewidth=0.5, color=color)
    ax.plot(fid_mean-fid_std, linestyle='--', linewidth=0.5, color=color)
    ax.set_xticks(range(len(filter_levels)))
    cutoffs = [0.] + [1./(np.pi*2*s) for s in filter_levels if s > 0]
    ax.set_xticklabels([f'{cutoff*100:.0f}' for cutoff in cutoffs])
    
    print(filter_levels)
    print(f'\n >>> FID Mean={fid_mean[0]} and std={fid_std[0]}')

    return fid_mat

def plot_fids(configs, save_dir):
    for cname, config in configs.items():
        print(f'\n{"-"*40}\n>>> Plotting config: {cname}')
        ### Setup plot
        fig = plt.figure(0, figsize=(8,6), constrained_layout=True)
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        ax.grid(True, which='both', linestyle='dotted')
        ax.set_title(config['title'])
        ax.set_xlabel(config['xlabel']+r' ($\times 0.01$)')
        ax.set_ylabel(config['ylabel'])

        ### Plot and save
        for i, name in enumerate(config['names']):
            plot_fid_levels(ax, config['paths'][i], name, config['colors'][i], config['markers'][i])
        ax.legend(loc=0)
        fig.savefig(os.path.join(save_dir, cname+'.pdf'))

if __name__ == '__main__':
    os.makedirs('logs_fids', exist_ok=True)
    plot_fids(plot_configs, save_dir='logs_fids')