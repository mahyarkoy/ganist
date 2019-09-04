import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import os
from run_ganist import readim_path_from_dir, readim_from_path, pyramid_draw, TFutil
from tf_ganist import tf_upsample, tf_downsample, tf_binomial_blur, tf_make_lap_pyramid, tf_reconst_lap_pyramid, tf_freq_shift, make_winsinc_blackman
import tf_ganist as tfg
from fft_test import apply_fft_images
run_seed = 0
np.random.seed(run_seed)
tf.set_random_seed(run_seed)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0, 1" for multiple

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
sess = tf.Session(config=config)
tfutil = TFutil(sess, None, None)

### read celeba 128
im_dir = '/media/evl/Public/Mahyar/Data/celeba/img_align_celeba/'
im_size = 128
dataset_size = 10
im_paths = readim_path_from_dir(im_dir)
np.random.shuffle(im_paths)
im_data = readim_from_path(im_paths[:dataset_size], im_size, center_crop=(121, 89))

### 1d filter test
impulse = np.zeros(128)
impulse[128//2] = 1.
kernel = np.array([1., 4., 6., 4., 1.])
kernel = kernel / np.sum(kernel)

### windowed sinc (Blackman)
fc = 1. / 4
ksize = 40
x = np.arange(ksize+1)
x[ksize//2] = 0
kernel = np.sin(2.* np.pi * fc * (x - ksize//2)) / (x - ksize//2)
kernel = kernel * (0.42 - 0.5*np.cos(2. * np.pi * x / ksize) + 0.08*np.cos(4. * np.pi * x) / ksize)
kernel[ksize//2] = 2. * np.pi * fc
kernel = kernel / np.sum(kernel)
kernel = np.convolve(kernel, kernel, 'full')

output = np.convolve(impulse, kernel, 'same')
imf = np.fft.fftn(output)
imf_s = np.fft.fftshift(imf)
imf_proc = np.abs(imf_s)

fig = plt.figure(0, figsize=(8,8))
ax = fig.add_subplot(2,1,1)
pa = ax.plot(output)
ax.set_title('Filter')
ax.grid(True, which='both', linestyle='dotted')

ax = fig.add_subplot(2,1,2)
pa = ax.plot(imf_proc)
ax.set_title('Fast Fourier Transform')
#ax.set_ylim(0., 1.)
ax.grid(True, which='both', linestyle='dotted')
fig.savefig('logs/1d_fft.jpg', dpi=300)
plt.show()

### make sampling ops
#im_ds = tf_downsample(im_data)
#im_us = tf_upsample(im_data)
#im_blur = tf_binomial_blur(im_data, kernel=kernel)
kernel_us = 2. * kernel
im_blur = tf_binomial_blur(tf_upsample(
	tf_downsample(tf_binomial_blur(im_data, kernel=kernel))), kernel=kernel_us)
#pyramid = tf_make_lap_pyramid(im_data, freq_shift=True)
#reconst = tf_reconst_lap_pyramid(pyramid, freq_shift=True)
im_diff = im_data - im_blur
fc = 1. / 4
im_shift, im_shifti = tf_freq_shift(im_diff, fc=fc)
kernel_sh = make_winsinc_blackman(fc=1./3)
im_shift_blur = tf_binomial_blur(im_shift, kernel=kernel_sh)
im_shifti_blur = tf_binomial_blur(im_shifti, kernel=kernel_sh)
im_diff_blur = tf_binomial_blur(im_diff, kernel=kernel_sh)
im_shift_rec = tf_freq_shift(im_shift, fc=fc)[0] + tf_freq_shift(im_shifti, fc=fc)[1]
im_shift_blur_rec = tf_freq_shift(im_shift_blur, fc=fc)[0] + \
	tf_freq_shift(im_shifti_blur, fc=fc)[1]

### apply sampling
#im_dso, im_uso, im_bluro, pyramid, reconst = sess.run([im_ds, im_us, im_blur, pyramid, reconst])
#im_diff = im_data - reconst
#pyramid_full = pyramid + [reconst]

#im_blur, im_diff, im_shift, im_diff_blur, im_shift_blur, im_shift_rec, im_shift_blur_rec = \
#	sess.run([im_blur, im_diff, im_shift, im_diff_blur, im_shift_blur, 
#		im_shift_rec, im_shift_blur_rec])

### apply fft
#im_diff_fft, im_diff_grey = apply_fft_images(im_diff, reshape=True)
#im_shift_fft, im_shift_grey = apply_fft_images(im_shift, reshape=True)
#im_diffb_fft, im_diffb_grey = apply_fft_images(im_diff_blur, reshape=True)
#im_shiftb_fft, im_shiftb_grey = apply_fft_images(im_shift_blur, reshape=True)

### draw image fft pyramid
#norm = 5.
#pyramid_draw([im_data, im_blur, im_diff_grey, im_diffb_grey, 
#	im_shift_grey, im_shiftb_grey, im_shift_rec, im_shift_blur_rec, im_shift_blur_rec + im_blur,
#	np.log(im_diff_fft)/norm-0.5, np.log(im_diffb_fft)/norm-0.5, 
#	np.log(im_shift_fft)/norm-0.5, np.log(im_shiftb_fft)/norm-0.5], 
#	path='logs/fft_pyramid_winsinc.png', im_shape=(10, 128, 128, 3))

### split setup
split = tfg.tf_split(im_data)
im_rec = tfg.tf_reconst_split(split)
split, im_rec = sess.run([split, im_rec])

im_fft, _ = apply_fft_images(im_data, reshape=True)
im_rec_fft, _ = apply_fft_images(im_rec, reshape=True)
split_l0_fft, _ = apply_fft_images(split[0], reshape=True)
split_l1_fft, _ = apply_fft_images(split[1], reshape=True)
split_l2_fft, _ = apply_fft_images(split[2], reshape=True)
split_l3_fft, _ = apply_fft_images(split[3], reshape=True)

norm = 5.
pyramid_draw([im_data, im_rec, split[0], split[1], split[2], split[3], 
	np.log(im_fft)/norm-0.5, np.log(im_rec_fft)/norm-0.5, 
	np.log(split_l0_fft)/norm-0.5, np.log(split_l1_fft)/norm-0.5,
	np.log(split_l2_fft)/norm-0.5, np.log(split_l3_fft)/norm-0.5], 
	path='logs/fft_split.png', im_shape=(10, 128, 128, 3))

#pyramid_sh = list()
#for pi in pyramid_full:
#	pyramid_sh.append(freq_shift(pi))

#pyramid_re = list()
#for pi in pyramid_sh:
#	pyramid_re.append(freq_shift(pi))

### plot pyramid
#pyramid_draw(pyramid_full, path='logs/pyramid_org.png')

### plot
#fig = plt.figure(0, figsize=(8,8))
#ax = fig.add_subplot(1,1,1)
#pa = ax.imshow((im_uso[0]+1.)/2.)
#fig.savefig('logs/im_uso_pad.jpg', dpi=300)
#plt.show()
#
#fig = plt.figure(0, figsize=(8,8))
#ax = fig.add_subplot(1,1,1)
#pa = ax.imshow((pyramid[0][0]+1.)/2.)
#fig.savefig('logs/im_dso.jpg', dpi=300)
#plt.show()
#
#fig = plt.figure(0, figsize=(8,8))
#ax = fig.add_subplot(1,1,1)
#pa = ax.imshow((pyramid[1][0]+1.)/2.)
#fig.savefig('logs/im_uso.jpg', dpi=300)
#plt.show()
#
#fig = plt.figure(0, figsize=(8,8))
#ax = fig.add_subplot(1,1,1)
#pa = ax.imshow((pyramid[2][0]+1.)/2.)
#fig.savefig('logs/im_bluro.jpg', dpi=300)
#plt.show()
#
#fig = plt.figure(0, figsize=(8,8))
#ax = fig.add_subplot(1,1,1)
#pa = ax.imshow((reconst[0]+1.)/2.)
#fig.savefig('logs/im_reconst.jpg', dpi=300)
#plt.show()
#
#fig = plt.figure(0, figsize=(8,8))
#ax = fig.add_subplot(1,1,1)
#pa = ax.imshow((im_data[0]+1.)/2.)
#fig.savefig('logs/im.jpg', dpi=300)
#plt.show()
#
#fig = plt.figure(0, figsize=(8,8))
#ax = fig.add_subplot(1,1,1)
#pa = ax.imshow((im_diff[0]+1.)/2.)
#fig.savefig('logs/im_diff.jpg', dpi=300)
#plt.show()