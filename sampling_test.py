import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import os
from run_ganist import readim_path_from_dir, readim_from_path
from tf_ganist import tf_upsample, tf_downsample, tf_binomial_blur, tf_make_lap_pyramid, tf_reconst_lap_pyramid
run_seed = 0
np.random.seed(run_seed)
tf.set_random_seed(run_seed)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0, 1" for multiple

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
sess = tf.Session(config=config)

### read celeba 128
im_dir = '/media/evl/Public/Mahyar/Data/celeba/img_align_celeba/'
im_size = 128
dataset_size = 10
im_paths = readim_path_from_dir(im_dir)
np.random.shuffle(im_paths)
im_data = readim_from_path(im_paths[:dataset_size], im_size, center_crop=(121, 89))

### make sampling ops
im_ds = tf_downsample(im_data)
im_us = tf_upsample(im_data)
im_blur = tf_binomial_blur(im_data)
pyramid = tf_make_lap_pyramid(im_data)
reconst = tf_reconst_lap_pyramid(pyramid)

### apply sampling
im_dso, im_uso, im_bluro, pyramid, reconst = sess.run([im_ds, im_us, im_blur, pyramid, reconst])
im_diff = im_data - reconst

### plot
fig = plt.figure(0, figsize=(8,8))
ax = fig.add_subplot(1,1,1)
pa = ax.imshow((im_uso[0]+1.)/2.)
fig.savefig('logs/im_uso_pad.jpg', dpi=300)
plt.show()

fig = plt.figure(0, figsize=(8,8))
ax = fig.add_subplot(1,1,1)
pa = ax.imshow((pyramid[0][0]+1.)/2.)
fig.savefig('logs/im_dso.jpg', dpi=300)
plt.show()

fig = plt.figure(0, figsize=(8,8))
ax = fig.add_subplot(1,1,1)
pa = ax.imshow((pyramid[1][0]+1.)/2.)
fig.savefig('logs/im_uso.jpg', dpi=300)
plt.show()

fig = plt.figure(0, figsize=(8,8))
ax = fig.add_subplot(1,1,1)
pa = ax.imshow((pyramid[2][0]+1.)/2.)
fig.savefig('logs/im_bluro.jpg', dpi=300)
plt.show()

fig = plt.figure(0, figsize=(8,8))
ax = fig.add_subplot(1,1,1)
pa = ax.imshow((reconst[0]+1.)/2.)
fig.savefig('logs/im_reconst.jpg', dpi=300)
plt.show()

fig = plt.figure(0, figsize=(8,8))
ax = fig.add_subplot(1,1,1)
pa = ax.imshow((im_data[0]+1.)/2.)
fig.savefig('logs/im.jpg', dpi=300)
plt.show()

fig = plt.figure(0, figsize=(8,8))
ax = fig.add_subplot(1,1,1)
pa = ax.imshow((im_diff[0]+1.)/2.)
fig.savefig('logs/im_diff.jpg', dpi=300)
plt.show()