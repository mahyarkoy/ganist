import numpy as np
import tensorflow as tf
from run_ganist import readim_path_from_dir, readim_from_path, TFutil

'''
TENSORFLOW SETUP
'''
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
### create a ganist instance
#ganist = tf_ganist.Ganist(sess, log_path_sum)
### init variables
#sess.run(tf.global_variables_initializer())
### init TFutil
tfutil = TFutil(sess)

'''
Drawing Freq Components
'''
def freq_shift(im, fc_x, fc_y):
	kernel_loc = 2.*np.pi*fc_x * np.arange(im_size).reshape((1, im_size, 1)) + \
		2.*np.pi*fc_y * np.arange(im_size).reshape((im_size, 1, 1))
	kernel_cos = np.cos(kernel_loc)
	return im * kernel_cos