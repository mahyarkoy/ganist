# %%
from util import read_celeba, create_lsun, apply_fft_win
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

lsun_path = '/dresden/users/mk1391/evl/Data/lsun/bedroom_train_lmdb'
im_size = 128
data_size = 100
images_celeba = read_celeba(im_size, data_size)
images_bedrooms, _ = create_lsun(lsun_path, im_size, data_size)

# %%
#plt.imshow((images_celeba[0]+1)/2)
#plt.imshow((images_bedrooms[0]+1)/2)
#Image.fromarray(np.clip(np.rint((images_celeba[0]+1)*255./2), 0., 255.).astype(np.uint8),'RGB').show()

# %%
fft_celeba = apply_fft_win(images_celeba, 'logs_draw/fft_celeba_128.png', windowing=True)
fft_bedrooms = apply_fft_win(images_bedrooms, 'logs_draw/fft_bedrooms_128.png', windowing=True)
fft_average_power_celeba = np.mean(np.abs(fft_celeba)**2, axis=0)
fft_average_power_bedrooms = np.mean(np.abs(fft_bedrooms)**2, axis=0)

# %%
plt.close()
fig = plt.figure(0)

ax = fig.add_subplot(1, 3, 1)
imax = ax.imshow(np.log(fft_average_power_celeba/np.amax(fft_average_power_celeba)), vmin=-13, vmax=0, cmap=plt.get_cmap('inferno'))
ax.set_title('CelebA 128')
dft_size = im_size
ticks_loc_x = [0, dft_size//2]
ticks_loc_y = [0, dft_size//2-1, dft_size-dft_size%2-1]
ax.set_xticks(ticks_loc_x)
ax.set_xticklabels([-0.5, 0])
ax.set_yticks(ticks_loc_y)
ax.set_yticklabels(['', 0, -0.5])
plt.colorbar(imax, ax=ax, fraction=0.046, pad=0.04)

ax = fig.add_subplot(1, 3, 2)
imax = ax.imshow(np.log(fft_average_power_bedrooms/np.amax(fft_average_power_bedrooms)), vmin=-13, vmax=0, cmap=plt.get_cmap('inferno'))
ax.set_title('Bedrooms 128')
dft_size = im_size
ticks_loc_x = [0, dft_size//2]
ticks_loc_y = [0, dft_size//2-1, dft_size-dft_size%2-1]
ax.set_xticks(ticks_loc_x)
ax.set_xticklabels([-0.5, 0])
ax.set_yticks(ticks_loc_y)
ax.set_yticklabels(['', 0, -0.5])
plt.colorbar(imax, ax=ax, fraction=0.046, pad=0.04)

ax = fig.add_subplot(1, 3, 3)
imax = ax.imshow(np.log(fft_average_power_celeba) - np.log(fft_average_power_bedrooms), vmin=-4, vmax=4, cmap=plt.get_cmap('seismic'))
ax.set_title('Diff')
dft_size = im_size
ticks_loc_x = [0, dft_size//2]
ticks_loc_y = [0, dft_size//2-1, dft_size-dft_size%2-1]
ax.set_xticks(ticks_loc_x)
ax.set_xticklabels([-0.5, 0])
ax.set_yticks(ticks_loc_y)
ax.set_yticklabels(['', 0, -0.5])
plt.colorbar(imax, ax=ax, fraction=0.046, pad=0.04)

fig.tight_layout()
#plt.show()
fig.savefig('logs_draw/fft_celeba_bedrooms_128_agg.png', dpi=300)
plt.close(fig)

# %%
