import os
import numpy as np
import matplotlib.pyplot as plt

logs_loss_mask_ssim = np.load(os.path.join('logs', 'logs_mask_ssim.npy'))
logs_ssim = np.load(os.path.join('logs', 'logs_ssim.npy'))
logs_train_D_fake = np.load(os.path.join('logs', 'logs_loss_train_D_fake.npy'))
logs_train_D_real = np.load(os.path.join('logs', 'logs_loss_train_D_real.npy'))
logs_loss_train_D_total = np.load(os.path.join('logs', 'logs_loss_train_D_total.npy'))
logs_loss_train_G2 = np.load(os.path.join('logs', 'logs_loss_train_G2.npy'))
logs_real_img_0 = np.load(os.path.join('logs', 'logs_img_0.npy'))
logs_real_img_1 = np.load(os.path.join('logs', 'logs_img_1.npy'))
logs_real_r_r = np.load(os.path.join('logs', 'logs_r_r.npy'))

fig, axs = plt.subplots(3)
axs[0].plot(logs_loss_train_G2, label='G2_loss')
axs[0].plot(logs_train_D_fake, label='D_fake')
axs[0].plot(logs_train_D_real, label='D_real')
axs[0].legend(loc='upper right')

axs[1].plot(logs_ssim, label='ssim')
axs[1].legend(loc='upper right')

axs[2].plot(logs_real_img_0, label='logs_real_img_0')
axs[2].plot(logs_real_img_1, label='logs_real_img_1')
axs[2].plot(logs_real_r_r, label='logs_real_r_r')
axs[2].legend(loc='upper right')

arg_max = np.argmax(logs_ssim)
max = logs_ssim[arg_max]
print("SSIM: max_value_{max_value}  epoch_{epoch}".format(max_value=max, epoch=arg_max))

plt.show()

