import numpy as np

history_GAN = np.load("history_GAN.npy", allow_pickle=True)

epoch = 200
history_GAN_new = {'epoch': 0,
                       'loss_train_G2': np.empty((epoch)),
                       'loss_train_D': np.empty((epoch)),
                       'loss_train_fake_D': np.empty((epoch)),
                       'loss_train_real_D': np.empty((epoch)),
                       'ssim_train': np.empty((epoch)),
                       'mask_ssim_train': np.empty((epoch)),
                       'num_real_I_PT2_train': np.empty((epoch), dtype=np.uint32),
                       'num_real_Ic_train': np.empty((epoch), dtype=np.uint32),
                       'num_real_It_train': np.empty((epoch), dtype=np.uint32),

                       'loss_valid_G2': np.empty((epoch)),
                       'loss_valid_D': np.empty((epoch)),
                       'loss_valid_fake_D': np.empty((epoch)),
                       'loss_valid_real_D': np.empty((epoch)),
                       'ssim_valid': np.empty((epoch)),
                       'mask_ssim_valid': np.empty((epoch)),
                       'num_real_I_PT2_valid': np.empty((epoch), dtype=np.uint32),
                       'num_real_Ic_valid': np.empty((epoch), dtype=np.uint32),
                       'num_real_It_valid': np.empty((epoch), dtype=np.uint32),
                       }

epoch = history_GAN[()]['epoch']
loss_train_G2 = history_GAN[()]['loss_train_G2'][:epoch]
loss_train_D = history_GAN[()]['loss_train_D'][:epoch]
loss_train_real_D = history_GAN[()]['loss_train_real_D'][:epoch]
loss_train_fake_D = history_GAN[()]['loss_train_fake_D'][:epoch]
ssim_train = history_GAN[()]['ssim_train'][:epoch],
mask_ssim_train = history_GAN[()]['mask_ssim_train'][:epoch],
num_real_I_PT2_train = history_GAN[()]['r_r_train'][:epoch]
num_real_Ic_train = history_GAN[()]['img_0_train'][:epoch]
num_real_It_train = history_GAN[()]['img_1_train'][:epoch]

loss_values_valid_G2 = history_GAN[()]['loss_valid_G2'][:epoch]
loss_values_valid_D = history_GAN[()]['loss_valid_D'][:epoch]
loss_values_valid_fake_D = history_GAN[()]['loss_valid_fake_D'][:epoch]
loss_values_valid_real_D = history_GAN[()]['loss_valid_real_D'][:epoch]
ssim_valid = history_GAN[()]['ssim_valid'][:epoch]
mask_ssim_valid = history_GAN[()]['mask_ssim_valid'][:epoch]
num_real_I_PT2_valid = history_GAN[()]['r_r_valid'][:epoch]
num_real_Ic_valid = history_GAN[()]['img_0_valid'][:epoch]
num_real_It_valid = history_GAN[()]['img_1_valid'][:epoch]


history_GAN_new['epoch'] = epoch
history_GAN_new['loss_train_G2'] = loss_train_G2
history_GAN_new['loss_train_D'] = loss_train_D
history_GAN_new['loss_train_fake_D'] = loss_train_fake_D
history_GAN_new['loss_train_real_D'] = loss_train_real_D
history_GAN_new['ssim_train'] = ssim_train
history_GAN_new['mask_ssim_train'] = mask_ssim_train
history_GAN_new['num_real_I_PT2_train']= num_real_I_PT2_train
history_GAN_new['num_real_Ic_train']= num_real_Ic_train
history_GAN_new['num_real_It_train']= num_real_It_train
history_GAN_new['loss_valid_G2'] = loss_values_valid_G2
history_GAN_new['loss_valid_D'] = loss_values_valid_D
history_GAN_new['loss_valid_fake_D'] = loss_values_valid_fake_D
history_GAN_new['loss_valid_real_D'] = loss_values_valid_real_D
history_GAN_new['ssim_valid'] = ssim_valid
history_GAN_new['mask_ssim_valid'] = mask_ssim_valid
history_GAN_new['num_real_I_PT2_valid'] = num_real_I_PT2_valid
history_GAN_new['num_real_Ic_valid'] = num_real_Ic_valid
history_GAN_new['num_real_It_valid'] = num_real_It_valid
np.save('./history_GAN_new.npy', history_GAN_new)

