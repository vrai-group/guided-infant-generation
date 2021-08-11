import os
import matplotlib.pyplot as plt

list = sorted(os.listdir("./weights"), key=lambda x: int(x.split('epoch_')[1].split('-')[0]))

loss = []
mse = []
ssim = []
mask_ssim = []

val_loss = []
val_ssim = []
val_mse = []
val_mask_ssim = []


for f in list:

    loss_v = float(f.split('-loss_')[1].split('-')[0])
    loss.append(loss_v)
    # mse_v = float(f.split('-mse_')[1].split('-')[0])
    # mse.append(mse_v)
    ssim_v = float(f.split('-ssim_')[1].split('-')[0])
    ssim.append(ssim_v)
    mask_ssim_v = float(f.split('-mask_ssim_')[1].split('-')[0])
    mask_ssim.append(mask_ssim_v)

    val_loss_v = float(f.split('-val_loss_')[1].split('-')[0])
    val_loss.append(val_loss_v)
    # val_mse_v = float(f.split('val_mse_')[1].split('-')[0])
    # val_mse.append(val_mse_v)
    val_ssim_v = float(f.split('-val_ssim_')[1].split('_')[0])
    val_ssim.append(val_ssim_v)
    val_mask_ssim_v = float(f.split('_val_mask_ssim_')[1].split('.h')[0])
    val_mask_ssim.append(val_mask_ssim_v)

# Plotting
fig, axs = plt.subplots(2)
axs[0].plot(range(len(loss)), loss, label = "loss")
axs[0].plot(range(len(val_loss)), val_loss, label = "val_loss")
axs[0].legend(loc='upper right')

axs[1].plot(range(len(ssim)), ssim, label = "ssim")
axs[1].plot(range(len(val_ssim)), val_ssim,  label = "val_ssim")
axs[1].legend(loc='lower right')

plt.show()

print("Max value SSIM train: ", max(ssim), " epoch: ", ssim.index(max(ssim)) + 1)
print("Max value SSIM valid: ", max(val_ssim),  " epoch: ", val_ssim.index(max(val_ssim)) + 1)

print("Min value LOSS train: ", min(loss), " epoch: ", loss.index(min(loss)) + 1)
print("Min value LOSS valid: ", min(val_loss),  " epoch: ", val_loss.index(min(val_loss)) + 1)










