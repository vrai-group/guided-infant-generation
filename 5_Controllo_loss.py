import os
import matplotlib.pyplot as plt

list = sorted(os.listdir("./weights"), key=lambda x: int(x.split('epoch_')[1].split('-')[0]))

loss = []
mse = []
ssim = []

val_loss = []
val_ssim = []
val_mse = []


for f in list:
    print(f)
    loss_v = float(f.split('-loss_')[1].split('-')[0])
    loss.append(loss_v)
    mse_v = float(f.split('-mse_')[1].split('-')[0])
    mse.append(mse_v)
    ssim_v = float(f.split('-m_ssim')[1].split('-')[0])
    ssim.append(ssim_v)

    val_loss_v = float(f.split('val_loss_')[1].split('-')[0])
    val_loss.append(val_loss_v)
    val_mse_v = float(f.split('val_mse_')[1].split('-')[0])
    val_mse.append(val_mse_v)
    val_ssim_v = float(f.split('val_m_ssim_')[1].split('.h')[0])
    val_ssim.append(val_ssim_v)

fig, axs = plt.subplots(2)
axs[0].plot(range(len(loss)), loss, label = "loss")
axs[0].plot(range(len(val_loss)), val_loss, label = "val_loss")

#axs[1].plot(range(len(ssim[5:30])), ssim[5:30], label = "ssim")
axs[1].plot(range(len(val_ssim[:30])), val_ssim[:30],  label = "val_ssim")
print(max(val_ssim[:30]))
print(min(val_loss[:30]))

# axs[0].plot(mse, range(len(mse)), label = "mse")
# axs[1].plot(val_mse, range(len(val_mse)), label = "val_mse")
plt.legend()

plt.show()










