"""
Questo script consente di effettuare una somma pesata tra i pesi salvati a diversi checkpoint
Utilizzano questo script durante l allenamento della rete. La mia idea è invece quella di salvare i checpoint
e di mediarli successivamente. Com è possibile notare da questa formula:

                        (1 - 0.999) * ema_old_weights[j] + 0.999 * updating_weights[j]

agli old_weights, o meglio i pesi di un checkpoint precedente, viene dato meno peso rispetto a quello del checkpoint più recente (updating_weights).
Questo perchè parto dal concetto che la rete migliora di epoca in epoca.

Paper: https://arxiv.org/abs/1806.04498
"""
import os
from model import G1

Config_file = __import__('0_config_utils')
config = Config_file.Config()

def add(model, model_ema, beta=0.9999):
    # for each model layer index
    for i in range(len(model.layers)):
        if i > 0 :
            updating_weights = model.layers[i].get_weights()  # original model's weights
            ema_old_weights = model_ema.layers[i].get_weights()  # ema model's weights
            ema_new_weights = []  # ema model's update weights

            # for each weight tensor of original model's weights list
            for j in range(len(updating_weights)):
                n_weight = (1 - 0.999) * ema_old_weights[j] + 0.999 * updating_weights[j]
                ema_new_weights.append(n_weight)
            # update weights
            model_ema.layers[i].set_weights(ema_new_weights)

    return model_ema

list_epochs = "_"
model_ema = None
print(sorted(os.listdir('./weights'), key=lambda x: x.split('epoch_')[1].split('_')[0]))
for i, checkpoint in enumerate(sorted(os.listdir('./weights'), key=lambda x: x.split('epoch_')[1].split('_')[0])):
    epoch = checkpoint.split('epoch_')[1].split('_')[0]
    list_epochs += epoch+"_"
    if i == 0:
        model_ema = G1.build_model(config)
        model_ema.load_weights('./weights/'+checkpoint)
    else:
        model = G1.build_model(config)
        model.load_weights('./weights/' + checkpoint)
        model_ema = add(model_ema, model)


# old_weights = model_ema.get_weights()
# new_weights = []
# for j in range(len(old_weights)):
#         n_weight = old_weights[j] / i
#         new_weights.append(n_weight)
#
#
#
# model_ema.set_weights(new_weights)
model_ema.save_weights("ema"+list_epochs+".hdf5")