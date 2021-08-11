import imageio
import os
filenames = os.listdir('./pred_train')
with imageio.get_writer('./movie_train.gif', mode='I', fps= 4) as writer:
    for filename in filenames:
        image = imageio.imread('./pred_train/'+filename)
        writer.append_data(image)