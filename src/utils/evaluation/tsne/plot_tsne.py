import os
import numpy as np
import matplotlib.pyplot as plt

def plot(dict_features, list_perplexity, dir_to_save, key_image_interested):
    name_dir_plots = os.path.join(dir_to_save, "plots")
    print("\n- Avvio il plot")
    print("\n- Immagine di interesse scelta: ", dict_features[key_image_interested]['path_target'])

    text=""
    for perplexity in list_perplexity:
        print("\n- Analizzo perplexity", perplexity)
        tsne_features_real = np.array([dict_features[k]['features_tsne_real_'+str(perplexity)] for k, v in dict_features.items()])
        tsne_features_generated = np.array([dict_features[k]['features_tsne_generated_'+str(perplexity)] for k, v in dict_features.items()])

        # Plot
        fig, ax = plt.subplots()
        ax.scatter(x=tsne_features_real[:,0], y=tsne_features_real[:,1], color='red', label="Real")
        ax.scatter(x=tsne_features_generated[:,0], y=tsne_features_generated[:,1], color='blue', label="Generated")

        # Immagine di interesse
        tsne_interested_real = dict_features[key_image_interested]['features_tsne_real_'+str(perplexity)]
        tsne_interested_generated = dict_features[key_image_interested]['features_tsne_generated_'+str(perplexity)]

        ax.scatter(x=tsne_interested_real[0], y=tsne_interested_real[1], color='orange', label="INTERESTED_real")
        ax.scatter(x=tsne_interested_generated[0], y=tsne_interested_generated[1], color='black', label="INTERESTED_generated")

        # Calcolo distanza euclidea
        distance = np.linalg.norm(tsne_interested_real - tsne_interested_generated)

        text += "Resoconto perplexity " + str(perplexity) + ":" \
               "\nCounter_image: " + str(key_image_interested) \
               + "\nPath: " + str(dict_features[key_image_interested]['path_target']) \
               + "\nDistance: " + str(distance) \
               + "\n\n##########\n\n"

        plt.legend()
        plt.savefig("tsne_perplexity_"+str(perplexity))

    with open(os.path.join(name_dir_plots, "Resoconto_tsne.txt"), "w") as f:
        f.write(text)