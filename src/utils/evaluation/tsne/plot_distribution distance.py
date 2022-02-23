import os
import numpy as np
import matplotlib.pyplot as plt

def _plot_distribution_distance(dict_features, list_perplexity, dir_to_save, type_dataset='test'):
    """
    Plot of BoxPlot of distance beetween generated (test) and target (real)
    """
    name_dir_plots = os.path.join(dir_to_save, "box_plots")
    os.makedirs(name_dir_plots, exist_ok=False)
    print("\n- Avvio il box_plot")

    for perplexity in list_perplexity:
        print("\n- Analizzo perplexity", perplexity)
        tsne_features_real = np.array([dict_features[k]['features_tsne_real_' + str(perplexity)] for k, v in dict_features.items() if 'test' in k])
        tsne_features_generated = np.array([dict_features[k]['features_tsne_generated_' + str(perplexity)] for k, v in dict_features.items() if 'test' in k])

        distances = []
        for id in range(tsne_features_real.shape[0]):
            distance = np.linalg.norm(tsne_features_real[id] - tsne_features_generated[id])
            distances.append(distance)

        # box_plot
        fig1, ax1 = plt.subplots()
        ax1.set_title('Perplexity '+str(perplexity))
        ax1.boxplot(distances)

        plt.savefig(os.path.join(name_dir_plots, "Box_plot_perplexity_"+str(perplexity)))
        plt.figure().clear()

        # Histogram
        plt.hist(distances, density=True, bins=30, color='b')  # density=False would make counts
        plt.ylabel('Count')
        plt.xlabel('Distance')
        plt.savefig(os.path.join(name_dir_plots, "Histogram_" + str(perplexity)))


# if __name__ == "__main__":
#     dic = np.load('./dict_vgg_pca_tsne_features_real_and_generated.npy', allow_pickle=True )[()]
#     list_perplexity = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100, 200, 300]
#     _plot_distribution_distance(dic, list_perplexity, './')
