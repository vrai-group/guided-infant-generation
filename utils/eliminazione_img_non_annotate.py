"""
Questo script serve per considerare solamente tutte le immagini che sono annotate.
Le restanti viene spostato in data/BabyPose/Non annotate
"""
import os
import pandas as pd
import shutil

Config_file = __import__('0_config_utils')
config = Config_file.Config()

for name_file_annotation in os.listdir(config.data_annotations_path):
    pz = name_file_annotation.split('_')[1].split('.')[0]
    if not os.path.exists(os.path.join('./data/BabyPose', 'non annotate', pz)):
        os.mkdir(os.path.join('./data/BabyPose', 'non annotate', pz))
        
        path_annotation = os.path.join(config.data_annotations_path, name_file_annotation)
        df_annotation = pd.read_csv(path_annotation, delimiter=';')

        list_pz_imgs = os.listdir(os.path.join('./data/BabyPose', pz))
        list_pz_imgs = [f for f in list_pz_imgs if '_8bit' in f]

        for img in list_pz_imgs:

            # L'immagine è presente nella lista dei log per cui non dobbiamo considerarla
            a = df_annotation['image'].values
            if img in a:
                continue
            else:
                sedici = img.split('_')[0]+'_16bit.png'

                shutil.move("./data/BabyPose/"+pz+"/"+img, os.path.join('./data/BabyPose', 'non annotate', pz))
                shutil.move('./data/BabyPose/' + pz + '/' + sedici, os.path.join('./data/BabyPose', 'non annotate', pz))


