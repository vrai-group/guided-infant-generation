In questa cartella è presente il dataset.
Ogni cartella è nominata con pz<num_unique_infante>. Per ogni pz abbiamo:
- 1000 immagini a 8 bit
- 1000 immagini a 16 bit
Per visulaizzare le 16 bit scaricare il programma Fiji. 
La dimensione delle immagini è altezza = 480 pixels mentre la larghezza = 640 pixels.

Il dataset originale è disponibile <a href="https://www.iosb.fraunhofer.de/en/competences/image-exploitation/object-recognition/sensor-networks/motion-analysis.html ">MINI-RGBD</a> (7 Gb) 
Di quest'ultimo sono state considerate solo le immagini di depth.
Inoltre sono state effettuate le seguenti trasformazioni per ottenere il Syntetich complete:
- Rotazione di 90 gradi in senso orario delle immagini
- Rotazione di 90 delle relative annotazioni
- Semplificazione delle annotazioni dei KP (da 24 a 14)

Le nuove annotazioni sono presenti nella cartella annotations, divise in file csv. Ogni file corrisponde ad un pz.
Aprendo il file, per ogni immagine abbiamo le relative annotazioni dei keypoints. In particolare, le annotazioni sono cosi scritte: <coordinata_x>,<coordinata_y>.
I keypoints considerati sono:
<br>
<img src="./annotations.png">
<br><br>

Nella cartella tfrecord sono presenti i set di: train, valid e test per la parte di training e validazione.
Inoltre, il file pair_tot_sets.pkl viene creato in automatico durante la creazione dei tfrecods e contiene le info: radius_head, radius_keypoints utilizzate durante la formazione dei sets.





