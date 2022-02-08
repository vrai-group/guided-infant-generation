Qui hai la possibilità di definire il <b>modulo di preprocessamento del dataset</b> che contenga i metodi per l'elaborazione del dataset.
I metodi possono essere ad esempio:
<ul>
	<li>Preprocess/Unprocess immagine</li>
	<li>Preprocess/Unprocess tfrecord dataset</li>
</ul>
Ogni modulo è nominato con la <b>famiglia</b> del dataset <i>(per maggiori info guarda il README in ../data)</i> e verrà invocato automaticamente dal framework in base al self.DATASET
inseirto all'interno della CONFIG.
Ex: Syntetich_complete