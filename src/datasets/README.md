Qui hai la possibilità di definire il <b>modulo di preprocessamento del dataset</b> che contenga i metodi per l'elaborazione del dataset.
I metodi possono essere ad esempio:
<ul>
	<li>Preprocess/Unprocess immagine</li>
	<li>Preprocess/Unprocess tfrecord dataset</li>
</ul>
Ogni modulo è chiamato con la <b>tipologia del dataset</b>, e verrà invocato automaticamente dal framework in base al dataset scelto.
Ex: Syntetich_complete