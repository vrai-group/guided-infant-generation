Qui hai la possibilità di definire <b>i dataset</b>.
Ogni cartella è nominata con il seguente pattern: [tipologia][underscore][note aggiuntive].
Ad ogni tipologia, è associato un <b>modulo di processamento</b> che possiamo definire src/datasets.

Ogni cartella contenente il dataset deve essere cosi strutturata:
<ul>
	<li><b>annotations</b>: conterra le annotazioni</li>
	<li>cartelle contenente le immagini</li>
	<li><b>tfrecord: </b> conterrà i dataset in formato <i>.tfrecord</i>. 
	All'interno di questa cartella possiamo trovare diverse sottocartele ognuna corrispondete ad una particolare "configurazione" del dataset.
	Ad esempio in una configurazione poniamo radius_key=1 mentre nell'altra radius_key=2 e via discorrendo.
	Ogni "configurazione" è descritta in un file pickle <i>(.pkl)</i> chiamato <i>sets_configs.pkl</i></li>
</ul>