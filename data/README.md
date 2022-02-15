Qui hai la possibilità di definire <b>i dataset</b>.
Ogni cartella è nominata con il seguente pattern: [tipologia][underscore][note aggiuntive].
Ad ogni <b>tipologia</b>, è associato un <b>modulo di processamento python</b> presente in src/datasets.

Ogni cartella contenente il dataset deve essere cosi strutturata:
<ul>
	<li><b>cartelle contenente le immagini:</b> ogni cartella deve essere nominato con pz[id_unique] in cui <i>id_unique</i> è un id numerico intero che riconosce univocamente gli infanti</li>
	<li><b>annotations</b>: questa directory, per ognuna delle cartelle pz[id_unique] conterrà le relative annotazioni delle immagini</li>
	<li><b>tfrecord: </b> conterrà i dataset in formato <i>.tfrecord</i>. 
	All'interno di questa cartella possiamo trovare diverse sottocartele ognuna corrispondete ad una particolare "configurazione" del dataset.
	Ad esempio in una configurazione poniamo radius_key=1 mentre nell'altra radius_key=2 e via discorrendo.
	Ogni "configurazione" è descritta in un file pickle <i>(.pkl)</i> chiamato <i>sets_configs.pkl</i>.
	<br>I file <i>.tfrecord</i> e il file <i>sets_config.pkl</i> sono creati dallo script src/0_DatasetGenerator.py.</li>
</ul>

Come esempio si consideri la struttura presente in <i>Syntetich_complete</i>