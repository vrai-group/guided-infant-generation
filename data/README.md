Qui hai la possibilità di definire <b>le cartelle contenenti i dataset</b>.
Ogni cartella è nominata con il seguente pattern: [tipologia]_[note aggiuntive].
è importante il naming poichè ad ogni <b>tipologia</b> è associato un modulo di processamento presente in src/datasets.

Ogni cartella contenente il dataset deve essere cosi strutturata:
<ul>
	<li><b>annotations</b>: conterra le annotazioni</li>
	<li>cartelle contenente le immagini</li>
	<li><b>tfrecord</b:> conterrà i tfrecords del dataset</li>
</ul>