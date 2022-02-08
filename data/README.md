Qui hai la possibilità di definire <b>le cartelle contenenti il dato</b>.
Ogni cartella è nominata con il seguente pattern: [famiglia]_[note aggiuntive]. L'underscore è importante inserirlo 
come separatore tra la tipologia e le note aggiuntive.Inoltre, è importante il naming poichè ad ogni <b>tipologia</b> è associato un <b>modulo di processamento</b> 
presente in src/datasets.

Ogni cartella contenente il dataset deve essere cosi strutturata:
<ul>
	<li><b>annotations</b>: conterra le annotazioni</li>
	<li>cartelle contenente le immagini</li>
	<li><b>tfrecord: </b> conterrà i dataset in tfrecord ognuno secondo la sua [tipologia]. Ogni tipologia corrisponde ad una cartella</li>
</ul>