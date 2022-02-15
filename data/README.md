Here you have the possibility to define the <b>datasets</b>.
Each folder is named with the following pattern: [type][underscore][notes].
Each <b>type</b> has an associated <b>python processing module</b> in src/datasets.

Each folder containing the dataset must be structured as follows:
<ul>
	<li><b>folders containing the images:</b> each folder must be named with pz[id_unique] where <i>id_unique</i> is an integer numeric id that uniquely recognises infants</li>
	<li><b>annotations</b>: this directory, for each of the folders pz[id_unique] will contain the relevant image annotations</li>
	<li><b>tfrecord: </b> will contain the datasets in <i>.tfrecord</i>. 
	Within this folder we can find several sub-folders, each corresponding to a particular "configuration" of the dataset.
	For example, in one configuration we set radius_key=1 while in the other radius_key=2 and so on.
	Each "configuration" is described in a pickle file <i>(.pkl)</i> called <i>sets_configs.pkl</i>.
	<br>The file <i>.tfrecord</i> and <i>sets_config.pkl</i> are created by the src/0_DatasetGenerator.py.</li>
</ul>

As an example, consider the structure in <i>Syntetich_complete</i>