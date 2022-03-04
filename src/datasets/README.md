Here you have the possibility to define the <b>python processing module</b> of a specific dataset.
Each python processing module is named with the [type] of the related dataset (for more info on dataset [type] see the README in <a href="../../data">../../data</a>).

Each python processing module should contain the dataset processing methods.
The methods defined are:
<ul>
	<li>process_image: preprocess the image</li>
	<li>unprocess_image: unprocess the image</li>
	<li>get_unprocess_dataset: get the tfrecord file not applying pre-processing</li>
	<li>preprocess_dataset: preprocess the tfrecord file</li>
</ul>