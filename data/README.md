Here you have the possibility to define the <b>datasets</b>.
 
<h3>1. Dataset directory structure</h3>

<b>Note</b>: As an example, consider the structure in <a href="./Syntetich_complete">./Syntetich_complete</a> dataset.
<br>
Each directory is named with the following pattern: [type][underscore][note]. Each directory be structured as follow:

```
├───[type][underscore][note]
	├───annotations
	├───pz[id unique]
	├───     .
	├───     .
	├───pz[id unique]
	└───tfrecord
		├───configuration_1
		├───configuration_2
		├───     .
		├───     .
		└───configuration_n
			├───[type]_train.tfrecord
			├───[type]_valid.tfrecord
			├───[type]_test.tfrecord
			└───set_configs.pkl
				
```

The content of each folder is described below:

<ul>
	<li><b>pz[id unique]</b>: these directories contain the images about a specific infant identified by the <i>id unique</i></li>
	<li><b>annotations</b>: this directory will contain the related annotations files for each of the pz[id unique] directory</li>
	<li><b>tfrecord</b>: this directory contain the dataset's <i>configurations</i>. The dataset configuration is the source data for the framework.
	Each configuration is rapresented by a sub-folder.<br>
	In each of them we have:
	<ul>
		<li>train/valid/test sets in <i>.tfrecord</i> format. These are the set to use during the training and evaluation phase</li>
		<li>sets_configs.pkl that describe the carateristics about the configuration <i>(radius_key=2, flip=True, etc..)</i></li>
		<li>dic_history.pkl dictonary in which we have for each sets the pair formed and the related positin in tfrecord file</li>
	</ul>
	These files are created by the <a href="./Dataset_configuration_generator.py"> ./Dataset_configuration_generator.py </a> script described in section 2.</li>
</ul>

<h3>2. Create dataset configuration with Dataset_configuration_generator.py</h3>

This script creates the dataset configuration that can be used by the framework.
To execute it, it is necessary to set the information on the <a href="./Dataset_configuration_generator.py#L343"> configuration part</a>. After that you can run the script:
```
python Dataset_configuration_generator.py
```
