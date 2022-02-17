
# Generating depth images of preterm infants with a given pose using GANs

<img src="./resources/workflow.png">

# Install enviroment

Code was run using the dependencies described in <i>Dockerfile</i>. To prepare the environment, follow these steps:

1. Downloading the Github repository
```
sudo apt install unzip
wget https://github.com/GiuseppeCannata/BabyPoseGuided/archive/master.zip
unzip master.zip
mv BabyPoseGuided-master BabyPoseGuided
```
2. Move in BabyPoseGuided directory
```
cd BabyPoseGuided
```
3. Build the Dockerfile to obtain the docker image. In the following instruction, we buid the Docker file in the direcory ./ calling the docker image <i>ganinfant</i> version <i>1.0</i>
```
docker build -t ganinfant:1.0 ./
```
4. We start the container <i>GANinfantcontainer</i> with the docker image <i>ganinfant:1.0</i>. 
In the following instruction we replace <id_gpu> with your GPU id, <local_directory> with the absolute path of the 
BabyPoseGuided directory described at point (1.) , <container_directory> mapping of the local BabyPoseGuided 
directory to the container
```
docker run --name "GANinfantcontainer" --gpus "<id_gpu>"  -v <local_directory>:<container_directory> -it ganinfant:1.0 bash 
```

# Define dataset
In order to use the framework you have the possibility to define your own dataset, containing depth images of 16 bit and 8 bit of size hegth=480, width=640. Each defined dataset must be placed in the <i>data</i> directory and comply with naming rules. 
For more information, you can refer to the README in the 
<a href="https://github.com/GiuseppeCannata/BabyPoseGuided/tree/master/data">data/REAMDE.md</a> folder.

In addition, you can download the MINI-RGBD dataset. 
For more information, you can refer to the README in the 
<a href="https://github.com/GiuseppeCannata/BabyPoseGuided/tree/master/data/Syntetich_complete">data/Syntetich_complete/REAMDE.md</a> folder.

# Define models
To use the framework you have the possibility to define your own architecture or use the existing ones.
For more information refer to the README in the 
<a href="https://github.com/GiuseppeCannata/BabyPoseGuided/tree/master/src/models">src/models/README.md</a> folder

# Usage
Once the dataset and models have been defined, the framework can be used.
In particular we need to set the <i>enviroment variables</i> defined in the src/CONFIG.py file.

<dl>
<dt><b>- MODE</b></dt>
<dd>
Specify the mode to start the framework. The list of MODE value is given below:
    <ul>
        <li>'train_G1': train generator G1 </li> 
        <li>'train_cDCGAN': train the conditional Generative Adversarial Network </li>
        <li>'evaluate_G1': calculate FID and IS scores of G1 </li>
        <li>'evaluate_GAN': calculate FID and IS scores of conditional Generative Adversarial Network </li>
        <li>'tsne_GAN': calculate tsne of all framework </li>
        <li>'inference_G1': inference on test set using G1 </li>
        <li>'inference_GAN': inference on test set using conditional Generative Adversarial Network </li>
		<li>'plot_history_G1':  plot history file of G1 training</li>
		<li>'plot_history_GAN': plot history file of GAN training</li>
    </ul>
</dd>

<dt><b>- DATASET</b></dt>
<dd>
Name of the dataset you want to use. 
The directory of dataset must be contained in <a href="https://github.com/GiuseppeCannata/BabyPoseGuided/tree/master/data">data</a> directory.
</dd>

<dt><b>- DATASET_CONFIGURATION</b></dt>
<dd>
Name of the dataset "configuration" you want to use. 
The directory of configuration must be contained in DATASET/tfrecord/DATASET_CONFIGURATION
</dd>

<dt><b>- ARCHITECTURE</b></dt>
<dd>
Name of the architecture you want to use. 
The directory of architecture must be contained in <a href="https://github.com/GiuseppeCannata/BabyPoseGuided/tree/master/src/models">models</a> directory.
</dd>

<dt><b>- OUTPUTS_DIR</b></dt>
<dd>
Name of directory in which to save the results of training, validation and inference.
</dd>

<dt><b>- G1_NAME_WEIGHTS_FILE </b></dt>
<dd>
Name of .hdf5 file to load in G1 model. 
</dd>

<dt><b>- G2_NAME_WEIGHTS_FILE</b></dt>
<dd>
Name of .hdf5 file to load in G2 model
</dd>

</dl>

Once the environment variables have been set, start src/main.py.


# Qualitative Results

<ul>
	<li> I_PT2 = I_PT1 + I_D </li>
	<li> I_PT1 = output of G1 generator </li>
	<li> I_D = output of G2 generator </li>
	<li> Ic = condition image </li>
	<li> It = target image </li>
	<li> Pt = target pose </li>
	<li> Mt = target binary mask </li>
</ul>

<table>
    <tr><th><img src="./resources/220-pz108_00100-pz104_00100.png"></th></tr>
	<tr><th><img src="./resources/231-pz108_00155-pz104_00155.png"></th></tr>
	<tr><th><img src="./resources/362-pz108_00810-pz104_00810.png"></th></tr>
	<tr><th><img src="./resources/389-pz108_00945-pz104_00945.png"></th></tr>
</table>





