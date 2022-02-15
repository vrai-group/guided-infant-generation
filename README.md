
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

# Usage

## dataset
In order to use the framework you have the possibility to define your own dataset, containing depth images of 16 bit and 8 bit of size hegth=480, width=640. Each defined dataset must be placed in the <i>data</i> directory and comply with naming rules. 
For more information, you can refer to the README in the 
<a href="https://github.com/GiuseppeCannata/BabyPoseGuided/tree/master/data">data/REAMDE.md</a> folder.

In addition, you can download the MINI-RGBD dataset. 
For more information, you can refer to the README in the 
<a href="https://github.com/GiuseppeCannata/BabyPoseGuided/tree/master/data/Syntetich_complete">data/Syntetich_complete/REAMDE.md</a> folder.

## models
To use the framework you have the possibility to define your own architecture or use the existing ones.
For more information refer to the README in the 
<a href="https://github.com/GiuseppeCannata/BabyPoseGuided/tree/master/src/models">src/models/README.md</a> folder




