FROM tensorflow/tensorflow:2.5.0-gpu
RUN apt-get install rar
RUN apt-get install nano
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install matplotlib
RUN pip3 install scipy
RUN pip3 install opencv-python
RUN pip3 install pandas
RUN pip3 install scikit-image

#TODO aggiungere wget al repo github
#Opzioni di run
#docker build -t ganinfant:1.0 .
#docker run --name "GANinfantcontainer" --gpus "device=2"  -v <directory locale>:<directory sul container> -it ganinfant:1.0 bash 