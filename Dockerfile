FROM tensorflow/tensorflow:2.5.0-gpu
RUN apt-get install rar
RUN apt-get install nano
#RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install matplotlib==3.3.4
RUN pip3 install scipy==1.5.4
RUN pip3 install opencv-python==4.5.2.54
RUN pip3 install pandas==1.1.5
RUN pip3 install imageio==2.9.0
RUN pip3 install scikit-image==0.17.2
RUN pip3 install PyYAML==6.0
#RUN cd src
