FROM nvidia/cuda:11.3.0-runtime-ubuntu20.04
RUN apt-get update
RUN apt update
RUN apt install -y python3 python3-pip git
RUN apt install -y libgl1-mesa-glx
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

COPY ./ /PIMP/VMP
WORKDIR /PIMP/VMP

RUN pip3 install -e .
RUN pip3 install -r requirements.txt

WORKDIR /PIMP/

RUN curl -sL https://deb.nodesource.com/setup_12.x | bash - 
RUN apt-get install -y nodejs

RUN pip3 install jupyterlab
CMD jupyter lab --ip=0.0.0.0 --allow-root
