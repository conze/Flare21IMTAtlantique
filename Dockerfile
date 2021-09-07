FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
	# we have found python3.7 in base docker
	python3-pip \
	python3-setuptools \
	build-essential \
	&& \
	apt-get clean && \
	python -m pip install --upgrade pip

WORKDIR /workspace
COPY ./ /workspace

RUN pip3 install pip -U
RUN pip3 install -U scikit-image
RUN pip3 install xlrd
RUN pip3 install nibabel
RUN pip3 install torch
RUN pip3 install torchvision

CMD ["bash", "predict.sh"]
