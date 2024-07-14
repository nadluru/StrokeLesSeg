FROM nvcr.io/nvidia/pytorch:22.01-py3

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output /nnunet_data \
    && chown algorithm:algorithm /opt/algorithm /input /output /nnunet_data

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

COPY requirements.txt /opt/algorithm/
RUN python -m pip install --user -r requirements.txt

COPY nnUNet /opt/algorithm/nnUNet
RUN cd /opt/algorithm/nnUNet \
	&& pip install --no-cache-dir -e .

COPY plan*.sh /opt/algorithm/
COPY train*.sh /opt/algorithm/


