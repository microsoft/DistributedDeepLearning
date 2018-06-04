# Ubuntu 16.04, CUDA 9.0
FROM nvidia/cuda:9.0-runtime-ubuntu16.04

ENV CNTK_VERSION=2.5.1
ENV CUDNN_VERSION=7.0.5.15-1+cuda9.0
ENV PYTHON_VERSION=3.5
ENV PY_VERSION=35
ENV NCCL_VERSION=2.1.15-1+cuda9.0

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        sudo \
        git \
        curl \
        nano \
        wget \
        ca-certificates \
        libopenmpi-dev \
        libcudnn7=$CUDNN_VERSION \
        libnccl2 \
        libnccl-dev \
        libjpeg-dev \
        libpng-dev \
        ssh \
        python$PYTHON_VERSION \
        python$PYTHON_VERSION-dev
    

# Get CNTK Binary Distribution
RUN CNTK_VERSION_DASHED=$(echo $CNTK_VERSION | tr . -) && \
    ([ "$CNTK_VERSION" != "2.4" ] || VERIFY_SHA256="true") && \
    CNTK_SHA256="8eebff81ef4111b2be5804303f1254cd20de5911a7678c8e64689e5c288dde40" && \
    wget -q https://cntk.ai/BinaryDrop/CNTK-${CNTK_VERSION_DASHED}-Linux-64bit-GPU.tar.gz && \
    ([ "$VERIFY_SHA256" != "true" ] || (echo "$CNTK_SHA256 CNTK-${CNTK_VERSION_DASHED}-Linux-64bit-GPU.tar.gz" | sha256sum --check --strict -)) && \
    tar -xzf CNTK-${CNTK_VERSION_DASHED}-Linux-64bit-GPU.tar.gz && \
    rm -f CNTK-${CNTK_VERSION_DASHED}-Linux-64bit-GPU.tar.gz && \
    /bin/bash /cntk/Scripts/install/linux/install-cntk.sh --py-version $PY_VERSION --docker --anaconda-basepath /opt/conda
    
ENV PATH /opt/conda/envs/py$PY_VERSION/bin:$PATH

# Set default NCCL parameters
RUN echo NCCL_DEBUG=INFO >> /etc/nccl.conf && \
    echo NCCL_SOCKET_IFNAME=^docker0 >> /etc/nccl.conf

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

WORKDIR /root


