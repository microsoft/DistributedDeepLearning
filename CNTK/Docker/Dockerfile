# Ubuntu 16.04, CUDA 9.0
FROM nvidia/cuda:9.0-devel-ubuntu16.04

ENV CNTK_VERSION=2.5.1
ENV CUDNN_VERSION=7.0.5.15-1+cuda9.0
ENV PYTHON_VERSION=3.5
ENV NCCL_VERSION=2.1.15-1+cuda9.0

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
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
        python$PYTHON_VERSION \
        python$PYTHON_VERSION-dev
    
RUN ln -s /usr/bin/python$PYTHON_VERSION /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install CNTK
RUN pip install --no-cache-dir https://cntk.ai/PythonWheel/GPU/cntk-$CNTK_VERSION-cp35-cp35m-linux_x86_64.whl h5py scipy jupyter ipykernel numpy toolz pandas scikit-learn


# Set default NCCL parameters
RUN echo NCCL_DEBUG=INFO >> /etc/nccl.conf && \
    echo NCCL_SOCKET_IFNAME=^docker0 >> /etc/nccl.conf

# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

WORKDIR /root
