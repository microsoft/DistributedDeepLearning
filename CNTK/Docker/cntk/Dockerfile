# Ubuntu 16.04, CUDA 9.0
FROM nvidia/cuda:9.0-runtime-ubuntu16.04

ARG CNTK_VERSION="2.5.1"
LABEL maintainer "MICROSOFT CORPORATION" \
    com.microsoft.cntk.version="$CNTK_VERSION"

ENV CNTK_VERSION="$CNTK_VERSION"

# Install CNTK as the default backend for Keras
ENV KERAS_BACKEND=cntk

RUN apt-get update && apt-get install -y --no-install-recommends \
    # General
    ca-certificates \
    wget \
    sudo \
    build-essential \
    openssh-client \
    openssh-server \
    && \
    # Clean-up
    apt-get -y autoremove \
    && \
    rm -rf /var/lib/apt/lists/*

# Get CNTK Binary Distribution
RUN CNTK_VERSION_DASHED=$(echo $CNTK_VERSION | tr . -) && \
    ([ "$CNTK_VERSION" != "2.4" ] || VERIFY_SHA256="true") && \
    CNTK_SHA256="8eebff81ef4111b2be5804303f1254cd20de5911a7678c8e64689e5c288dde40" && \
    wget -q https://cntk.ai/BinaryDrop/CNTK-${CNTK_VERSION_DASHED}-Linux-64bit-GPU.tar.gz && \
    ([ "$VERIFY_SHA256" != "true" ] || (echo "$CNTK_SHA256 CNTK-${CNTK_VERSION_DASHED}-Linux-64bit-GPU.tar.gz" | sha256sum --check --strict -)) && \
    tar -xzf CNTK-${CNTK_VERSION_DASHED}-Linux-64bit-GPU.tar.gz && \
    rm -f CNTK-${CNTK_VERSION_DASHED}-Linux-64bit-GPU.tar.gz && \
    /bin/bash /cntk/Scripts/install/linux/install-cntk.sh --py-version 35 --docker

WORKDIR /root
ENV PATH /cntk/cntk/bin:/root/anaconda3/envs/cntk-py35/bin:$PATH
ENV LD_LIBRARY_PATH /cntk/cntk/lib:/cntk/cntk/dependencies/lib:$LD_LIBRARY_PATH

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.gz && \
    tar zxf openmpi-3.0.0.tar.gz && \
    cd openmpi-3.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Create a wrapper for OpenMPI to allow running as root by default
RUN mv /usr/local/bin/mpirun /usr/local/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/local/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root "$@"' >> /usr/local/bin/mpirun && \
    chmod a+x /usr/local/bin/mpirun

# Configure OpenMPI to run good defaults:
#   --bind-to none --map-by slot --mca btl_tcp_if_exclude lo,docker0
RUN echo "hwloc_base_binding_policy = none" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "rmaps_base_mapping_policy = slot" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "btl_tcp_if_exclude = lo,docker0" >> /usr/local/etc/openmpi-mca-params.conf

# Set default NCCL parameters
RUN echo NCCL_DEBUG=INFO >> /etc/nccl.conf && \
    echo NCCL_SOCKET_IFNAME=^docker0 >> /etc/nccl.conf

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

