FROM ubuntu:16.04

COPY environment.yml .

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        wget \
        curl \
        gfortran \
        apt-transport-https \
        jq \
        locales \
        git && \
     	rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

ENV ENV_NAME=py3.6
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda env create -q --name $ENV_NAME -f environment.yml && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/envs/$ENV_NAME/bin:/opt/conda/bin:$PATH


# Install Azure CLI
RUN echo "deb [arch=amd64] https://packages.microsoft.com/repos/azure-cli/ xenial main" | \
    tee /etc/apt/sources.list.d/azure-cli.list && \
    curl -L https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    azure-cli
