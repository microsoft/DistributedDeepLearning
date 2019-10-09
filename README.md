# Introduction 
This repo contains a cookiecutter template for running distributed training of deep learning models using Azure Machine Learning. You can create clusters with 0 nodes which will incur no cost and scale this up to hundreds of nodes. It is also possible to use low priority nodes to reduce costs even further.

The project contains the following:  
#### Tensorflow Benchmark 
This is a demo template that allows you to easily run [tf_cnn_benchmarks](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks) on Azure ML. This is a great way to test performance as well as compare to other platforms  
#### Tensorflow Imagenet 
This is another demo template that shows you how to train a ResNet50 model using Imagenet on Azure. We include scripts for processing the Imagenet data, transforming them to TF Records as well as leveraging AzCopy to quickly upload the data to the cloud.
#### Tensorflow Template 
This is a blank template you can use for your own distributed training projects. It allows you to leverage all the tooling built around the previous two demos to speed up the time it takes to run your model in a distributed fashion on Azure.  
#### PyTorch Benchmark 
This is a demo template that allows you to easily run a simple PyTorch benchmarking script on Azure ML. This is a great way to test performance as well as compare to other platforms
#### PyTorch Imagenet 
This is another demo template that shows you how to train a ResNet50 model using Imagenet on Azure. We include scripts for processing the Imagenet data as well as leveraging AzCopy to quickly upload the data to the cloud.  
#### PyTorch Template 
This is a blank template you can use for your own distributed training projects. It allows you to leverage all the tooling built around the previous two demos to speed up the time it takes to run your model in a distributed fashion on Azure.

# Prerequisites
Before you get started you need a PC running Ubuntu and the following installed:  
[Docker installed](https://docs.docker.com/install/linux/docker-ce/ubuntu/)  
[Nvidia runtime for docker](https://github.com/NVIDIA/nvidia-container-runtime) [Required for local execution]  
Python>=3.6  
[Cookiecutter installed](https://cookiecutter.readthedocs.io/en/latest/)  
[Git installed](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)  

> **Note:**
> You will need to run docker without sudo, to do this run:
> ```
> sudo usermod -aG docker $USER
> newgrp docker 
>```

# Setup
## Using the template

Once you have Cookiecutter installed you can either directly invoke project creation as follows:
```bash
cookiecutter gh:Microsoft/distributeddeeplearning
```
or clone locally and then invoke
```bash
git clone https://github.com/Microsoft/distributeddeeplearning.git
cookiecutter distributeddeeplearning
```
Cookiecutter will then ask you about a number of fields which it will use to construct your project. 
If you simply want to select the defaults don't write or select anything just press enter. Many of them can be left at the default values, the ones that are absolutely necessary are _highlighted_

**project_title:**          The title of your project  

**project_name:**           The folder in which your project will be created. Make sure it is a valid linux folder name  

**resource_group:**         The name of the resource group in Azure under which all the resources will be created. It is fine if it already exists  

**workspace:**              The AML workspace that the project will use. If it doesn't already exist it will create it

**sub_id:**                 The subscription id for your project, you can look this up on the portal or run a command on the cloud shell to get it. It isn't mandatory though, the application will give you an option to select it later.  

**vm_size:**                The VM type to use for distributed training

**minimum_number_nodes:**   The minimum number of nodes in the cluster. Set to 0 if you want it to scale down after use to reduce costs  
**maximum_number_nodes:**   The maximum number of nodes in the cluster

**cluster_name:**           The name of the cluster to use. It will create it if it doesn't exist  

**container_registry:**     The name of your dockerhub or other account which you may want to push your control plane docker container. If you don't have one or don't want to push the container to it simply leave as default

**type:**                   The type of project you want:
* all: All of them 
* template: Just create a template for distributed training
* benchmark: Create project that will run the Tensorflow benchmarks
* imagenet: Create an example project that will run against the imagenet data. (You will need to download the imagenet data)  
  
**region:**                 Which region to create Azure resources in  

**experiment_name:**        The name of the experiment  

**data:**                   The absolute path on your computer where you will store the imagenet data. The location needs to have around 400GB of space   

**image_name:**             The name to give the control plane docker image  

**datastore_name:**         Name of the datastore that will be created as part of the project  

**container_name:**         The name of the container in your storage account that will hold the data  

Once the project is created you will still be able to change many of the above options as they will be present in .env file that will be created. 

## Building environment
Distributed training is complex and often has a number of moving parts. To reduce the overhead of installing packages and managing environments we use a docker container to encapsulate our enviroment. So once you have created the project simply navigate to the root folder created by cookiecutter and run:
```bash
make build
```
This will build your docker container. Inside your docker container will be an appropriately set up conda environment a number of utilities such as AzCopy as well as everything you will need to run your distributed training job. 
Once your container is built run:
```bash
make run
```
This will put you in an environment inside your container in a tmux session (for a tutorial on tmux see [here](https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/)). The tmux control key has been mapped to **ctrl+a** rather than the standard ctrl+b so as not to interfere with outer tmux session if you are already a tmux user. You can alter this in the tmux.conf file in the Docker folder. The docker container will map the location you launched it from to the location /workspace inside the docker container. Therefore you can edit files outside of the container in the project folder and the changes will be reflected inside the container.

## Imagenet data
If you have selected **all**, **tensorflow_imagenet** or **pytorch_imagenet** in the type question during cookiecutter invocation then you will need to have **ILSVRC2012_img_train.tar** and **ILSVRC2012_img_val.tar** present in the direcotry you specified as your data directory. Go to the [download page](http://www.image-net.org/download-images) (you may need to register an account), and find the page for ILSVRC2012. You will need to download the two files mentioned earlier.

## Template selection
Based on the option you selected for **type** during the cookiecutter invocation you will get all or one of the options below. Cookiecutter will create your project folder which will contain the tempalte folders. When inside your project folder make sure you have run the **make build** and **make run** commands as mentioned in _building environment_ section above. Once you run the run command you will be greeted by a prompt, this is now your control plane. First you will need to set everything up. To do this run
```bash
inv setup 
```
It will ask you to log in so follow the prompts in the terminal. If you selected **all** in the template type it will also prepare the imagenet data.
Now you will be ready to run the templates.

#### Tensorflow Benchmark
This is a demo template allows you to easily run tf_cnn_benchmarks on Azure ML. This is a great way to test performance as well as compare to other platforms. To use this you must either select benchmark or all when invoking cookiecutter. 
Once setup is complete then simply run:
```bash
inv tf-benchmark.submit.local.synthetic
```
to run things locally on a single GPU. Note that the first time you run things you will have to build the environment.
To run things on a cluster simply run:
```bash
inv tf-benchmark.submit.remote.synthetic
```
Note that this will create the cluster if it wasn't created earlier and create the appropriate environment.

#### Tensorflow Imagenet
This is the second demo template that will train a ResNet50 model on imagenet. It allows the options of using synthetic data, image data as well as tfrecords. To use this you must either select **tensorflow_imagenet** or **all** when cookiecutter asks what type of project you want to create.
The run things locally using synthetic data simply run:
```
inv tf-imagenet.submit.local.synthetic
```

To run things on a remote cluster with real data in tfrecords format simply run:
```
inv tf-imagenet.submit.remote.tfrecords
```

This only covers a small number of commands, to see the full list of commands simply run inv --list.
#### Tensorflow Experiment
This is the option that you should use if you want to run your own training script. It is up to you to add the appropriate training scripts and modify the tensorflow_experiment.py file to run the appropriate commands. If you want to see how to invoke things simply look at the other examples.


#### Pytorch Benchmark
This is a demo template allows you to easily run a simple PyTorch benchmarking script on Azure ML. To use this you must either select benchmark or all when invoking cookiecutter. 
Once setup is complete then simply run:
```bash
inv pytorch-benchmark.submit.local.synthetic
```
to run things locally on a single GPU. Note that the first time you run things you will have to build the environment.
To run things on a cluster simply run:
```bash
inv pytorch-benchmark.submit.remote.synthetic
```
Note that this will create the cluster if it wasn't created earlier and create the appropriate environment.

#### PyTorch Imagenet
This is the second demo template that will train a ResNet50 model on imagenet. It allows the options of using synthetic data or image data. To use this you must either select **pytorch_imagenet** or **all** when cookiecutter asks what type of project you want to create.
The run things locally using synthetic data simply run:
```
inv pytorch-imagenet.submit.local.synthetic
```

To run things on a remote cluster with real data in tfrecords format simply run:
```
inv pytorch-imagenet.submit.remote.tfrecords
```
#### Pytorch Experiment
This is the option that you should use if you want to run your own training script. It is up to you to add the appropriate training scripts and modify the pytorch_experiment.py file to run the appropriate commands. If you want to see how to invoke things simply look at the other examples.


# Architecture
Below is a diagram that shows how the project is set up.

<p align="center">
  <img width="1000" src="./images/architecture1.png">
</p>

The docker container you created using **make build** is the control plane and from there we can invoke jobs to execute either locally or in the cloud. Local execution is meant for debugging and will run on a single GPU. The mapping of data locations is handled by the control scripts. During local execution the appropriate location is mapped to the container. During remote execution the data store created during setup will be mounted on to each of the VMs in the cluster.

## Project structure
The original project structure is as shown below.

```.
├── cookiecutter.json  <-- Cookiecutter json that holds all the variables for the projects  
├── hooks  
│  ├── post_gen_project.py  
│  └── pre_gen_project.py  
├── images  
│  └── demo.svg  
├── LICENSE  
├── README.md <-- This readme  
└── {{cookiecutter.project_name}}  
   ├── _dotenv_template <-- Template that is read and translated into .env file  
   ├── control <-- Holds all files for the control plane  
   │  ├── Docker <-- Contains the files used to build the control plane docker container
   │  │  ├── azure_requirements.txt <-- Azure python requirements
   │  │  ├── bash.completion <-- Completion script for invoke
   │  │  ├── dockerfile
   │  │  ├── environment.yml <-- Conda environment specification for control plane
   │  │  ├── jupyter_notebook_config.py 
   │  │  └── tmux.conf <-- Tmux configuration
   │  └── src
   │     ├── aml_compute.py <-- Module that holds methods for creating cluster and submitting experiments using Azure ML
   │     ├── config.py <-- Module for loading and working with .env config
   │     └── logging.conf <-- Logging configuration for control plane
   ├── Makefile <-- Makefile to build and run control plane
   ├── scripts
   │  ├── convert_imagenet_to_tf_records.py <-- Script for transforming imagenet data to tf records
   │  ├── image.py <-- Invoke module for working with images
   │  ├── imagenet_nounid_to_class.json <-- Imagenet nounid lookup
   │  ├── prepare_imagenet.py <-- Script for preparing imagenet data
   │  ├── storage.py <-- Invoke module for using Azure storage
   │  └── tfrecords.py <-- Invoke module for working with tf records
   ├── tasks.py <-- Main invoke module
   ├── PyTorch_benchmark<-- Template for running PyTorch benchmarks
   │  ├── environment_cpu.yml
   │  ├── environment_gpu.yml<-- Conda specification file used by Azure ML to create environment to run project in
   │  ├── pytorch_benchmark.py<-- Invoke module for running benchmarks
   │  └── src
   │     └── pytorch_synthetic_benchmark.py
   ├── PyTorch_imagenet
   │  ├── environment_cpu.yml
   │  ├── environment_gpu.yml<-- Conda specification file used by Azure ML to create environment to run project in
   │  ├── pytorch_imagenet.py<-- Invoke module for running benchmarks
   │  └── src
   │     ├── imagenet_pytorch_horovod.py
   │     ├── logging.conf
   │     └── timer.py
   ├── PyTorch_experiment<-- PyTorch distributed training template [Put your code here]
   │  ├── environment_cpu.yml
   │  ├── environment_gpu.yml<-- Conda specification file used by Azure ML to create environment to run project in
   │  ├── pytorch_experiment.py<-- Invoke module for running benchmarks
   │  └── src
   │     └── train_model.py
   ├── TensorFlow_benchmark <-- Template for running Tensorflow benchmarks
   │  ├── environment_cpu.yml 
   │  ├── environment_gpu.yml <-- Conda specification file used by Azure ML to create environment to run project in
   │  ├── src <-- Folder where tensorflow benchmarks code will be cloned into
   |  └── tensorflow_benchmark.py <-- Invoke module for running benchmarks
   ├── TensorFlow_experiment <-- Tensorflow distributed training template [Put your code here]
   │  ├── environment_cpu.yml
   │  ├── environment_gpu.yml <-- Conda specification file used by Azure ML to create environment to run project in
   │  ├── src
   │  │  ├── logging.conf
   │  │  └── train_model.py <-- Template file
   │  └── tensorflow_experiment.py <-- Invoke module for running template
   └── TensorFlow_imagenet
      ├── environment_cpu.yml
      ├── environment_gpu.yml <-- Conda specification file used by Azure ML to create environment to run project in
      ├── src <-- Code for training ResNet50 model on imagenet
      │  ├── data
      │  │  ├── __init__.py
      │  │  ├── images.py
      │  │  ├── synthetic.py
      │  │  └── tfrecords.py
      │  ├── defaults.py
      │  ├── imagenet_preprocessing.py
      │  ├── logging.conf
      │  ├── resnet_main.py <-- Main entry script
      │  ├── resnet_model.py
      │  ├── resnet_run_loop.py
      │  ├── timer.py
      │  └── utils.py
      └── tensorflow_imagenet.py <-- Invoke module for running imagenet experiment
```
Depending on the options chosen only certain branches will be moved over to your project.


## Options
These are the options when using the template. These can differ depenting on the type of project you choose to create. To see this list youself simply run:
```
inv --list
```
```
  delete                                     Delete the resource group and all associated resources
  experiments                                Prints list of experiments
  interactive (i)                            Open IPython terminal and load in modules to work with AzureML
  login                                      Log in to Azure CLI
  runs                                       Prints information on last N runs in specified experiment
  select-subscription                        Select Azure subscription to use
  setup                                      Setup the environment and process the imagenet data
  tensorboard                                Runs tensorboard in a seperate tmux session
  pytorch-benchmark.submit.local.synthetic    Submit PyTorch training job using synthetic data for local execution
  pytorch-benchmark.submit.remote.synthetic   Submit PyTorch training job using synthetic data to remote cluster
  pytorch-imagenet.submit.local.images        Submit PyTorch training job using real imagenet data for local execution
  pytorch-imagenet.submit.local.synthetic     Submit PyTorch training job using synthetic imagenet data for local execution
  pytorch-imagenet.submit.remote.images       Submit PyTorch training job using real imagenet data to remote cluster
  pytorch-imagenet.submit.remote.synthetic    Submit PyTorch training job using synthetic imagenet data to remote cluster
  storage.create-resource-group
  storage.store-key                          Retrieves premium storage account key from Azure and stores it in .env file
  storage.image.create-container             Creates container based on the parameters found in the .env file
  storage.image.download-data                Download training and validation data from blob container specified in .env file
  storage.image.download-training            Download training data from blob container specified in .env file
  storage.image.download-validation          Download validation data from blob container specified in .env file
  storage.image.prepare-imagenet             Prepare imagenet data found in download_dir and push results to target_dir
  storage.image.upload-data                  Upload training and validation data to container specified in .env file
  storage.image.upload-training-data         Upload training data to container specified in .env file
  storage.image.upload-validation-data       Upload validation data to container specified in .env file
  storage.create-container                   Creates container based on the parameters found in the .env file
  storage.create-premium-storage             Creates premium storage account. By default the values are loaded from the local .env file
  storage.tfrecords.upload-validation-data   Upload tfrecords validation data to container specified in .env file
  tf-benchmark.submit.local.synthetic        Submits TensorFlow benchmark job using synthetic data for local execution
  tf-benchmark.submit.remote.synthetic       Submits TensorFlow benchmark job using synthetic data on remote cluster
  tf-experiment.submit.local.images          This command isn't implemented please modify to use.
  tf-experiment.submit.local.synthetic       This command isn't implemented please modify to use.
  tf-experiment.submit.remote.images         This command isn't implemented please modify to use.
  tf-experiment.submit.remote.synthetic      This command isn't implemented please modify to use.
  tf-imagenet.submit.local.images            Submit TensorFlow training job using real imagenet data for local execution
  tf-imagenet.submit.local.synthetic         Submit TensorFlow training job using synthetic imagenet data for local execution
  tf-imagenet.submit.local.tfrecords         Submit TensorFlow training job using real imagenet data as tfrecords for local execution
  tf-imagenet.submit.remote.images           Submit TensorFlow training job using real imagenet data to remote cluster
  tf-imagenet.submit.remote.synthetic        Submit TensorFlow training job using synthetic imagenet data to remote cluster
  tf-imagenet.submit.remote.tfrecords        Submit TensorFlow training job using real imagenet data as tfrecords to remote cluster
```

# Contributing
This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
