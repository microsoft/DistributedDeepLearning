""" This is an example template that you can use to create functions that you can call with invoke
"""
from invoke import task, Collection
import os
from config import load_config


_BASE_PATH = os.path.dirname(os.path.abspath( __file__ ))
env_values = load_config()


@task
def submit_local(c):
    """This command isn't implemented please modify to use.

    The call below will work for submitting jobs to execute locally on a GPU.
    """
    raise NotImplementedError(
        "You need to modify this call before being able to use it"
    )
    from aml_compute import TFExperimentCLI
    exp = TFExperimentCLI("<YOUR-EXPERIMENT-NAME>")
    run = exp.submit_local(
        os.path.join(_BASE_PATH, "src"),
        "<YOUR-TRAINING-SCRIPT>",
        {"YOUR": "ARGS"},
        dependencies_file=os.path.join(_BASE_PATH, "environment_gpu.yml"),
        wait_for_completion=True,
    )
    print(run)


@task
def submit_remote(c, node_count=int(env_values["CLUSTER_MAX_NODES"])):
    """This command isn't implemented please modify to use.

    The call below will work for submitting jobs to execute on a remote cluster using GPUs.
    """
    raise NotImplementedError(
        "You need to modify this call before being able to use it"
    )
    from aml_compute import TFExperimentCLI
    exp = TFExperimentCLI("<YOUR-EXPERIMENT-NAME>")
    run = exp.submit(
        os.path.join(_BASE_PATH, "src"),
        "<YOUR-TRAINING-SCRIPT>",
        {"YOUR": "ARGS"},
        node_count=node_count,
        dependencies_file=os.path.join(_BASE_PATH, "environment_gpu.yml"),
        wait_for_completion=True,
    )
    print(run)


@task
def submit_images_remote(c, node_count=int(env_values["CLUSTER_MAX_NODES"])):
    """This command isn't implemented please modify to use.

    The call below will work for submitting jobs to execute on a remote cluster using GPUs.
    Notive that we are passing in a {datastore} parameter to the path. This tells the submit
    method that we want the location as mapped by the datastore to be inserted here. Upon
    execution the appropriate path will be prepended to the training_data_path and validation_data_path.
    """
    raise NotImplementedError(
        "You need to modify this call before being able to use it"
    )
    from aml_compute import TFExperimentCLI
    exp = TFExperimentCLI("<YOUR-EXPERIMENT-NAME>")
    run = exp.submit(
        os.path.join(_BASE_PATH, "src"),
        "<YOUR-TRAINING-SCRIPT>",
        {
            "--training_data_path": "{datastore}/train",
            "--validation_data_path": "{datastore}/validation",
            "--epochs": "1",
            "--data_type": "images",
            "--data-format": "channels_first",
        },
        node_count=node_count,
        dependencies_file=os.path.join(_BASE_PATH, "environment_gpu.yml"),
        wait_for_completion=True,
    )
    print(run)


@task
def submit_images_local(c):
    """This command isn't implemented please modify to use.

    The call below will work for submitting jobs to execute locally on a GPU.
    Here we also map a volume to the docker container executing locally. This is the 
    location we tell our script to look for our training and validation data. Feel free to 
    adjust the other arguments as required by your trainining script.
    """
    raise NotImplementedError(
        "You need to modify this call before being able to use it"
    )
    from aml_compute import TFExperimentCLI
    exp = TFExperimentCLI("<YOUR-EXPERIMENT-NAME>")
    run = exp.submit_local(
        os.path.join(_BASE_PATH, "src"),
        "<YOUR-TRAINING-SCRIPT>",
        {
            "--training_data_path": "/data/train",
            "--validation_data_path": "/data/validation",
            "--epochs": "1",
            "--data_type": "images",
            "--data-format": "channels_first",
        },
        dependencies_file="TensorFlow_imagenet/environment_gpu.yml",
        docker_args=["-v", f"{env_values['data']}:/data"],
        wait_for_completion=True,
    )
    print(run)


remote_collection = Collection("remote")
remote_collection.add_task(submit_images_remote, "images")
remote_collection.add_task(submit_remote, "synthetic")

local_collection = Collection("local")
local_collection.add_task(submit_images_local, "images")
local_collection.add_task(submit_local, "synthetic")

submit_collection = Collection("submit", local_collection, remote_collection)
namespace = Collection("tf_experiment", submit_collection)
