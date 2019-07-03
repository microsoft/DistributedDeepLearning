"""Module for running PyTorch training on Imagenet data
"""
from invoke import task, Collection
import os
from config import load_config


_BASE_PATH = os.path.dirname(os.path.abspath(__file__))
env_values = load_config()



@task
def submit_synthetic(c, node_count=int(env_values["CLUSTER_MAX_NODES"]), epochs=1):
    """Submit PyTorch training job using synthetic imagenet data to remote cluster
    
    Args:
        node_count (int, optional): The number of nodes to use in cluster. Defaults to env_values['CLUSTER_MAX_NODES'].
        epochs (int, optional): Number of epochs to run training for. Defaults to 1.
    """
    from aml_compute import PyTorchExperimentCLI

    exp = PyTorchExperimentCLI("pytorch_synthetic_images_remote")
    run = exp.submit(
        os.path.join(_BASE_PATH, "src"),
        "imagenet_pytorch_horovod.py",
        {"--epochs": epochs, "--use_gpu":True},
        node_count=node_count,
        dependencies_file=os.path.join(_BASE_PATH, "environment_gpu.yml"),
        wait_for_completion=True,
    )
    print(run)


@task
def submit_synthetic_local(c, epochs=1):
    """Submit PyTorch training job using synthetic imagenet data for local execution
    
    Args:
        epochs (int, optional): Number of epochs to run training for. Defaults to 1.
    """
    from aml_compute import PyTorchExperimentCLI

    exp = PyTorchExperimentCLI("pytorch_synthetic_images_local")
    run = exp.submit_local(
        os.path.join(_BASE_PATH, "src"),
        "imagenet_pytorch_horovod.py",
        {"--epochs": epochs, "--use_gpu":True},
        dependencies_file=os.path.join(_BASE_PATH, "environment_gpu.yml"),
        wait_for_completion=True,
    )
    print(run)


@task
def submit_images(c, node_count=int(env_values["CLUSTER_MAX_NODES"]), epochs=1):
    """Submit PyTorch training job using real imagenet data to remote cluster
    
    Args:
        node_count (int, optional): The number of nodes to use in cluster. Defaults to env_values['CLUSTER_MAX_NODES'].
        epochs (int, optional): Number of epochs to run training for. Defaults to 1.
    """
    from aml_compute import PyTorchExperimentCLI

    exp = PyTorchExperimentCLI("pytorch_real_images_remote")
    run = exp.submit(
        os.path.join(_BASE_PATH, "src"),
         "imagenet_pytorch_horovod.py",
        {
            "--use_gpu":True, 
            "--epochs":epochs,
            "--training_data_path": "{datastore}/train",
            "--validation_data_path": "{datastore}/validation",
        },
        node_count=node_count,
        dependencies_file=os.path.join(_BASE_PATH, "environment_gpu.yml"),
        wait_for_completion=True,
    )
    print(run)


@task
def submit_images_local(c, epochs=1):
    """Submit PyTorch training job using real imagenet data for local execution
    
    Args:
        epochs (int, optional): Number of epochs to run training for. Defaults to 1.
    """
    from aml_compute import PyTorchExperimentCLI

    exp = PyTorchExperimentCLI("pytorch_real_images_local")
    run = exp.submit_local(
        os.path.join(_BASE_PATH, "src"),
          "imagenet_pytorch_horovod.py",
        {
            "--epochs": epochs, 
            "--use_gpu":True, 
            "--training_data_path": "/data/train",
            "--validation_data_path": "/data/validation",
        },
        dependencies_file=os.path.join(_BASE_PATH, "environment_gpu.yml"),
        docker_args=["-v", f"{env_values['DATA']}:/data"],
        wait_for_completion=True,
    )
    print(run)



remote_collection = Collection("remote")
remote_collection.add_task(submit_images, "images")
remote_collection.add_task(submit_synthetic, "synthetic")

local_collection = Collection("local")
local_collection.add_task(submit_images_local, "images")
local_collection.add_task(submit_synthetic_local, "synthetic")

submit_collection = Collection("submit", local_collection, remote_collection)
namespace = Collection("pytorch_imagenet", submit_collection)

