"""Module for running TensorFlow training on Imagenet data
"""
from invoke import task, Collection
import os
from config import load_config


_BASE_PATH = os.path.dirname(os.path.abspath(__file__))
env_values = load_config()


@task
def submit_synthetic(c, node_count=int(env_values["CLUSTER_MAX_NODES"]), epochs=1):
    """Submit TensorFlow training job using synthetic imagenet data to remote cluster
    
    Args:
        node_count (int, optional): The number of nodes to use in cluster. Defaults to env_values['CLUSTER_MAX_NODES'].
        epochs (int, optional): Number of epochs to run training for. Defaults to 1.
    """
    from aml_compute import TFExperimentCLI

    exp = TFExperimentCLI("synthetic_images_remote")
    run = exp.submit(
        os.path.join(_BASE_PATH, "src"),
        "resnet_main.py",
        {"--epochs": epochs},
        node_count=node_count,
        dependencies_file=os.path.join(_BASE_PATH, "environment_gpu.yml"),
        wait_for_completion=True,
    )
    print(run)


@task
def submit_synthetic_local(c, epochs=1):
    """Submit TensorFlow training job using synthetic imagenet data for local execution
    
    Args:
        epochs (int, optional): Number of epochs to run training for. Defaults to 1.
    """
    from aml_compute import TFExperimentCLI

    exp = TFExperimentCLI("synthetic_images_local")
    run = exp.submit_local(
        os.path.join(_BASE_PATH, "src"),
        "resnet_main.py",
        {"--epochs": epochs},
        dependencies_file=os.path.join(_BASE_PATH, "environment_gpu.yml"),
        wait_for_completion=True,
    )
    print(run)


@task
def submit_images(c, node_count=int(env_values["CLUSTER_MAX_NODES"]), epochs=1):
    """Submit TensorFlow training job using real imagenet data to remote cluster
    
    Args:
        node_count (int, optional): The number of nodes to use in cluster. Defaults to env_values['CLUSTER_MAX_NODES'].
        epochs (int, optional): Number of epochs to run training for. Defaults to 1.
    """
    from aml_compute import TFExperimentCLI

    exp = TFExperimentCLI("real_images_remote")
    run = exp.submit(
        os.path.join(_BASE_PATH, "src"),
        "resnet_main.py",
        {
            "--training_data_path": "{datastore}/train",
            "--validation_data_path": "{datastore}/validation",
            "--epochs": epochs,
            "--data_type": "images",
            "--data-format": "channels_first",
        },
        node_count=node_count,
        dependencies_file=os.path.join(_BASE_PATH, "environment_gpu.yml"),
        wait_for_completion=True,
    )
    print(run)


@task
def submit_images_local(c, epochs=1):
    """Submit TensorFlow training job using real imagenet data for local execution
    
    Args:
        epochs (int, optional): Number of epochs to run training for. Defaults to 1.
    """
    from aml_compute import TFExperimentCLI

    exp = TFExperimentCLI("real_images_local")
    run = exp.submit_local(
        os.path.join(_BASE_PATH, "src"),
        "resnet_main.py",
        {
            "--training_data_path": "/data/train",
            "--validation_data_path": "/data/validation",
            "--epochs": epochs,
            "--data_type": "images",
            "--data-format": "channels_first",
        },
        dependencies_file=os.path.join(_BASE_PATH, "environment_gpu.yml"),
        docker_args=["-v", f"{env_values['DATA']}:/data"],
        wait_for_completion=True,
    )
    print(run)


@task
def submit_tfrecords(c, node_count=int(env_values["CLUSTER_MAX_NODES"]), epochs=1):
    """Submit TensorFlow training job using real imagenet data as tfrecords to remote cluster
    
    Args:
        node_count (int, optional): The number of nodes to use in cluster. Defaults to env_values['CLUSTER_MAX_NODES'].
        epochs (int, optional): Number of epochs to run training for. Defaults to 1.
    """
    from aml_compute import TFExperimentCLI

    exp = TFExperimentCLI("real_tfrecords_remote")
    run = exp.submit(
        os.path.join(_BASE_PATH, "src"),
        "resnet_main.py",
        {
            "--training_data_path": "{datastore}/tfrecords/train",
            "--validation_data_path": "{datastore}/tfrecords/validation",
            "--epochs": epochs,
            "--data_type": "tfrecords",
            "--data-format": "channels_first",
        },
        node_count=node_count,
        dependencies_file=os.path.join(_BASE_PATH, "environment_gpu.yml"),
        wait_for_completion=True,
    )
    print(run)


@task
def submit_tfrecords_local(c, epochs=1):
    """Submit TensorFlow training job using real imagenet data as tfrecords for local execution
    
    Args:
        epochs (int, optional): Number of epochs to run training for. Defaults to 1.
    """
    from aml_compute import TFExperimentCLI

    exp = TFExperimentCLI("real_tfrecords_local")
    run = exp.submit_local(
        os.path.join(_BASE_PATH, "src"),
        "resnet_main.py",
        {
            "--training_data_path": "/data/tfrecords/train",
            "--validation_data_path": "/data/tfrecords/validation",
            "--epochs": epochs,
            "--data_type": "tfrecords",
            "--data-format": "channels_first",
        },
        dependencies_file=os.path.join(_BASE_PATH, "environment_gpu.yml"),
        docker_args=["-v", f"{env_values['DATA']}:/data"],
        wait_for_completion=True,
    )
    print(run)


remote_collection = Collection("remote")
remote_collection.add_task(submit_images, "images")
remote_collection.add_task(submit_tfrecords, "tfrecords")
remote_collection.add_task(submit_synthetic, "synthetic")

local_collection = Collection("local")
local_collection.add_task(submit_images_local, "images")
local_collection.add_task(submit_tfrecords_local, "tfrecords")
local_collection.add_task(submit_synthetic_local, "synthetic")

submit_collection = Collection("submit", local_collection, remote_collection)
namespace = Collection("tf_imagenet", submit_collection)

