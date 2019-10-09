"""Module for running PyTorch benchmark using synthetic data
"""
from invoke import task, Collection
import os
from config import load_config


_BASE_PATH = os.path.dirname(os.path.abspath(__file__))
env_values = load_config()


@task
def submit_benchmark_remote(c, node_count=int(env_values["CLUSTER_MAX_NODES"])):
    """Submit PyTorch training job using synthetic data to remote cluster
    
    Args:
        node_count (int, optional): The number of nodes to use in cluster. Defaults to env_values['CLUSTER_MAX_NODES'].
    """
    from aml_compute import PyTorchExperimentCLI

    exp = PyTorchExperimentCLI("synthetic_benchmark_remote")
    run = exp.submit(
        os.path.join(_BASE_PATH, "src"),
        "pytorch_synthetic_benchmark.py",
        {"--model": "resnet50", "--batch-size": 64},
        node_count=node_count,
        dependencies_file=os.path.join(_BASE_PATH, "environment_gpu.yml"),
        wait_for_completion=True,
    )
    print(run)


@task
def submit_benchmark_local(c):
    """Submit PyTorch training job using synthetic data for local execution
    
    """
    from aml_compute import TFExperimentCLI

    exp = TFExperimentCLI("synthetic_images_local")
    run = exp.submit_local(
        os.path.join(_BASE_PATH, "src"),
        "pytorch_synthetic_benchmark.py",
        {"--model": "resnet50", "--batch-size": 64},
        dependencies_file=os.path.join(_BASE_PATH, "environment_gpu.yml"),
        wait_for_completion=True,
    )
    print(run)


remote_collection = Collection("remote")
remote_collection.add_task(submit_benchmark_remote, "synthetic")

local_collection = Collection("local")
local_collection.add_task(submit_benchmark_local, "synthetic")

submit_collection = Collection("submit", local_collection, remote_collection)
namespace = Collection("pytorch_benchmark", submit_collection)

