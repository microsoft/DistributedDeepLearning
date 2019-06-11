from invoke import task, Collection
import os
from config import load_config


_BASE_PATH = os.path.dirname(os.path.abspath(__file__))
env_values = load_config()


def _benchmark_code_exists():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.exists(os.path.join(dir_path, "src", "tf_cnn_benchmarks.py"))


@task
def clone_benchmarks(c):
    """Clones the Tensorflow benchmarks from https://github.com/tensorflow/benchmarks.git into the src folder
    """
    if _benchmark_code_exists():
        return None
    c.run(
        "git clone -b cnn_tf_v1.12_compatible  https://github.com/tensorflow/benchmarks.git"
    )
    dir_path = os.path.dirname(os.path.realpath(__file__))
    c.run(
        f"cp -r benchmarks/scripts/tf_cnn_benchmarks/* {os.path.join(dir_path, 'src')}"
    )
    c.run("rm -r benchmarks")


@task(pre=[clone_benchmarks])
def submit_tf_benchmark(c, node_count=int(env_values["CLUSTER_MAX_NODES"])):
    """Submits TensorFlow benchmark job using synthetic data on remote cluster
    
    Args:
        node_count (int, optional): The number of nodes to use in cluster. Defaults to env_values['CLUSTER_MAX_NODES'].

    Note:
        Runs ResNet 50 model with batch size of 256 and mixed precision
    """
    from aml_compute import TFExperimentCLI

    exp = TFExperimentCLI("tf_benchmark")
    run = exp.submit(
        os.path.join(_BASE_PATH, "src"),
        "tf_cnn_benchmarks.py",
        {
            "--model": "resnet50",
            "--batch_size": 256,
            "--variable_update": "horovod",
            "--use_fp16": "",
        },
        node_count=node_count,
        dependencies_file=os.path.join(_BASE_PATH, "environment_gpu.yml"),
        wait_for_completion=True,
    )
    print(run)


@task(pre=[clone_benchmarks])
def submit_tf_benchmark_local(c):
    """Submits TensorFlow benchmark job using synthetic data for local execution

    Note:
        Runs ResNet 50 model with batch size of 256 and mixed precision
    """
    from aml_compute import TFExperimentCLI

    exp = TFExperimentCLI("tf_benchmark")
    run = exp.submit_local(
        os.path.join(_BASE_PATH, "src"),
        "tf_cnn_benchmarks.py",
        {
            "--model": "resnet50",
            "--batch_size": 256,
            "--variable_update": "horovod",
            "--use_fp16": "",
        },
        dependencies_file=os.path.join(_BASE_PATH, "environment_gpu.yml"),
        wait_for_completion=True,
    )
    print(run)


remote_collection = Collection("remote")
remote_collection.add_task(submit_tf_benchmark, "synthetic")
local_collection = Collection("local")
local_collection.add_task(submit_tf_benchmark_local, "synthetic")
submit_collection = Collection("submit", local_collection, remote_collection)
namespace = Collection("tf_benchmark", submit_collection)

