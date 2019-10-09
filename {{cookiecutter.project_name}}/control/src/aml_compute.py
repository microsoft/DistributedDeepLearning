import logging
import logging.config
import os

import azureml.core
import fire
from amltoolz import Workspace
from azureml import core
from azureml.core import Datastore
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.conda_dependencies import (
    CondaDependencies,
    TENSORFLOW_DEFAULT_VERSION,
)
from azureml.core.runconfig import EnvironmentDefinition
from azureml.tensorboard import Tensorboard
from azureml.train.dnn import TensorFlow, PyTorch
from config import load_config
from toolz import curry, pipe
from pprint import pformat
from time import sleep


logging.config.fileConfig(os.getenv("LOG_CONFIG", "logging.conf"))

config_dict = load_config()

_DEFAULT_AML_PATH = config_dict.get("DEFAULT_AML_PATH", "aml_config/azml_config.json")
_CLUSTER_NAME = config_dict.get("CLUSTER_NAME", "gpucluster24rv3")
_CLUSTER_VM_SIZE = config_dict.get("CLUSTER_VM_SIZE", "Standard_NC24rs_v3")
_CLUSTER_MIN_NODES = int(config_dict.get("CLUSTER_MIN_NODES", 0))
_CLUSTER_MAX_NODES = int(config_dict.get("CLUSTER_MAX_NODES", 2))
_WORKSPACE = config_dict.get("WORKSPACE", "workspace")
_RESOURCE_GROUP = config_dict.get("RESOURCE_GROUP", "amlccrg")
_SUBSCRIPTION_ID = config_dict.get("SUBSCRIPTION_ID", None)
_REGION = config_dict.get("REGION", "eastus")
_DEPENDENCIES_FILE = config_dict.get(
    "DEPENDENCIES_FILE", "../../experiment/src/environment_gpu.yml"
)
_DATASTORE_NAME = config_dict.get("DATASTORE_NAME", "datastore")
_CONTAINER_NAME = config_dict.get("CONTAINER_NAME", "container")
_ACCOUNT_NAME = config_dict.get("ACCOUNT_NAME", None)
_ACCOUNT_KEY = config_dict.get("ACCOUNT_KEY", None)


def _create_cluster(
    workspace,
    cluster_name=_CLUSTER_NAME,
    vm_size=_CLUSTER_VM_SIZE,
    min_nodes=_CLUSTER_MIN_NODES,
    max_nodes=_CLUSTER_MAX_NODES,
):
    logger = logging.getLogger(__name__)
    try:
        compute_target = ComputeTarget(workspace=workspace, name=cluster_name)
        logger.info("Found existing compute target.")
    except ComputeTargetException:
        logger.info("Creating a new compute target...")
        compute_config = AmlCompute.provisioning_configuration(
            vm_size=vm_size, min_nodes=min_nodes, max_nodes=max_nodes
        )

        # create the cluster
        compute_target = ComputeTarget.create(workspace, cluster_name, compute_config)
        compute_target.wait_for_completion(show_output=True)

    # use get_status() to get a detailed status for the current AmlCompute.
    logger.debug(compute_target.get_status().serialize())

    return compute_target


def _prepare_environment_definition(base_image, dependencies_file, distributed):
    logger = logging.getLogger(__name__)
    env_def = EnvironmentDefinition()
    conda_dep = CondaDependencies(conda_dependencies_file_path=dependencies_file)
    env_def.python.user_managed_dependencies = False
    env_def.python.conda_dependencies = conda_dep
    env_def.docker.enabled = True
    env_def.docker.gpu_support = True
    env_def.docker.base_image = base_image
    env_def.docker.shm_size = "8g"
    env_def.environment_variables["NCCL_SOCKET_IFNAME"] = "eth0"
    env_def.environment_variables["NCCL_IB_DISABLE"] = 1

    if distributed:
        env_def.environment_variables["DISTRIBUTED"] = "True"
    else:
        env_def.environment_variables["DISTRIBUTED"] = "False"
        logger.info("Adding runtime argument")
        # Adds runtime argument since we aliased nvidia-docker to docker in order to be able to run them as
        # sibling containers. Without this we will get CUDA library errors
        env_def.docker.arguments.extend(["--runtime", "nvidia"])

    return env_def


@curry
def _create_estimator(
    estimator_class,
    dependencies_file,
    project_folder,
    entry_script,
    compute_target,
    script_params,
    base_image,
    node_count=_CLUSTER_MAX_NODES,
    process_count_per_node=4,
    docker_args=(),
):
    logger = logging.getLogger(__name__)
    logger.debug(f"Base image {base_image}")
    logger.debug(f"Loading dependencies from {dependencies_file}")

    # If the compute target is "local" then don't run distributed
    distributed = not (isinstance(compute_target, str) and compute_target == "local")
    env_def = _prepare_environment_definition(base_image, dependencies_file, distributed)
    env_def.docker.arguments.extend(list(docker_args))

    estimator = estimator_class(
        project_folder,
        entry_script=entry_script,
        compute_target=compute_target,
        script_params=script_params,
        node_count=node_count,
        process_count_per_node=process_count_per_node,
        distributed_backend="mpi" if distributed else None,
        environment_definition=env_def,
    )

    logger.debug(estimator.conda_dependencies.__dict__)
    return estimator


def _create_datastore(
    aml_workspace,
    datastore_name,
    container_name,
    account_name,
    account_key,
    create_if_not_exists=True,
):
    ds = Datastore.register_azure_blob_container(
        workspace=aml_workspace,
        datastore_name=datastore_name,
        container_name=container_name,
        account_name=account_name,
        account_key=account_key,
        create_if_not_exists=create_if_not_exists,
    )
    return ds


class ExperimentCLI(object):
    def __init__(
        self,
        experiment_name,
        workspace_name=_WORKSPACE,
        resource_group=_RESOURCE_GROUP,
        subscription_id=_SUBSCRIPTION_ID,
        workspace_region=_REGION,
        config_path=_DEFAULT_AML_PATH,
    ):

        self._logger = logging.getLogger(__name__)
        self._logger.info("SDK version:" + str(azureml.core.VERSION))
        self._ws = workspace_for_user(
            workspace_name=workspace_name,
            resource_group=resource_group,
            subscription_id=subscription_id,
            workspace_region=workspace_region,
            config_path=config_path,
        ).aml_workspace
        self._experiment = core.Experiment(self._ws, name=experiment_name)
        self._cluster = None
        self._datastore = None

    def create_cluster(
        self,
        name=_CLUSTER_NAME,
        vm_size=_CLUSTER_VM_SIZE,
        min_nodes=_CLUSTER_MIN_NODES,
        max_nodes=_CLUSTER_MAX_NODES,
    ):
        """Creates AzureML cluster
        
        Args:
            name (string, optional): The name you wish to assign the cluster. 
                                     Defaults to _CLUSTER_NAME.
            vm_size (string, optional): The type of sku to use for your vm. 
                                        Defaults to _CLUSTER_VM_SIZE.
            min_nodes (int, optional): Minimum number of nodes in cluster. 
                                       Use 0 if you don't want to incur costs when it isn't being used. 
                                       Defaults to _CLUSTER_MIN_NODES.
            max_nodes (int, optional): Maximum number of nodes in cluster. 
                                       Defaults to _CLUSTER_MAX_NODES.
        
        Returns:
            ExperimentCLI: Experiment object
        """
        self._cluster = _create_cluster(
            self._ws,
            cluster_name=name,
            vm_size=vm_size,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
        )
        return self

    def create_datastore(
        self,
        datastore_name=_DATASTORE_NAME,
        container_name=_CONTAINER_NAME,
        account_name=_ACCOUNT_NAME,
        account_key=_ACCOUNT_KEY,
    ):
        """Creates datastore
        
        Args:
            datastore_name (string, optional): Name you wish to assign to your datastore. Defaults to _DATASTORE_NAME.
            container_name (string, optional): Name of your container. Defaults to _CONTAINER_NAME.
            account_name (string, optional): Storage account name. Defaults to _ACCOUNT_NAME.
            account_key (string, optional): The storage account key. Defaults to _ACCOUNT_KEY.
        
        Returns:
            ExperimentCLI: Experiment object
        """
        assert account_name is not None, "Account name for Datastore not set"
        assert account_key is not None, "Account key for Datastore not set"

        self._datastore = _create_datastore(
            self._ws,
            datastore_name=datastore_name,
            container_name=container_name,
            account_name=account_name,
            account_key=account_key,
        )
        return self

    @property
    def cluster(self):
        if self._cluster is None:
            self.create_cluster()
        return self._cluster

    @property
    def datastore(self):
        if self._datastore is None:
            self.create_datastore()
        return self._datastore


def _has_key(input_dict, key):
    for v in input_dict.values:
        if key in v:
            return True
    return False


def _fill_param_with(input_dict, parameters_dict):
    return {key: value.format(**parameters_dict) for key, value in input_dict.items()}


class TFExperimentCLI(ExperimentCLI):
    """Creates Experiment object that can be used to create clusters and submit experiments
    
    Returns:
        TFExperimentCLI: Experiment object
    """

    def submit_local(
        self,
        project_folder,
        entry_script,
        script_params,
        dependencies_file=_DEPENDENCIES_FILE,
        wait_for_completion=True,
        docker_args=(),
    ):
        """Submit experiment for local execution
        
        Args:
            project_folder (string): Path of you source files for the experiment
            entry_script (string): The filename of your script to run. Must be found in your project_folder
            script_params (dict): Dictionary of script parameters
            dependencies_file (string, optional): The location of your environment.yml to use to create the
                                                  environment your training script requires. 
                                                  Defaults to _DEPENDENCIES_FILE.
            wait_for_completion (bool, optional): Whether to block until experiment is done. Defaults to True.
            docker_args (tuple, optional): Docker arguments to pass. Defaults to ().
        """
        self._logger.info("Running in local mode")
        self._submit(
            dependencies_file,
            project_folder,
            entry_script,
            "local",
            script_params,
            1,
            1,
            docker_args,
            wait_for_completion,
        )

    def submit(
        self,
        project_folder,
        entry_script,
        script_params,
        dependencies_file=_DEPENDENCIES_FILE,
        node_count=_CLUSTER_MAX_NODES,
        process_count_per_node=4,
        wait_for_completion=True,
        docker_args=(),
    ):
        """Submit experiment for remote execution on AzureML clusters
        
        Args:
            project_folder (string): Path of you source files for the experiment
            entry_script (string): The filename of your script to run. Must be found in your project_folder
            script_params (dict): Dictionary of script parameters
            dependencies_file (string, optional): The location of your environment.yml to use to
                                                  create the environment your training script requires. 
                                                  Defaults to _DEPENDENCIES_FILE.
            node_count (int, optional): [description]. Defaults to _CLUSTER_MAX_NODES.
            process_count_per_node (int, optional): Number of precesses to run on each node. 
                                                    Usually should be the same as the number of GPU for GPU exeuction. 
                                                    Defaults to 4.
            wait_for_completion (bool, optional): Whether to block until experiment is done. Defaults to True.
            docker_args (tuple, optional): Docker arguments to pass. Defaults to ().
        
        Returns:
            azureml.core.Run: AzureML Run object
        """
        self._logger.debug(script_params)

        transformed_params = self._complete_datastore(script_params)
        self._logger.debug("Transformed script params")
        self._logger.debug(transformed_params)

        return self._submit(
            dependencies_file,
            project_folder,
            entry_script,
            self.cluster,
            transformed_params,
            node_count,
            process_count_per_node,
            docker_args,
            wait_for_completion,
        )

    def _submit(
        self,
        dependencies_file,
        project_folder,
        entry_script,
        cluster,
        script_params,
        node_count,
        process_count_per_node,
        docker_args,
        wait_for_completion,
    ):
        self._logger.debug(script_params)
        base_image = "mcr.microsoft.com/azureml/base-gpu:intelmpi2018.3-cuda9.0-cudnn7-ubuntu16.04"

        estimator = _create_estimator(
            TensorFlow,
            dependencies_file,
            project_folder,
            entry_script,
            cluster,
            script_params,
            base_image,
            node_count=node_count,
            process_count_per_node=process_count_per_node,
            docker_args=docker_args
        )
        # TEMPORARY HACK: Bugs with AML necessitate the code below, once fixed remove
        estimator.conda_dependencies.remove_pip_package("horovod==0.15.2")
        estimator.conda_dependencies.remove_pip_package(
            "tensorflow==" + TENSORFLOW_DEFAULT_VERSION
        )
        estimator.conda_dependencies.add_pip_package("tensorflow-gpu==1.12.0")
        estimator.conda_dependencies.add_pip_package("horovod==0.15.2")

        self._logger.debug(estimator.conda_dependencies.__dict__)
        run = self._experiment.submit(estimator)
        if wait_for_completion:
            run.wait_for_completion(show_output=True)
        return run

    def _complete_datastore(self, script_params):
        def _replace(value):
            if isinstance(value, str) and "{datastore}" in value:
                data_path = value.replace("{datastore}/", "")
                return self.datastore.path(data_path).as_mount()
            else:
                return value

        return {key: _replace(value) for key, value in script_params.items()}


class PyTorchExperimentCLI(ExperimentCLI):
    """Creates Experiment object that can be used to create clusters and submit experiments
    
    Returns:
        PyTorchExperimentCLI: Experiment object
    """

    def submit_local(
        self,
        project_folder,
        entry_script,
        script_params,
        dependencies_file=_DEPENDENCIES_FILE,
        wait_for_completion=True,
        docker_args=(),
    ):
        """Submit experiment for local execution
        
        Args:
            project_folder (string): Path of you source files for the experiment
            entry_script (string): The filename of your script to run. Must be found in your project_folder
            script_params (dict): Dictionary of script parameters
            dependencies_file (string, optional): The location of your environment.yml to use to create the
                                                  environment your training script requires. 
                                                  Defaults to _DEPENDENCIES_FILE.
            wait_for_completion (bool, optional): Whether to block until experiment is done. Defaults to True.
            docker_args (tuple, optional): Docker arguments to pass. Defaults to ().
        """
        self._logger.info("Running in local mode")
        self._submit(
            dependencies_file,
            project_folder,
            entry_script,
            "local",
            script_params,
            1,
            1,
            docker_args,
            wait_for_completion,
        )

    def submit(
        self,
        project_folder,
        entry_script,
        script_params,
        dependencies_file=_DEPENDENCIES_FILE,
        node_count=_CLUSTER_MAX_NODES,
        process_count_per_node=4,
        wait_for_completion=True,
        docker_args=(),
    ):
        """Submit experiment for remote execution on AzureML clusters
        
        Args:
            project_folder (string): Path of you source files for the experiment
            entry_script (string): The filename of your script to run. Must be found in your project_folder
            script_params (dict): Dictionary of script parameters
            dependencies_file (string, optional): The location of your environment.yml to use to
                                                  create the environment your training script requires. 
                                                  Defaults to _DEPENDENCIES_FILE.
            node_count (int, optional): [description]. Defaults to _CLUSTER_MAX_NODES.
            process_count_per_node (int, optional): Number of precesses to run on each node. 
                                                    Usually should be the same as the number of GPU for GPU exeuction. 
                                                    Defaults to 4.
            wait_for_completion (bool, optional): Whether to block until experiment is done. Defaults to True.
            docker_args (tuple, optional): Docker arguments to pass. Defaults to ().
        
        Returns:
            azureml.core.Run: AzureML Run object
        """
        self._logger.debug(script_params)

        transformed_params = self._complete_datastore(script_params)
        self._logger.debug("Transformed script params")
        self._logger.debug(transformed_params)

        return self._submit(
            dependencies_file,
            project_folder,
            entry_script,
            self.cluster,
            transformed_params,
            node_count,
            process_count_per_node,
            docker_args,
            wait_for_completion,
        )

    def _submit(
        self,
        dependencies_file,
        project_folder,
        entry_script,
        cluster,
        script_params,
        node_count,
        process_count_per_node,
        docker_args,
        wait_for_completion,
    ):
        self._logger.debug(script_params)
        base_image = "mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda9.0-cudnn7-ubuntu16.04"
        estimator = _create_estimator(
            PyTorch,
            dependencies_file,
            project_folder,
            entry_script,
            cluster,
            script_params,
            base_image,
            node_count=node_count,
            process_count_per_node=process_count_per_node,
            docker_args=docker_args,
        )

        self._logger.debug(estimator.conda_dependencies.__dict__)
        run = self._experiment.submit(estimator)
        if wait_for_completion:
            run.wait_for_completion(show_output=True)
        return run

    def _complete_datastore(self, script_params):
        def _replace(value):
            if isinstance(value, str) and "{datastore}" in value:
                data_path = value.replace("{datastore}/", "")
                return self.datastore.path(data_path).as_mount()
            else:
                return value

        return {key: _replace(value) for key, value in script_params.items()}


def workspace_for_user(
    workspace_name=_WORKSPACE,
    resource_group=_RESOURCE_GROUP,
    subscription_id=_SUBSCRIPTION_ID,
    workspace_region=_REGION,
    config_path=_DEFAULT_AML_PATH,
):
    """ Creates or gets amltoolz.Workspace instance which represents an AML Workspace.

    Args:
        workspace_name (str): Name of workspace
        resource_group (str): Name of Azure Resource group
        subscription_id (str): Azure Subscription ID
        workspace_region (str): Azure region to create resources in
        config_path (str): Path to save AML config to

    Returns:
        amltoolz.Workspace: Either a new workspace created or gets one as identified by name, region and resource group
    """
    return Workspace(
        workspace_name=workspace_name,
        resource_group=resource_group,
        subscription_id=subscription_id,
        workspace_region=workspace_region,
        config_path=config_path,
    )


def tensorboard(runs):
    """ Returns Tensorboard object instantiated with one or more runs

    You can start Tensorboard session by calling start on Tensorboard object
    To stop simply call stop on same object
    Args:
        runs (azureml.core.script_run.ScriptRun or list):

    Returns:
        azureml.tensorboard.Tensorboard

    Examples:
        >>> tb = tensorboard(runs)
        >>> tb.start() # Start Tensorboard
        >>> tb.stop() # Stop Tensorboard
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting tensorboard {pformat(runs)}")
    if isinstance(runs, list):
        return Tensorboard(runs)
    else:
        return Tensorboard([runs])


def _start_and_wait(tb):
    logger = logging.getLogger(__name__)
    try:
        tb.start()
        while True:
            sleep(10)
    except KeyboardInterrupt:
        logger.info("Exiting Tensorboard")
    finally:
        tb.stop()


def _select_runs(experiment, runs=None, status=("Running",)):
    logger = logging.getLogger(__name__)
    try:
        if runs:
            selected_runs = [experiment.runs[run].aml_run for run in runs]
        else:
            selected_runs = [
                run.aml_run for run in experiment.runs if run.aml_run.status in status
            ]
            if len(selected_runs) == 0:
                logger.warn("No runs found")
        return selected_runs
    except KeyError as e:
        logger.warn(f"Did not find run!")
        raise e


def tensorboard_cli(experiment, runs=None, status=("Running",)):
    logger = logging.getLogger(__name__)
    ws = workspace_for_user()
    ws.experiments.refresh()
    try:
        exp_obj = ws.experiments[experiment]
        exp_obj.runs.refresh()
        runs = _select_runs(exp_obj, runs=runs, status=status)
        logger.debug(pformat(runs))
        pipe(runs, tensorboard, _start_and_wait)

    except KeyError:
        logger.warn(f"Did not find experiment {experiment}!")
        logger.warn("Your experiments are:")
        for exp in ws.experiments:
            logger.warn(f"{exp}")


if __name__ == "__main__":
    """ Access workspace and run TensorFlow experiments
    """
    fire.Fire(
        {
            "workspace": workspace_for_user,
            "tf-experiment": TFExperimentCLI,
            "tensorboard": tensorboard_cli,
        }
    )

