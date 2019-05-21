from invoke import task
from storage import create_container
from config import load_config
import logging


def upload_data_from_to(
    c, remote_path, local_path, container_name, account_name, account_key
):
    cmd = (
        f"azcopy --source {local_path} --destination  https://{account_name}.blob.core.windows.net/{container_name}/{remote_path} "
        f"--dest-key {account_key} --quiet --recursive --exclude-older"
    )
    c.run(cmd, pty=True)


@task(pre=[create_container])
def upload_training_data(c):
    """Upload training data to container specified in .env file
    """
    env_values = load_config()
    container_name = env_values.get("CONTAINER_NAME")
    account_name = env_values.get("ACCOUNT_NAME")
    account_key = env_values.get("ACCOUNT_KEY")
    upload_data_from_to(
        c, "train", "/data/train", container_name, account_name, account_key
    )


@task(pre=[create_container])
def upload_validation_data(c):
    """Upload validation data to container specified in .env file
    """
    env_values = load_config()
    container_name = env_values.get("CONTAINER_NAME")
    account_name = env_values.get("ACCOUNT_NAME")
    account_key = env_values.get("ACCOUNT_KEY")
    upload_data_from_to(
        c, "validation", "/data/validation", container_name, account_name, account_key
    )


@task(pre=[upload_training_data, upload_validation_data])
def upload_data(c):
    """Upload training and validation data to container specified in .env file
    """
    print("Data uploaded")


def download_data_from_to(
    c, remote_path, local_path, container_name, account_name, account_key
):
    cmd = (
        f"azcopy --source https://{account_name}.blob.core.windows.net/{container_name}/{remote_path}  --destination {local_path} "
        f"--source-key {account_key} --quiet --recursive --exclude-older"
    )
    c.run(cmd, pty=True)


@task
def download_training(c):
    """Download training data from blob container specified in .env file
    """
    env_values = load_config()
    container_name = env_values.get("CONTAINER_NAME")
    account_name = env_values.get("ACCOUNT_NAME")
    account_key = env_values.get("ACCOUNT_KEY")
    download_data_from_to(
        c, "train", "/data/train", container_name, account_name, account_key
    )


@task
def download_validation(c):
    """Download validation data from blob container specified in .env file
    """
    env_values = load_config()
    container_name = env_values.get("CONTAINER_NAME")
    account_name = env_values.get("ACCOUNT_NAME")
    account_key = env_values.get("ACCOUNT_KEY")
    download_data_from_to(
        c, "validation", "/data/validation", container_name, account_name, account_key
    )


@task(pre=[download_training, download_validation])
def download_data(c):
    """Download training and validation data from blob container specified in .env file
    """
    print("Data downloaded")


@task
def prepare_imagenet(c, download_dir="/data", target_dir="/data"):
    """Prepare imagenet data found in download_dir and push results to target_dir
    
    Args:
        download_dir (str, optional): Location where imagenet tar file should be found. Defaults to "/data".
        target_dir (str, optional): Location where to copy uncompressed imagenet data to. Defaults to "/data".
    """
    from prepare_imagenet import main as prepare_imagenet_data
    logger = logging.getLogger(__name__)
    prepare_imagenet_data(download_dir, target_dir, checksum=False)
    logger.info("Data preparation complete")
