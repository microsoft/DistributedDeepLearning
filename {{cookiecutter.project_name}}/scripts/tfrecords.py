from invoke import task
from storage import create_container
from image import upload_data_from_to, download_data_from_to
import os
from glob import glob
from config import load_config
import logging


_BASE_PATH = os.path.dirname(os.path.abspath(__file__))


@task(pre=[create_container])
def upload_training_data(c):
    """Upload tfrecords training data to container specified in .env file
    """
    env_values = load_config()
    container_name = env_values.get("CONTAINER_NAME")
    account_name = env_values.get("ACCOUNT_NAME")
    account_key = env_values.get("ACCOUNT_KEY")
    upload_data_from_to(
        c,
        "tfrecords/train",
        "/data/tfrecords/train",
        container_name,
        account_name,
        account_key,
    )


@task(pre=[create_container])
def upload_validation_data(c):
    """Upload tfrecords validation data to container specified in .env file
    """
    env_values = load_config()
    container_name = env_values.get("CONTAINER_NAME")
    account_name = env_values.get("ACCOUNT_NAME")
    account_key = env_values.get("ACCOUNT_KEY")
    upload_data_from_to(
        c,
        "tfrecords/validation",
        "/data/tfrecords/validation",
        container_name,
        account_name,
        account_key,
    )


@task(pre=[upload_training_data, upload_validation_data])
def upload_data(c):
    """Upload tfrecords training and validation data to container specified in .env file
    """
    print("Data uploaded")


@task
def download_training(c):
    """Download tfrecords training data from blob container specified in .env file
    """
    env_values = load_config()
    container_name = env_values.get("CONTAINER_NAME")
    account_name = env_values.get("ACCOUNT_NAME")
    account_key = env_values.get("ACCOUNT_KEY")
    download_data_from_to(
        c,
        "tfrecords/train",
        "/data/tfrecords/train",
        container_name,
        account_name,
        account_key,
    )


@task
def download_validation(c):
    """Download tfrecords validation data from blob container specified in .env file
    """
    env_values = load_config()
    container_name = env_values.get("CONTAINER_NAME")
    account_name = env_values.get("ACCOUNT_NAME")
    account_key = env_values.get("ACCOUNT_KEY")
    download_data_from_to(
        c,
        "tfrecords/validation",
        "/data/tfrecords/validation",
        container_name,
        account_name,
        account_key,
    )


@task(pre=[download_training, download_validation])
def download_data(c):
    """Download tfrecords training and validation data from blob container specified in .env file
    """
    print("Data downloaded")


def _number_img_files_in(data_path):
    logger = logging.getLogger(__name__)
    all_files = glob(os.path.join(data_path, "**", "*.JPEG"), recursive=True)
    len_files = len(all_files)
    logger.info(f"Found {len_files} files")
    return len_files


@task
def generate_tf_records(c):
    """Convert imagenet images to tfrecords
    """
    print("Preparing tf records")
    if (
        _number_img_files_in("/data/train") == 0
        or _number_img_files_in("/data/validation") == 0
    ):
        raise Exception(
            "Not enough files found please make sure you have downloaded and processed the imagenet data"
        )

    from convert_imagenet_to_tf_records import main as convert_tf_records

    convert_tf_records(
        "/data/train",
        "/data/validation",
        "/data/tfrecords",
        os.path.join(_BASE_PATH, "imagenet_class_index.json"),
    )
