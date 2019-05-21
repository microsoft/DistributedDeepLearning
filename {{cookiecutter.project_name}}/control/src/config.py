import logging
from dotenv import find_dotenv, dotenv_values


def load_config():
    """ Load the variables from the .env file

    Returns:
        .env variables(dict)

    """
    logger = logging.getLogger(__name__)
    dot_env_path = find_dotenv(raise_error_if_not_found=True)
    logger.info(f"Found config in {dot_env_path}")
    return dotenv_values(dot_env_path)

