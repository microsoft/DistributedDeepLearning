import os
import shutil


def _remove_directory(dirpath):
    if os.path.exists(dirpath):
        try:
            print(f"Deleting directory {dirpath}")
            shutil.rmtree(dirpath)
        except PermissionError:
            print(
                f"The directory contains files that can't be removed please delete {dirpath} and run again"
            )


_remove_directory("{{cookiecutter.experiment_name}}")
print(
    """
Generating project {{cookiecutter.project_name}}
"""
)


