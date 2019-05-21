import shutil


def _copy_directories(src, dst):
    try:
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".git"))
    except PermissionError:
        print(f"Could not copy files from {src} to {dst}, permission error")


def _remove_directories(*directories):
    for folder in directories:
        shutil.rmtree(folder)


def _copy_env_file():
    shutil.move("_dotenv_template", ".env")


_CHOICES_DICT = {
    "template":  ("TensorFlow_benchmark",  "TensorFlow_imagenet"),
    "benchmark": ("TensorFlow_experiment", "TensorFlow_imagenet"),
    "imagenet":  ("TensorFlow_benchmark",  "TensorFlow_experiment")
}

if __name__ == "__main__":
    _copy_env_file()
    if {{cookiecutter._remove_unused_projects}}:
        _remove_directories(_CHOICES_DICT.get({{cookiecutter.type}}, tuple()))
