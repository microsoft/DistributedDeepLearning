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

_ALL_DIRECTORIES = "TensorFlow_benchmark",  "TensorFlow_imagenet", "TensorFlow_experiment",  "PyTorch_benchmark",  "PyTorch_imagenet", "PyTorch_experiment"

_CHOICES_DICT = {
    "tensorflow_template":  filter(lambda x: x.lower()!="tensorflow_experiment", _ALL_DIRECTORIES),
    "tensorflow_benchmark": filter(lambda x: x.lower()!="tensorflow_benchmark", _ALL_DIRECTORIES),
    "tensorflow_imagenet":  filter(lambda x: x.lower()!="tensorflow_imagenet", _ALL_DIRECTORIES),
    "pytorch_imagenet":  filter(lambda x: x.lower()!="pytorch_imagenet", _ALL_DIRECTORIES),
    "pytorch_benchmark":  filter(lambda x: x.lower()!="pytorch_benchmark", _ALL_DIRECTORIES),
    "pytorch_template":  filter(lambda x: x.lower()!="pytorch_experiment", _ALL_DIRECTORIES),
}

if __name__ == "__main__":
    _copy_env_file()
    if {{cookiecutter._remove_unused_projects}}:
        _remove_directories(_CHOICES_DICT.get({{cookiecutter.type}}, tuple()))
