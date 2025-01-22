from os import getenv
from setuptools import setup, find_packages
from subprocess import check_output


def get_version():
    major = minor = patch = "0"
    return ".".join((major, minor, patch))


version = get_version()

# Create a version number that is accessible from python runtime
with open("src/learning/version.py", "w") as f:
    f.write(f'__version__ = "{version}"\n')

setup(
    name="learning",
    version=version,
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas==2.2.1",
        "pandas-stubs==2.2.1.240316",
        "matplotlib==3.8.3",
    ],
    python_requires=">=3.10",
)
