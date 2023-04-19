import os
import sys

from setuptools import setup, find_packages
from setuptools.command.install import install

setup(
    name='determinant',
    version='0.1.0',
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=[
        # List your package's core dependencies here, if any.
        "transformers>=4.21.3",
        "torch==1.13.1",
        "ray==2.3.1",
        "fastapi==0.95.0",
        "redis==4.5.4",
        "slack-bolt==1.16.1",
        "slack-sdk==3.19.5",
        "Pillow==9.4.0"
    ],
    python_requires=">=3.7.0,<=3.10.0",
)


# python -m pip install determinant==0.1.0
