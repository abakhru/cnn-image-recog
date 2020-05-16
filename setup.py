#!/usr/bin/env python

from pathlib import Path
from setuptools import find_packages, setup

readme = 'CNN model for Image Classification'

all_reqs = (Path(__file__).parent.joinpath('requirements.txt').read_text().splitlines())
install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')]
tests_require = ['pytest']

setup(
    long_description=readme,
    name='cnn-image-recog',
    version='0.1.0',
    description='CNN model for image recognition',
    python_requires='==3.*,>=3.6.0',
    author='Amit Bakhru',
    author_email='bakhru@me.com',
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=install_requires,
    dependency_links=dependency_links,
)
