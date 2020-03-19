#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

readme = 'CNN model for Image Classification'

setup(
    long_description=readme,
    name='cnn-image-recog',
    version='0.1.0',
    description='CNN model for image recognition',
    python_requires='==3.*,>=3.6.0',
    author='Amit Bakhru',
    author_email='bakhru@me.com',
    packages=['cnn-image-recog'],
    package_dir={"cnn-image-recog": "src"},
    package_data={},
    install_requires=['colorlog', 'keras==2.3.1', 'matplotlib', 'tensorflow==2.1.0', 'click'],
)
