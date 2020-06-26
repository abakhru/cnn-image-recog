#!/usr/bin/env python

"""The setup script."""

from pathlib import Path
from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

all_reqs = (Path(__file__).parent.joinpath('requirements.txt').read_text().splitlines())
install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')]

setup(
    author="Amit Bakhru",
    author_email='bakhru@me.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="CNN model for Image Classification",
    install_requires=install_requires,
    dependency_links=dependency_links,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='cnn_image_recog',
    name='cnn_image_recog',
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    test_suite='tests',
    url='https://github.com/abakhru/cnn_image_recog',
    version='0.1.0',
    zip_safe=False,
)
