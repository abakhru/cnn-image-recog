#!/usr/bin/env python

"""The setup script."""

from pathlib import Path
from setuptools import setup, find_packages

setup(author="Amit Bakhru",
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
      install_requires=Path('requirements.txt').read_text().split('\n'),
      license="MIT license",
      long_description=Path('README.md').read_text(),
      include_package_data=True,
      keywords='cnn_image_recog',
      name='cnn_image_recog',
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      test_suite='tests',
      url='https://github.com/abakhru/cnn_image_recog',
      version='0.1.0',
      zip_safe=False,
      )
