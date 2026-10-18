#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
import glob
try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup

    
setup(
    name="CONF1DENCE",
    url="https://github.com/200k33p3r/CONF1DENCE.git",
    version="0.1.0",
    author="Martin Ying",
    author_email="martin.ying.gr@dartmouth.edu",
    packages=["CONF1DENCE",
              "CONF1DENCE.utils",
              "CONF1DENCE.train",
              "CONF1DENCE.predict"],
    license="LICENSE",
    description="COnditional Normalizing Flow based 1D stellar EvolutioN acCElerator",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "torch>=1.12.0",
        "zuko==1.2.0",
        "h5py>=3.0.0",
        "psutil>=5.8.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        'dev': [
            'jupyter>=1.0.0',
            'notebook>=6.4.0',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)

# write top level __init__.py file with the correct absolute path to package repo
toplevelstr = ("""try:
    from ._version import __version__
except(ImportError):
    pass

               
from . import utils
from . import train
from . import predict      
"""
)

with open('CONF1DENCE/__init__.py','w') as ff:
  ff.write(toplevelstr)
  ff.write('\n')
  ff.write("""__abspath__ = '{0}/'\n""".format(os.path.dirname(os.path.abspath(__file__))))
