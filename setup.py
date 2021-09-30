#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name='sde',
    description='Utilities for inference in SDE models using Mici',
    author='Matt Graham',
    url='https://github.com/thiery-lab/manifold-mcmc-for-diffusions.git',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'mici==0.1.10',
        'symnum==0.1.2',
        'numpy==1.21',
        'sympy==1.8',
        'jax==0.2.21',
        'jaxlib==0.1.71'
    ],
    extras_require={
        'notebook':  [
            'matplotlib==3.4',
            'arviz==0.11.2',
            'corner==2.2',
            'jupyter>=1.0',
        ],
        'scripts': [
            'matplotlib==3.4',
            'arviz==0.11.2',
            'pandas==1.3',
        ]
    }
)
