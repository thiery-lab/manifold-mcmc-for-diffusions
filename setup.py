#!/usr/bin/env python3

from setuptools import setup

setup(
    name='sde',
    description='Utilities for inference in SDE models using Mici',
    author='Matt Graham',
    url='https://github.com/thiery-lab/manifold-mcmc-for-diffusions.git',
    packages=['sde'],
    python_requires='>=3.6',
    install_requires=[
        'mici==0.1.3', 'sympy>=1.4', 'numpy>=1.15', 'scipy>=1.1', 
        'jax==0.1.55', 'jaxlib==0.1.37'],
    extras_require={
        'notebook':  [
            'matplotlib>=3.1', 'jupyter>=1.0', 'arviz>=0.5', 'corner>=2.0']
    }

)
