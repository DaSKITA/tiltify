from distutils.core import setup
from setuptools import find_packages

setup(
    name='tiltify',
    version='0.1',
    description='A framework for experimentation on privacy policies with NLP',
    author='Michael Gebauer, Faraz Maschhur',
    packages=find_packages(exclude=['test'])
    )
