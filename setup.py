# This setup.py file will be responsible for creating my machine learning application as package.
# So that we can install this package in our project and use it.
# With this setup.py, i'll be able to build my entire machine learning application as a package and deploy in PyPi.
# From Python PypI, anybody can install the package and use it.
from setuptools import find_packages, setup
from typing import List


HYPEN_E_DOT = '-e .'


def get_requirements(file_path:str)->List[str]:
    """
    This function will return the list of requirements.
    """
    requirement_pkg = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirement_pkg = [req.replace('\n', '') for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirement_pkg.remove(HYPEN_E_DOT)

    return requirement_pkg


setup(
name='AI_ML_Project',
version='0.0.1',
author='Chetan',
author_email='chetanmirajkar5@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
)