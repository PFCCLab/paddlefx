import pathlib

import setuptools_scm

from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent.resolve()


def _get_version():
    ver = setuptools_scm.get_version()
    return ver


def _get_install_requires():
    with open('requirements.txt', 'r') as f:
        install_requires = f.readlines()
    return install_requires


if __name__ == "__main__":
    # TODO: add more info
    # TODO: add ext_modules
    setup(
        name='paddlefx',
        version=_get_version(),
        description='paddlefx is an experimental project of paddle python IR.',
        license='Apache 2.0',
        license_files=('LICENSE',),
        python_requires='>=3.7',
        install_requires=_get_install_requires(),
        package_dir={'': 'src'},
        packages=find_packages(where='src'),
    )
