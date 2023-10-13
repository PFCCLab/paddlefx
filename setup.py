from __future__ import annotations

import os
import os.path as _osp
import pathlib
import subprocess
import sys
import warnings

import setuptools_scm

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

HERE = pathlib.Path(__file__).parent.resolve()


def _get_version():
    ver = setuptools_scm.get_version()
    return ver


def _get_install_requires():
    with open('requirements.txt', 'r') as f:
        install_requires = f.readlines()
    return install_requires


def _run_cmd(cmd: str, cwd=None):
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        shell=True,
        executable='/bin/bash',
        cwd=cwd,
    )
    stdout, _ = p.communicate()
    out = stdout.decode().strip()
    print(out)
    if p.returncode != 0:
        raise Exception(f'CMD {cmd} failed')
    else:
        return 0


class CMakeBuildExt(build_ext):
    def build_extensions(self):
        os.makedirs(self.build_temp, exist_ok=True)
        os.makedirs(self.build_lib, exist_ok=True)

        debug = int(os.environ.get('DEBUG', 0)) if self.debug is None else self.debug
        build_type = 'Debug' if debug else 'Release'

        cmake_args = (
            f'-DCMAKE_BUILD_TYPE={build_type} -DPython3_EXECUTABLE={sys.executable} '
        )
        cmake_args += f'-DCMAKE_INSTALL_PREFIX={HERE}/src/paddlefx '
        if 'CMAKE_ARGS' in os.environ:
            cmake_args += os.environ.get('CMAKE_ARGS', '')
            cmake_args += ' '

        try:
            import ninja

            ninja_executable_path = _osp.join(ninja.BIN_DIR, 'ninja')
            cmake_args += f'-GNinja -DCMAKE_MAKE_PROGRAM={ninja_executable_path} '
        except ImportError:
            raise Exception('please install ninja first.')

        cmd = f'cmake {cmake_args} -S{HERE} -B{self.build_temp};'
        cmd += f'cmake --build {self.build_temp} --target install'
        _run_cmd(cmd)

        try:
            import mypy  # noqa

            cmd = 'stubgen -m _eval_frame -o .'
            _run_cmd(cmd, cwd=f'{HERE}/src/paddlefx')
        except ImportError:
            warnings.warn('No mypy package is found for stub generating')

        # copy extensions
        for ext in self.extensions:
            fullname = self.get_ext_fullname(ext.name)
            filename = self.get_ext_filename(fullname)
            src = _osp.join(self.build_temp, filename)
            dst = _osp.join(_osp.realpath(self.build_lib), filename)
            os.makedirs(_osp.dirname(dst), exist_ok=True)
            self.copy_file(src, dst)


if __name__ == '__main__':
    ext_modules = [Extension('paddlefx._eval_frame', [])]
    cmdclass = {'build_ext': CMakeBuildExt}
    # TODO: add more info
    setup(
        name='paddlefx',
        description='paddlefx is an experimental project of paddle python IR.',
        license='Apache 2.0',
        license_files=('LICENSE',),
        python_requires='>=3.7',
        install_requires=_get_install_requires(),
        package_dir={'': 'src'},
        packages=find_packages(where='src'),
        package_data={
            'paddlefx': ['py.typed', '*.pyi'],
        },
        ext_modules=ext_modules,
        cmdclass=cmdclass,
    )
