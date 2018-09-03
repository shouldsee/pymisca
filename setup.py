#!/usr/bin/env python
from distutils.core import setup

setup(
	name='pymisca',
	version='0.1',
	packages=['pymisca','pymisca.tensorflow_extra_'],
	license='GPL3',
	author='Feng Geng',
	author_email='shouldsee.gem@gmail.com',
	long_description=open('README.md').read(),
	install_requires = ['numpy',
		'scipy',
		'matplotlib',]
)

