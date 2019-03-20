#!/usr/bin/env python
#from setuptools import setup
from distutils.core import setup

#from pip.req import parse_requirements
#required = parse_requirements('requirements.txt', session='hack')
#with open('requirements.txt') as f:
#    required = f.read().splitlines()

required = ['numpy','scipy','matplotlib']

print required
setup(
	name='pymisca',
	version='0.1',
	packages=['pymisca',
              'pymisca.tensorflow_extra_',
             'pymisca.model_collection',
             'pymisca.iterative',
             ],
	license='GPL3',
	author='Feng Geng',
	author_email='shouldsee.gem@gmail.com',
	long_description=open('README.md').read(),
#	install_requires,
	install_requires = required,
#		['numpy',
#		'scipy',
#		'matplotlib',]
)

