#!/usr/bin/env python
#from setuptools import setup
from distutils.core import setup
#import setuptools
import os,glob,sys

#os.chdir()

#from pip.req import parse_requirements
#required = parse_requirements('requirements.txt', session='hack')
#with open('requirements.txt') as f:
#    required = f.read().splitlines()

# required = ['numpy','scipy','matplotlib',
# 'fisher@https://github.com/brentp/fishers_exact_test/archive/master.zip',]
DIR= os.path.dirname(__file__)
if DIR:
	os.chdir(DIR)

FILE =  'requirements.txt'
#FILE = os.path.join(os.path.dirname(__file__), 'requirements.txt')
required = [ x.strip() for x in open( FILE,'r')  if not x.strip().startswith('#') ] 
required = [ x.strip() for x in required if x.find(' @ ')==-1 and x ] 
required = [ x.strip() for x in required if x.find('git+')==-1 and x ] 
#required = [ x.strip() for x in required if x ] 

#required = ['git+https://github.com/shouldsee/mixem/archive/9ad994805009545f5befb65b8de9c877bb4f3137.zip']
print (required)
setup(
	name='pymisca',
	version='0.1',
	packages=['pymisca',
              'pymisca.tensorflow_extra_',
              'pymisca.model_collection',
              'pymisca.iterative',
              'pymisca.example-pipe',
              'pymisca.lazydicts',
              'pymisca.atto_jobs_list',
             ],
#   entry_points = {
#           'console_scripts': [
#               'command-name = pymisca.directory_hashmirror_0520:main',                  
#           ],              
#       },
    scripts = glob.glob('bin/*.py') ,
    package_data={'pymisca': ['*.sh','*.json','*.csv','*.tsv','*.npy','*.pk',
                              'templates/*.html',
#                              'resources/*','genomeConfigs/*',
                             ],
# #                  'runtime_data':['wraptool/*.{ext}'.format(**locals()) 
# #                                  for ext in 
# #                                  ['json','csv','tsv','npy','pk']],
                 },
    include_package_data=True,    
#     scripts = {'bin':'*.py'},
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

