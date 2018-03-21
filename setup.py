from distutils.core import setup

setup(
	name='pymisca',
	version='0.1',
	packages=['pymisca',],
	license='GPL3',
	author='Feng Geng',
	author_email='shouldsee.gem@gmail.com',
	long_description=open('README.md').read(),
	install_requires = ['numpy',
		'scipy',
		'matplotlib',]
)

