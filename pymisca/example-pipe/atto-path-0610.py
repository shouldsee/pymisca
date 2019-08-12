import pymisca.ext as pyext
import pymisca.atto_util 
import os,sys
# print __file__
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  ### hotfix for runtime import
# sys.path.append('/home/shouldsee/repos/pymisca')
with pyext.getPathStack(['','home']):
	print(pyext.shellexec('ls -lhtr '))
	# pyext.dir__indexSubd
# assert 0 
import importlib
# print importlib.import_module("examples.pipe-step2-0608")

s = '/home/shouldsee/Documents/repos/pymisca/example-output/AttoString@--@--RUN_MODULE:-:examples.pipe-step1-0608--@--@/AttoString@--@--RUN_MODULE:-:examples.pipe-step2-0608--@--@'
# s = "/home/shouldsee/Documents/repos/pymisca/example-output/AttoString@--@--RUN_MODULE:-:examples.pipe-step1-0608_::_RUN_PARAMS:-:@----@--@--@/AttoString@--@--RUN_MODULE:-:pymisca.example-pipe.pipe-step2-0608_::_RUN_PARAMS:-:@----@--@--@"
print(os.path.split(s))
# assert 0
pyext.shellexec('rm -rf %s'%s)

x = pymisca.atto_util.AttoPath(s)
x.getPathStack()