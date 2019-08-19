
import pymisca.ext as pyext
from pymisca.lazydicts.stepper import Stepper
import sys
stepper = Stepper.copy()
stepper['input/list'] = range(10)
stepper['callback/main'] = pyext.PlainFunction(lambda x: x**2)
# stepper['callback/main'] = pyext.PlainFunction(lambda x: x**2)
res = stepper['output/list']

stepper = Stepper.copy()
stepper['input/list'] = range(10)
stepper['callback/main'] = pyext.PlainFunction(lambda x: sys.stderr.write('invisible\n'))
# stepper['callback/main'] = pyext.PlainFunction(lambda x: x**2)
res = stepper['output/list']

stepper = Stepper.copy()
stepper['input/list'] = range(10)
stepper['callback/context'] = None
stepper['callback/main'] = pyext.PlainFunction(lambda x: sys.stderr.write('visible\n'))
res = stepper['output/list']
print(res)