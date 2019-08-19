import pymisca.ext as pyext
import time
d = {}
def f():
    pass
with pyext.scope__timer(d,) as _timer:
    time.sleep(0.3)
f()
print d
pyext.func__timer(d)(lambda:time.sleep(0.3))()
print d

from pymisca.atto_string import AttoCaster
class temp(AttoCaster):
    def __init__(self,*a,**kw):
        self._data= {}
        with self._timer(OFNAME='test.json') as f:
            pyext.time.sleep(0.5)
            
t = temp()
print (t._data)