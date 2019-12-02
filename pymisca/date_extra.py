import time
import datetime
import tzlocal
import sys
import collections, functools
import os
_this_mod = sys.modules[__name__]
_DICT_CLASS = collections.OrderedDict

import pymisca.header
import decorator

# import tzlocal
def date__formatIso(obj=None):
    if obj is None:
        obj = datetime.datetime.utcnow()
    local_tz = tzlocal.get_localzone() 
    s  = obj.replace(microsecond=0,tzinfo=local_tz).isoformat()
    return s

# import time
# import collections
# import datetime
import pymisca.shell
import json
class scope__timer(object):    
    def __init__(self,data = None, key = None, OFNAME = None, show=0):
        if data is None:
            data = collections.OrderedDict()
        self.data = data
        self.key = key
        self.show = show
        self.OFNAME = os.path.abspath(OFNAME) if OFNAME else OFNAME
#         if OUTNAME is not None
#         self.f=open(OUTNAME,"w")
        return 
    
    def __enter__(self):
        key = self.key = self.key or pymisca.header.get__frameName(level=1)
        self.data.setdefault(key, collections.OrderedDict())
        self.d = self.data[key]
        
#         sys.modules["__main__"]
        self.start = datetime.datetime.now()

        return self

    def __exit__(self, *args):
        self.end = datetime.datetime.now()
        self.dt = self.end - self.start
        d = self.d
        d['start'] =_this_mod.date__formatIso(getattr(self,"start"))
        d['end'] = _this_mod.date__formatIso(getattr(self,"end"))
        d['dt'] = float(getattr(self,"dt").total_seconds())
        if self.OFNAME is not None:
            if pymisca.shell.file__notEmpty(self.OFNAME):
                data = pymisca.shell.read__json( self.OFNAME )
                data.update(self.data)
            else:
                data = self.data
                
            with open(self.OFNAME, "w") as f:
                json.dump(data, f, indent=4)
        if self.show:
            print(json.dumps(d,indent=4))
#                 f.close()

ScopeTimer = scope__timer
# timer = func__timer

def func__timer(timeDict,key=None, debug=0):
    def dec(f,key=key):
        if key is None:
            key = f.__name__
            
#         @decorator.decorator
        @functools.wraps(f)            
        def wrap(*args, **kw):
#             ts = time.time()
            ts = datetime.datetime.now()
            
            result = f(*args, **kw)
#             te = time.time()
            te = datetime.datetime.now()
    
            timeDict[key] = d = _DICT_CLASS()
            d['start'] =_this_mod.date__formatIso(ts)
            d['end'] = _this_mod.date__formatIso(te)
            d['dt'] = float((te - ts).total_seconds())
            
            if debug:
                print(d)
#             print 'func:%r args:[%r, %r] took: %2.4f sec' % \
#               (f.__name__, args, kw, te-ts)
            return result
        return wrap
    return dec
timer = func__timer