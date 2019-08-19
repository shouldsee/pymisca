import lazydict
import pymisca.ext as pyext

import pymisca.atto_jobs
from pymisca.module_wrapper import type__resolve as _t

if '__file__' not in locals():
    __file__ = '__debug__'

jobs = lazydict.LazyDictionary()

###
if 0:
    jobs["SCRIPTS"] = lambda self,key:{
        "__self__":__file__,
        "wraptool.template-dna-star-map-0722": "/data/repos/wraptool/wraptool/template-dna-star-map-0722.py",
    }
