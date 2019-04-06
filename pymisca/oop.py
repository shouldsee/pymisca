# import util as pyutil
# import pymisca.util as pyutil
import collections
import glob
import os
import warnings
# import pymisca.ext as pyext
##### OOP utility
class util_obj(object):
    def __init__(self,**kwargs):
        self.set_attr(**kwargs)
        pass
    
    def reset(self,method):
        mthd = getattr(self,method)
        if isinstance(mthd, functools.partial):
            setattr(self,method,mthd.func)
        else:
            warnings.warn("[WARN]:Trying to reset a native method")
        pass

    def partial(self,attrN,**param):
        attr = getattr(self,attrN)
        newattr = functools.partial(attr,**param)
        setattr(self,attrN,newattr)
        pass
    def set_attr(self,**param):
        for k,v in param.items():
            setattr(self, k, v)
        return self
    
    def __getitem__(self,k):
        return self.__dict__[k]
    def __setitem__(self,k,v):
        self.__dict__[k] = v
#     def update()


