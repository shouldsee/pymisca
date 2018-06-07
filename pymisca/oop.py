import util as pyutil

##### OOP utility
class util_obj(object):
    def __init__(self,**kwargs):
        self.set_attr(**kwargs)
        pass
    
    def reset(h,method):
        mthd = getattr(h,method)
        if isinstance(mthd, functools.partial):
            setattr(h,method,mthd.func)
        else:
            print "[WARN]:Trying to reset a native method"
        pass

    def partial(h,attrN,**param):
        attr = getattr(h,attrN)
        newattr = functools.partial(attr,**param)
        setattr(h,attrN,newattr)
        pass
    def set_attr(h,**param):
        for k,v in param.items():
            setattr(h, k, v)
        return h