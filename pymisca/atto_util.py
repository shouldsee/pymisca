import os
import pymisca.atto_string
class AttoPath(unicode):
    @classmethod
    def _name(cls):
        return cls.__name__        
    
    def toAttoString(self):
        s = pymisca.atto_string.AttoString.fromContainer(self.split(os.sep))
        s = type(self).__name__ + s
        return s
    
    @classmethod
    def fromAttoString(cls, v):        
        assert v.startswith(cls._name())
        v = v[len(cls._name()):]
        v = pymisca.atto_string.AttoString.new(v).toContainer()
        v = os.sep.join(v)
        v = cls(v)
        return v
    
    def __new__(cls,s):
        self = super(AttoPath,cls).__new__(cls, s)    
        return self
    def __repr__(self):
        s = '{0}({1})'.format(type(self).__name__, 
                                super(AttoPath,self).__repr__())
        
        return s
#         return self.toAttoString()

import pymisca.header,sys
TYPE_REGISTRY = pymisca.header.module__getClasses(sys.modules[__name__]) 

if __name__ == '__main__':    
    import pymisca.ext as pyext    
    s = AttoPath(pyext.os.getcwd()).toAttoString() 
    print(s,)
    v = AttoPath.fromAttoString(s)
    print(v,)
    
    s = 'AttoPath@--_::_home_::_shouldsee_::_Documents_::_repos_::_pymisca--@'
    v0 = v = pyext.AttoString.new(s).toContainer()
    print(v.__repr__())
    v1 = v = pyext.AttoString.fromContainer([v])
    v2 = v = v.toContainer()
    pyext.printlines([v0,v1,v2])    
#     def toAttoString():
        
# pyext.path_extra