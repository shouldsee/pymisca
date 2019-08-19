import os
import pymisca.atto_string
import pymisca.tree
# import pymisca.module_wrapper
AttoString = pymisca.atto_string.AttoString
import pymisca.ext as pyext
_DICT_CLASS = pyext._DICT_CLASS


AttoPath=pymisca.atto_string.AttoPath

    
    
# class AttoPath(unicode):
#     sep = os.sep
#     @classmethod
#     def _name(cls):
#         return cls.__name__        
    
#     def as_list(self):
#         return self.split(os.sep)
#     def toAttoString(self):
#         s = pymisca.atto_string.AttoString.fromContainer(self.as_list())
#         s = AttoString.new( type(self).__name__ + s ) 
#         return s
    
#     @classmethod
#     def fromAttoString(cls, v):        
#         assert v.startswith(cls._name())
#         v = v[len(cls._name()):]
#         v = pymisca.atto_string.AttoString.new(v).toContainer()
#         v = os.sep.join(v)
#         v = cls(v)
#         return v
    
#     def __new__(cls,s):
#         self = super(AttoPath,cls).__new__(cls, s)    
#         return self
#     def __repr__(self):
#         s = '{0}({1})'.format(type(self).__name__, 
#                                 super(AttoPath,self).__repr__())
        
#         return s

#     def getPathStack(self,**kw):
#         assert 0
#         def missingDirCallback(pathLastElement, parent):
#             DB_JOB = AttoString.fromAttoString(str(pathLastElement)).toContainer()
#             DB_JOB['INPUTDIR'] = unicode(parent)
#             DB_JOB.update({
#                 'INPLACE':1
#                 })
#             DB_JOB.update(kw)
# #             res = pymisca.module_wrapper.worker__stepWithModule(DB_JOB,**kw)
#             newDir = os.path.basename(res['LAST_DIR'])
#             assert  newDir == pathLastElement,(newDir,pathLastElement, parent)
#             return

#         lst = self.as_list()
#         pymisca.tree.getPathStack(lst,
#                 missingDirCallback = missingDirCallback)

#     def resolve(self, lastN=1):
#         alive = True
#         for ele in self.as_list()[::-1]:
#             alive = alive & AttoString.isAttoString(ele)
#             if not alive:
#                 break


        # last = self.DIR
        # lst = []
        # i = -1
        # while True:
        #     i += 1
        #     if i == lastN:
        #         break
        #     last,curr = last.rsplit('/',1)
            
        #     try:
        #         lst.append(self.loads(curr))
        #     except Exception as e:
        #         print('stopped at %s'%pyext.ppJson(dict(last=last,curr=curr)))
        #         break
                           
                
#         return lst[::-1]        
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
if 0:
    class PathResolver(object):
        def __init__(self, DIR, scope,
                    dumps = None,loads = None):
            if dumps is None:
                dumps = lambda x:json.dumps(x).replace('/',':')
            if loads is None:
                loads = lambda x:json.loads(x)
                
            self.DIR = INPUTDIR
            self.scope = scope
            self.dumps = dumps
            self.loads = loads
            
        def copy(self,):
            return copy.copy(self)
        def __getitem__(self,key):
            s = self.copy()
            v = self.scope[key]
            s.DIR = os.path.join(s.DIR, self.dumps(v) )
            return s
        def __repr__(self):
            return '%s(%r)' % (self.__class__.__name__, dict(DIR=self.DIR,))    
        
        def resolve(self, lastN=1):

            sep = '/'
            last = self.DIR
            lst = []
            i = -1
            while True:
                i += 1
                if i == lastN:
                    break
                last,curr = last.rsplit('/',1)
                
                try:
                    lst.append(self.loads(curr))
                except Exception as e:
                    print('stopped at %s'%pyext.ppJson(dict(last=last,curr=curr)))
                    break
                               
                    
            return lst[::-1]