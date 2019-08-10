'''
Contains a custom tree TreeDict(), with some utilities on transforming a Iterator()
Author: Feng Geng
Email: fg368@cam.ac.uk
'''
import contextlib2
import functools,itertools,collections

import pymisca.patch
import json
import os
import warnings
# import contextlib2

import filelock
import pymisca.atto_string
import pymisca.shell
import copy

import traceback,sys,warnings

import shutil
import tempfile

class TempDirScope(object):
    def __init__(self,getTempDirName = tempfile.mkdtemp,keep=0,**kw):
#         self._d = getTempDirName()
        d = getTempDirName()
        self._stack = getPathStack([d],**kw)
        self.keep = keep
        
    def __getattr__(self,key):
#         return getattr(self._stack,key)
        return self._stack.__getattribute__(key)
    def __enter__(self):
        self._stack.__enter__()
        return self._stack

    def __exit__(self, *a,**kw):
        self._stack.__exit__(*a,**kw)
        if not self.keep:
            shutil.rmtree(self.d)

        
def missingDirCallback__attoShortDirectory(pathElement, parentDir, mode = 511, force=0):
    _d = parentDir
    
    if len(pathElement)<=255:
        _d = _d / pathElement
#         _d = _d.makedirs_p(mode=mode)
    else:
#         res = parentDir.glob("DIR_MAPPER.json")
        FNAME = _d / "DIR_MAPPER.json"
    
        with filelock.FileLock(FNAME+'.lock') as lock:
            
            if pymisca.shell.file__notEmpty(FNAME):
                DIR_MAPPER = pymisca.shell.read__json(FNAME,)
            else:
                DIR_MAPPER = collections.OrderedDict()
#                 pyext._DICT_CLASS()

            res = DIR_MAPPER.get( pathElement, None)
            
            if res is None:
                res = pymisca.atto_string.AttoShortDirectory( len(DIR_MAPPER) )
                res = res.toAttoString()
                DIR_MAPPER[pathElement] = res
                with open(FNAME,'w') as f:
                    f.write(json.dumps(DIR_MAPPER, indent=4))
#                 pyext.printlines([pyext.ppJson(DIR_MAPPER)], FNAME)
            else:
                pass
            
        _d = _d / res
        
    if not force:
        assert _d.isdir(),'Directory does not exists "%s". Use force=1 to force creation' % _d
    else:
        _d.makedirs_p(mode=mode)
#         print('making dir: %s'%_d,)
#         print('exists %s '%_d, _d.exists())
        
    return _d    

def getAttoDirectory(pathList,
                     missingDirCallback = missingDirCallback__attoShortDirectory,
                     **kw):
#     kw.pop('force',None)
    kw['missingDirCallback'] = missingDirCallback
    res = getPathStack(pathList,**kw)

    return res

def AttoDirectory__toDir(attoDir,**kw):
    stack = getAttoDirectory([attoDir],**kw)
    stack.close()
    return stack.d

class PathStack(contextlib2.ExitStack):
    def __init__(self,root,**kw):
        super(PathStack,self).__init__(*a,**kw)
        self._root=root
    pass

def getPathStack(pathList, stack=None, _class=None, 
    force=None, 
    missingDirCallback = None,
    mode = 511,
                 close = False,
traceback_limit  =10,              
	debug=None, printOnExit=None):

    '''
    missingDirCallback: 
        = 1 : create directory if absent
        = None: do nothing and raise error
        = callable : call missingDirCallback(pathElement, parentDir)  
    '''
    ##### legacy
#     if force is not None:
# #         if force == 1:
# #             missingDirCallback = missingDirCallback__attoShortDirectory
#         missingDirCallback = force

#     stack = None
#     assert stack is None,
    assert not isinstance(pathList, basestring),'Path list can not be a single basetring:{pathList}'.format(**locals())

    pathList = sum( [x.split(os.sep) for x in pathList], [] )
#     if stack is not None:
#         warnings.warn('[getPathStack] argument "stack" has been removed')
#         stack = None
        
    if _class is None:
        def _class(*x):
            p = pymisca.patch.FrozenPath(*x)
            p.debug=debug
            p.printOnExit = printOnExit
            return p
    e = None
    # env = {}
    for i,pathElement in enumerate(pathList):
#         print pathElement
        if not i:
            pathElement = {'':os.sep}.get(pathElement,pathElement)

            if stack is None:
                ##### [IMP] __init__()
                stack = contextlib2.ExitStack()
                d = _class('.')
                stack.enter_context(d)
                stack._root = d.realpath()
                stack.d_rel = lambda: stack.d.realpath().relpath(stack._root)
#                 d = _class(pathElement)
            else:
                stack = copy.deepcopy(stack)
                d = stack.d
        try:
#         if 1:
            _d = d.realpath() / pathElement
            _d = _class(_d)
            if not _d.isdir():
#             if not os.path.isdir(pathElement):
#                 assert not os.path.isfile( pathElement )
                if missingDirCallback is None:
                    assert force,'Directory does not exists "%s". Use force=1 to force creation' % _d
                    _d.makedirs_p(mode=mode)
                elif callable(missingDirCallback):
                    res = missingDirCallback(pathElement, d.realpath(), force=force)
                    if res:
                        _d = res

            stack.enter_context(_d)
            d = _d
        except Exception as e:
            _TRACEBACK = traceback.extract_tb(  sys.exc_info()[-1], limit=traceback_limit)
#             e.strerror = pyext.ppJson(e.TRACEBACK)

            s =  '\n[INFO] closest path is "{d}"\n'.format(**locals()) \
            + 'with choices %s '%  json.dumps(d.dirs()[:10],indent=4) +'\n[ERROR] ' \
            + getattr(e,'strerror','[no e.strerror]')
    
            stack.close()
            warnings.warn(s)
            warnings.warn(json.dumps(_TRACEBACK,indent=4))
            raise e
            
#             break 
            
#     if e is not None:
#         try:
# #             e.strerror = pyext.ppJson(e.TRACEBACK)
#             e.strerror =  '\n[INFO] closest path is "{d}"\n'.format(**locals()) \
#             + 'with choices %s '%  json.dumps(d.dirs(),indent=4) +'\n[ERROR] ' \
#             + e.strerror
# #     else:
# #             print('[Exception is None]')
# #     #         stack.__exit__()
#         except:
#             pass
#         finally:
#             stack.close()
#             print(pyext.ppJson(e.TRACEBACK))
#             raise e
            
#             raise e
#     except:
#         raise e
    stack.d = d
    if close:
        stack.close()
    return stack


class TreeDict_0410(collections.OrderedDict):    
    '''
    TreeDict[['a','b','c']] = 
    '''
    def __init__(self,  *args, **kwargs):
        self._sep = '.'
        self.debug = 0
        super(TreeDict, self).__init__(*args,**kwargs)

    def set__sep(self,sep):
        self._sep = sep
    @staticmethod
    def _key_sanitise(key):
        key0 = key
        if not isinstance(key,tuple):
            if isinstance(key,list):
                key = tuple(key)
            else:
                ### wrap non-tuple as a singleton
                key = (key,)
        assert isinstance(key,tuple)
#         assert len(key),'key cannot be empty:%s' %(key0,)
        return key
            
# def __init__(self, *args, **kwargs):
#     super(MyUpdateDict, self).__init__() #<-- HERE call to super __init__
#         self.update(*args, **kwargs)        
    
    def getRoute(self, key):
        route = key.split(self._sep)
        route = tuple(route)
        if self.debug:
            print (route)
        return route
    
    def new__node(self,parent):
        return collections.OrderedDict() 
    #         self.__dict__ = self
    def get(self,key,default=None):
        key = self._key_sanitise(key)
#         root = res = parent = super(TreeDict, self)
#         parent = self
        parent = self
        if self.debug:
            print ('[get.key]',key)
#             assert isinstance(key,tuple)
#                 assert 0
#         print key
        for i, _key in enumerate(key):
            if parent is self:
                getter = super(TreeDict,parent).get
            else:
                getter = parent.get          
            child = getter(_key,None)
                
            if child is None:
#                 if parent is self:
#                     parent[(_key,)] = 
                child = parent[_key] = self.new__node(parent)
            parent = child
            
        return parent
    

        
    def __setitem__(self,key,value):
        key = self._key_sanitise(key)
#         assert len(key)

        parent = self.get(key[:-1])
        if self.debug:
            print '[parent]',parent
        if parent is self:
            setter = super(TreeDict,parent).__setitem__
        else:
            setter = parent.__setitem__
        setter(key[-1],  value)
        
    def __delitem__(self,key):
        key = self._key_sanitise(key)
#         assert len(key)
        parent = self.get(key[:-1])
        
        if parent is self:
            deller = super(TreeDict,parent).__delitem__
        else:
            deller = parent.__delitem__
        deller(key[-1])
    def getFlat(self, key, default=None):
        '''
        self.getFlat('a.b.c') == self[['a','b','c']]
        '''
        key = self.getRoute(key)
        return self.get( key,default=default)
    
    def setFlat(self, key, value):
        '''
        shortcut to a node
        '''
        if self.debug:
            print ('self.setFlat("%s","%s")'%(key,value))
        assert key,'[key must be a string]%s'%key
        key = self.getRoute(key)
        self[key] = value
#         self.__setitem__(key,value)
        
    @classmethod
    def from_flatDict(cls, d, sep=None):
        '''
        transform a flat 
        '''
        tdict = cls()
        if sep is not None:
            tdict.set__sep(sep)
        it = d.iteritems()                
        for k, v in it:
#             print k
            tdict.setFlat(k,v)
        return tdict

    

class TreeDict(collections.OrderedDict):
    """Implementation of perl's autovivification feature."""
    def __init__(self,*a,**kw):
        super(TreeDict,self).__init__(*a,**kw)
        self._sep='.'
#         for k,v in self.iteritems():
        [ self.walk([k]) for k in self]
#             self[k] = TreeDict(v)
            
	self._parent = None
        
    def __missing__(self, key):
        '''
        An instrumental symplification taken from 
        https://stackoverflow.com/a/6781411/8083313
        Original (probably) SO:
        https://stackoverflow.com/q/635483/8083313
        '''
        value = self[key] = type(self)()
        return value
    def set__sep(self,sep):
        self._sep=sep

    def walk(self, route, ):
        d = self
        for key in route:
            res = d[key]
            if isinstance(res,dict):
                if not isinstance(res,TreeDict):
                    #### cast any dict-like object to a TreeDict
                    d[key] = res = type(self)(res)
            d = res
        return d
        
    def getRoute(self, flatPath):
        if self._sep:
            route = flatPath.split(self._sep)
        else:
            route = [flatPath]
        return route

    def getFlatLeaf(self, flatPath):
#         route = flatPath.split(self._sep)
        route = self.getRoute(flatPath)
    
        return self.walk(route)
        
    def setFlatLeaf(self,flatPath, value):
        route = self.getRoute(flatPath)
        d = self.walk(route[:-1])
        d[route[-1]] = value
        
    @classmethod
    def from_flatPathDict(cls, flatPathDict,sep=None, sort=True):
        self = cls()
        if sep is not None:
            self.set__sep(sep)

        it = flatPathDict.iteritems()
        if sort is True:
            it = sorted(it,key= lambda x:x[0])
        for k,v in it:
            self.setFlatLeaf( k, v)
        return self

def treeDict__truncate(d, level):
    d = d.copy()
    for k,v in d.iteritems():
        if isinstance(v,dict):
            if level > 0:
                v = treeDict__truncate(v, level=level-1)
            else:
                v = v.keys()
        d[k] = v
    return d
TreeDict.truncate = treeDict__truncate
    


if __name__ =='__main__':
    ### def it__window()
    assert list(it__window((range(1)),n=2,step=1,fill=None,keep=1)) == [(1,None)]
    assert list(it__window((range(4)),n=2,step=1,fill=None,keep=1)) == [(0, 1), (1, 2), (2, 3), (3, None)]    
