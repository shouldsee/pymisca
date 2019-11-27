#!/usr/bin/env python2
'''
Author: Feng Geng (fg368@cam.ac.uk)
An alternative serilaiser that allows customised delimiter
'''
# import pymisca.ptn
import collections
import regex
import warnings
import sys,os
_DICT_CLASS = collections.OrderedDict
_this_mod = sys.modules[__name__]
import warnings

import pymisca.patch
# import numpy as np
import pymisca.shell
import copy
import errno,shutil



def quote(s, BRAKET = None):
    if BRAKET is None:
        BRAKET = dict(BRA='"',KET='"')
    s = u'{BRA}{s}{KET}'.format(s=s,**BRAKET)
    return s

def isSafeType(v):
    return True

def cast__empty__element(v):
    if v in [{},[],'']:
        return ''
    else:
        assert isSafeType(v),(v,type(v))
        return v
    
def container__castEmpty(v):
    if isinstance(v,list):
        v = map(cast__empty__element,v)
    elif isinstance(v,dict):
        for k in v:
            v[k] = cast__empty__element(v[k])
#     else:
    v = cast__empty__element(v)
    
    return v


def string__cut(s,L):
    return s[:L],s[L:]
def debug__output(s,debug=0):
    if debug:
        print(s)
    return s

def string__iter__elementWithLevel(s,SEP,BRA,KET,level,debug=0):
    _this_func = string__iter__elementWithLevel
#     i = i_old = 0
#     level = 0
    initLevel = level
    _debugPipe = lambda *x:debug__output(x,debug=debug)
    _mem = ''
#     memory = ''
    _s = s
    msg = None
    toBreak = 0
    while True:
        if not _s:
            assert level == initLevel,(level,initLevel)
#             if level == 0:
#                 yield _debugPipe('END', level, '',_s)
            break
            
        if _s.startswith(BRA):
            _a,_s = string__cut(_s,len(BRA))
            level += 1
#             yield _a,level,'BRA',_s
            yield _debugPipe('BRA',level,_a,_s)
    
            ele = None
            for ele in _this_func(_s, SEP,BRA,KET,level,debug):
                yield ele      
            assert ele is not None,(_s,s)
            
            assert ele[0] =='KET',ele
            level = ele[1] - 1 
            _s = ele[-1]
            
        elif _s.startswith(KET):
            _a,_s = string__cut(_s,len(KET))
            assert level == initLevel,(level,initLevel)
            msg = 'KET'
            yield _debugPipe(msg,level,_a,_s)
            toBreak = 1
                
        elif _s.startswith(SEP):
            _a ,_s = string__cut(_s,len(SEP))
            yield _debugPipe('SEP',level,_a,_s)
#             yield _debugPipe('MEM',level,_mem,_s)
            _mem = ''
        else:
            _a, _s = string__cut(_s,1)
            _mem += _a
            yield _debugPipe('PLA',level,_a,_s)
            
        if toBreak:
            break
            
PY_TYPES = {}

def _register(func):
    return func


import datetime
class  datetime_util(object):
#     @classmethod
    def __new__(cls,ob):
        if not isinstance(ob,datetime.datetime):
            if isinstance(ob,float):
                ob = datetime.datetime.fromtimestamp(ob)
                return ob
            else:
                assert 0
        else:
            pass
        return ob
    
PY_TYPES["datetime_util"]=datetime_util

@_register
class AttoShortDirectory(unicode):
    
#     def __init__(self,data):
#         self.data = data
#         return 
    
#     def __repr__(self,):
#         return self.toAttoString()
    
    @classmethod
    def fromAttoString(cls, s):
        v = _this_mod.attoMeta__fromAttoString(cls,s).toContainer()
        assert len(v) == 1
        v = v[0]
        v = cls(v)
        return v
    
    def toAttoString(self):
        s = _this_mod.attoMeta__toAttoString(self, [unicode(self)])
        return s
PY_TYPES['AttoShortDirectory'] = AttoShortDirectory

@_register
class AttoStringDict(_DICT_CLASS):
        
    def toAttoString(self):
        return attoMeta__toAttoString(self, 
                                      AttoString.fromContainer(_DICT_CLASS(self)))

    @classmethod
    def fromAttoString(cls,v):
        return attoMeta__fromAttoString(cls,v).toContainer()
PY_TYPES['AttoStringDict'] = AttoStringDict

@_register
class AttoStringList(list):
        
    def toAttoString(self):
        return attoMeta__toAttoString(self, 
                                      AttoString.fromContainer(list(self)))

    @classmethod
    def fromAttoString(cls,v):
        return attoMeta__fromAttoString(cls,v).toContainer()    
    
PY_TYPES['AttoStringList'] = AttoStringList

# class AttoPath(unicode):
class AttoPath(pymisca.patch.FrozenPath):
#     sep = os.sep
    sep = "/"
    @classmethod
    def _name(cls):
        return cls.__name__    
    def __or__(self,other):
        return other(self) 
    def __iter__(self):
        return self.split(self.sep).__iter__()
    
    def join(self,sep):
        return type(self)(sep.join(self))
    
    def as_list(self):
        return self.split(os.sep)
    def toAttoString(self):
        s = _this_mod.AttoString.fromContainer(self.as_list())
        s = _this_mod.AttoString.new( type(self).__name__ + s ) 
        return s
    
    @classmethod
    def fromAttoString(cls, v):        
        assert v.startswith(cls._name())
        v = v[len(cls._name()):]
        v = _this_mod.AttoString.new(v).toContainer()
        v = os.sep.join(v)
        v = cls(v)
        return v
    
    def __new__(cls,s):
#         s = cls.sep.join(map(ValidAttoString, s.split(cls.sep)))
        s = AttoString.new(s, validate=True, 
                           BLACKLIST=''.join(
                               set(AttoString.BLACKLIST)
                               -set(cls.sep),
                           )
                          )
        self = super(AttoPath,cls).__new__(cls, s)    
        return self
    def __repr__(self):
        s = '{0}({1})'.format(type(self).__name__, 
                                super(AttoPath,self).__repr__())
        
        return s

    def getPathStack(self,**kw):
        assert 0
        def missingDirCallback(pathLastElement, parent):
            DB_JOB = AttoString.fromAttoString(str(pathLastElement)).toContainer()
            DB_JOB['INPUTDIR'] = unicode(parent)
            DB_JOB.update({
                'INPLACE':1
                })
            DB_JOB.update(kw)
#             res = pymisca.module_wrapper.worker__stepWithModule(DB_JOB,**kw)
            newDir = os.path.basename(res['LAST_DIR'])
            assert  newDir == pathLastElement,(newDir,pathLastElement, parent)
            return

        lst = self.as_list()
        pymisca.tree.getPathStack(lst,
                missingDirCallback = missingDirCallback)

    def resolve(self, lastN=1):
        alive = True
        for ele in self.as_list()[::-1]:
            alive = alive & AttoString.isAttoString(ele)
            if not alive:
                break
PY_TYPES['AttoPath'] = AttoPath

class AttoHostDirectory(AttoPath):
    ptn = '([^@]+)@([^:]+):([^\s]*)'
    groupName = ['USERNAME','HOSTNAME','DIR']
    template = '{USERNAME}@{HOSTNAME}:{DIR}'
#     sep = os.sep
#     sep = '/'
    @classmethod
    def _name(cls):
        return cls.__name__        
#     def to_AttoPath(self):
#         return AttoPath(self)
    
    def is_remote(self):
        d = self.to_dict()
        return bool( d.get('HOSTNAME','') or d.get('USERNAME',''))
#         return 

    def to_dict(self):
        
        m  = next(re.finditer(self.ptn, self),None)
        d = _DICT_CLASS()
        if m:
            res = zip(self.groupName, m.groups())
            d.update(res)
            d['DIR'] = AttoPath(d['DIR'])
        else:
            d['DIR'] = AttoPath(self)
        return d
    
    @classmethod
    def from_dict(cls,d):
        d['DIR'] = AttoPath(d['DIR'])  
        if d.get('HOSTNAME',''):
            self = cls.template.format(**d)
        else:
            self = AttoPath(d['DIR'])

        self = cls(self)
        return self
    
    def toAttoString(self):
        return attoMeta__toAttoString(self, 
                                      AttoString.fromContainer(self.to_dict()))

    @classmethod
    def fromAttoString(cls,v):
        res = attoMeta__fromAttoString(cls,v).toContainer()
        res = cls.from_dict(res)
        return res
PY_TYPES['AttoHostDirectory'] = AttoHostDirectory

import pymisca.date_extra
import filelock
class AttoCaster(object):
    _DICT_CLASS = collections.OrderedDict
    PARAMS_TRACED = []
    def __init__(self, *a, **kw):
        pass
    def __getitem__(self,key):
        return self._data.__getitem__(key)
    
    def __setitem__(self,key,val):
        return self._data.__setitem__(key,val)
    
    def get(self,*a):
        return self._data.get(*a)
    
    @classmethod
    def _cast(cls, kwargs, rule = None):
        '''
        Casting values of "kwargs" according to "rule" dictionary
        '''
        if rule is None:
            rule = copy.copy(cls.PARAMS_TRACED )
        rule = cls._DICT_CLASS(rule)
        
        import pymisca.module_wrapper as _this_mod
        kwargs = _this_mod.dict__rule__cast(kwargs, rule)
        return kwargs        
    
    @classmethod
    def _lock(cls, DIR = None):
        return filelock.FileLock("%s.lock"% cls.__name__)     
    
#     @classmethod
    def _timer( self, data = None,key = None,**kw):
        if key is None:
            key = type(self).__name__
        data = self["TIME_DICT"] = self.get("TIME_DICT", self._DICT_CLASS())        
        return pymisca.date_extra.scope__timer(data=data,key=key,**kw)
#         return filelock.FileLock("%s.lock"% cls.__name__)     
PY_TYPES["AttoCaster"] = AttoCaster

class AttoJobResult(AttoCaster):
    PARAMS_TRACED = _DICT_CLASS([ 
        ("INPUTDIR",("AttoPath",None)),
        ("OUTDIR",("AttoPath",None)),
        ("LAST_DIR",("AttoPath",None)),
        ("NCORE",("int",1)),
        ("FORCE",("int",0)),
    ])
    
    def __repr__(self):
        return '%r%s'%(type(self), self._data)
#         return repr(self._data)
    
    def __init__(self, fromDict=0, **kwargs):
        self._data = self._cast(kwargs)
        self._result = None
        if not fromDict:
            self._run()

    def _run(self):
        import pymisca.module_wrapper as _this_mod
#         import pymisca.module_wrapper
        assert self._result is None
        res = self._result = self._cast( _this_mod.worker__stepWithModule(self._data))
        return res

    def __getitem__(self, key ):
        res = self._result.__getitem__(key)
        return res

    def __or__(self, other ):
        return other(self)
    
    def to_dict(self):
        return self._result
    
    @classmethod
    def from_dict(cls,kwargs):
        res = cls(fromDict=1,**kwargs)
        return res

    
    def toAttoString(self):
        return attoMeta__toAttoString(self, 
                                      AttoString.fromContainer(self.to_dict()))

    @classmethod
    def fromAttoString(cls,v):
        res = attoMeta__fromAttoString(cls,v).toContainer()
        res = cls.from_dict(res)
#         res = cls(fromDict=1,**kwargs)
        return res    
#     def toAttoString(self)
    
PY_TYPES['AttoJobResult'] = AttoJobResult


class CopyTo(object):
    def __init__(self, OUTDIR = None, INPUTDIR=None, basename = None, force=0,
                 DRY=0,
                 errorFunc=shutil.copy2):
        if OUTDIR is None:
            OUTDIR = os.getcwd()
        self.dest = OUTDIR
        self.src = INPUTDIR
        self.basename = basename
        self.force = force
        self.errorFunc = errorFunc
        self.DRY = DRY
#         self.allowCopy = 1
        
    def call_tuple(self, (FNAME, basename)):
#         _FNAME = os.path.realpath(FNAME)
#         FNAME = 
        _type = type(FNAME)
        if not os.path.isabs(FNAME):
            srcDir = self.src or os.getcwd()
            _src = os.path.join( srcDir, FNAME)
        else:
            _src = FNAME
        if not os.path.exists(_src):
            assert 0,("path does not exist", _src)
            
        _src;
        _basename = basename or self.basename or os.path.basename( _src)
        _dest = os.path.join( self.dest, _basename)
        _dest = _type(_dest)
        if not self.DRY:
            if os.path.isdir( _src):            
                pymisca.shell.dir__link( _src, _dest, force=self.force)

    #             assert 0,"can only download file, not diretory"
            else:
                if not pymisca.shell.file__notEmpty( _src):
                    assert 0, ("FILE is empty", _src)
                else:

                    pymisca.shell.real__dir(fname=_dest)
                    if os.path.abspath(_src) == os.path.abspath(_dest):
                        pass
                    else:
                        if os.path.isfile(_dest):
                            if self.force:
                                os.remove(_dest)
                            else:
                                assert 0,(self.force, "Specify force=1 to overwrite",_src,_dest)
                        try:
                            os.link( _src, _dest)
                        except OSError as e:
                            if e.errno == errno.EXDEV:
                                self.errorFunc(_src,_dest)
                            else:
                                raise e
        else:
            pass
        return _dest
    
    def __call__(self, FNAME, basename = None):
        return self.call_tuple((FNAME,basename))
PY_TYPES['CopyTo'] = CopyTo

# class AttoString( pymisca.ptn.WrapString):
# class ValidAttoString(AttoString):



class AttoString( unicode ):
# class AttoString( unicode):
#     SEP = '_@@_'
#     COLON = '@-@'
#     NULL_STRING_LIST = ['NA','None','null']
#     _DICT_CLASS = _DICT_CLASS     
    
    SEP = '.__.'
    COLON = '.--.'
    NULL_STRING_LIST = ['NA','None','null']
    _DICT_CLASS = _DICT_CLASS 
    DBRA = {'BRA':'@--','KET':'__@'} 
    BLACKLIST = " /"
 
    @classmethod
    def validate(cls,v,BLACKLIST=None):
        if BLACKLIST is None:
            BLACKLIST = cls.BLACKLIST
#         v = self
        SEP = cls.SEP.replace(".","\\.")
        COLON = cls.COLON.replace(".","\\.")
        assert ' ' in BLACKLIST,(BLACKLIST,)
#         INVALID = ' /'
        ptn =  '([{BLACKLIST}]|{SEP}|{COLON})'.format(**locals())
        if next(re.finditer( ptn,  v),None) is not None:    
            raise RuntimeError('"{0}" matches invalider regex  {1}'.format(v,ptn) )
        return v
            
        
    @classmethod
    def new(cls, v, force=None , validate=False, **kw):
        if force is not None:
            validate = not force
            
        self = cls.__new__(cls,v )
#     def new(cls, *a,**kw):
#         self = cls.__new__(cls,*a,**kw )
        
        #### check the AttoString is valid
        if validate:
            cls.validate(self, **kw)
            
        return self
        
    def __new__(cls, s, VERSION=None,**kw):
        del kw
        self = super(AttoString, cls).__new__(cls, s)
        if VERSION is None:
            VERSION = '20190528'
        self.VERSION = VERSION
        return self   
    # def 
    def __repr__(self):
        return '%s(%s)'%( self.__class__.__name__,  super(AttoString,self).__repr__())

#     def toAttoString(self):
#         return attoMeta__toAttoString(self,[unicode(self)])

#     @classmethod
#     def fromAttoString(cls,v,**kw):
#         return cls.new(attoMeta__fromAttoString(cls,v),**kw).dewrap()
#         # .__repr__())

        # return '%s%s'%( self.__class__.__name__,  super(AttoString,self).__repr__())
    
    @classmethod
    def DBRA_STRIPPED(cls):
        res = {k:v.strip('\\') for k,v in cls.DBRA.items()}
        return res
    
    ####### legacy override
    @classmethod
    def fromDict(cls,v):
        assert 0,'Use "{0}.fromContainer()" instead'.format(cls.__name__)
        
    def toDict(s):
        cls = type(s)
        assert 0,'Use "{0}.toContainer()" instead'.format(cls.__name__)
    ####### legacy override
        
#     DBRA = {'BRA':'\[','KET':'\]'}
#     PTN_BRAKETED = '{BRA}((?>(?!{BRA}|{KET}).)*|(?R)*){KET}'
#     PTN_BRAKETED = '{BRA}(.*(?!{BRA}|{KET}).*|(?R)*){KET}'
    @classmethod
    def BRA(cls):
        return cls.DBRA_STRIPPED()['BRA']
    @classmethod
    def KET(cls):
        return cls.DBRA_STRIPPED()['KET']
    
    @classmethod
    def _elementWithLevel(cls, s,level=0,debug=0):
        it = string__iter__elementWithLevel(s,
                                       SEP=cls.SEP,
                                       BRA=cls.BRA(),
                                       KET = cls.KET(),
                                       level=level,
                                       debug=debug)
        return it

    @classmethod
    def _fullmatch(cls, s,debug=0):
        if s:
            it = cls._elementWithLevel(s,level=0,debug=debug)
            res = min (x[1] for x in it) == 1
        else:
            res = False
        return res
    
    def fullmatch(self,**kw):
        return self._fullmatch(self,**kw)
    
    def dewrap(self):
        assert self.fullmatch()
        s = self.new(self[len(self.BRA()):-len(self.KET())],force=1)
        return s
    
    def rewrap(self):
        s = self
        s = quote(s, self.__class__.DBRA_STRIPPED())
        s = self.new(s,force=1)
        return s
    
    def toContainer(self,*a,**kw):
        res = _toContainer(self, *a,**kw)
        if isinstance(res,list):
            res = AttoStringList(res)
        elif isinstance(res,dict):
            res = AttoStringDict(res)
        else:
            warnings.warn('trying to cast a non-container type %s'%(type(res)))
        return res
    
#     def fromContainer(self,)
            
def attoMeta__toAttoString(self=None, v=None,cls=None):
    if cls is not None:
        pass
    else:
        cls = self.__class__
    s = '%s%s'%( cls.__name__, AttoString.fromContainer( v ))
    _LIMIT = 255
    if len(s)> _LIMIT:
        _L = len(s)
        warnings.warn('Producing AttoString longer than {_LIMIT}:length {_L}'.format(**locals()))
    s = AttoString.new(s,force=1)
    return s

def attoMeta__fromAttoString(cls,v):
    cname = cls.__name__
    assert  v.startswith(cname),"'{1}' must starts with {0}".format(cname,v)
    return AttoString.new(v[ len(cname): ],force=1)



class PY_INT(int):

    def toAttoString(self):
        return attoMeta__toAttoString(self,[str(self)])

    @classmethod
    def fromAttoString(cls,v):
        return int(attoMeta__fromAttoString(cls,v).toContainer()[0])


class PY_FLOAT(float):
    def toAttoString(self):
        return attoMeta__toAttoString(self,[str(self)])

    @classmethod
    def fromAttoString(cls,v):
        return float(attoMeta__fromAttoString(cls,v).toContainer()[0])


class PY_BOOL(object):
    def __init__(self,v):
        self.v = v

    def toAttoString(self):
        return attoMeta__toAttoString(self,[str(int(self.v))])

    @classmethod
    def fromAttoString(cls,v):
        return float(attoMeta__fromAttoString(cls,v).toContainer()[0])
    
class PY_FILE(file):
    '''
    '''
    def __init__(self, *a, **kw):
        if isinstance(a[0],file):
            f = a[0]
            a = [f.name,f.mode]
#             kw['encoding'] = f.encoding
            pass
#         else:
        super(PY_FILE,self).__init__(*a,**kw)
            
    def toAttoString(self):
        d = _DICT_CLASS()
        for k in ['name','mode','encoding']:
            val = getattr(self, k)
            if val is not None:
                d[k] = val 
            
        d['name'] = os.path.realpath(d['name']).split(os.sep)
#         d['name'] = 
        return attoMeta__toAttoString(self, d)

    def toJSON(self):
        return 'TBC-nothing'
    
    @classmethod
    def fromAttoString(cls,v):
        d = attoMeta__fromAttoString(cls,v).toContainer()
        name = os.sep.join(d.pop('name'))
        return cls(name,**d)
    
#         return float(attoMeta__fromAttoString(cls,v).toContainer()[0])


PY_TYPES.update({bool:PY_BOOL, float:PY_FLOAT, int:PY_INT, file:PY_FILE, })

@pymisca.header.setItem(PY_TYPES)
class ValidAttoString(AttoString):
    def __new__(cls,s):
        #### recursion error
        # s = super(ValidAttoString, cls).new(s,validate=True)
        ####
        s = AttoString.new(s,validate=True)
        return s

from pymisca.header import func__setAsAttr
import imp
import types
import importlib

class PY_MODULE(types.ModuleType):
    
    @classmethod
    def from_kw(cls,**d):
        return cls.from_dict(d)
    
    @classmethod
    def from_dict(cls,d):
#         if '__name__' in d and '__file__' not in d:
#             importlib.import_module(d['__name__'])
    
        if not '__name__' in d and '__file__' in d:
            d['__name__'] = d['__file__']
            
        if d['__file__'].endswith('.py'):
            mod = imp.load_source(d['__name__'],d['__file__'])
        elif d['__file__'].endswith('.pyc'):
            mod = imp.load_compiled(d['__name__'],d['__file__'])
        else:
            assert 0,(d['__file__'],)
        return mod
    
    @classmethod
    def fromAttoString(cls,v):
        d = attoMeta__fromAttoString(cls,v).toContainer()
        mod = cls.from_dict(d)
        return mod
    
    def __new__(cls,mod):
        
#         @classmethod
#         def fromAttoString():
#             pass
        
        @func__setAsAttr(mod,)
        def toJSON(self=mod):
            return self.__name__
        
        @func__setAsAttr(mod,)
        def toAttoString(self=mod):
            d = _DICT_CLASS()
            d['__name__'] = self.__name__
            d['__package__']  = self.__package__
            d['__file__'] = _this_mod.AttoPath(self.__file__)
            return attoMeta__toAttoString(cls=cls,v=AttoString.fromContainer(d))
        @func__setAsAttr(mod,)
        def fromAttoString(v):
            return cls.fromAttoString(v)

        return mod
PY_TYPES[type(sys)] = PY_MODULE
PY_TYPES['PY_MODULE'] = PY_MODULE



# import numpy as np
# class NUMPY_NDARRAY(np.ndarray):
#     def toAttoString(self):
#         return attoMeta__toAttoString(self, list(self))

#     @classmethod
#     def fromAttoString(cls,v):
#         return cls( attoMeta__fromAttoString(cls,v).toContainer() )
#     def toJSON(self):
#         return '<NUMPY_NDARRAY>'
    
# PY_TYPES[np.ndarray] = PY_TYPES['NUMPY_NDARRAY'] = NUMPY_NDARRAY
# d['NUMPY_NDARRAY']
# @classmethod
import re
def wrap1__fromContainer(cls, v,

#                    serialiser
                   **kw):
    this_func = wrap1__fromContainer
    '''
    Serialise a dictionary into a custom string
    '''
    SEP = cls.SEP
    COLON = cls.COLON
    NULL_STRING_LIST = cls.NULL_STRING_LIST
    strict = getattr(cls,'strict',False) or kw.get('strict',0)
    

    if v is None:
        v = NULL_STRING_LIST[0]
    if type(v) in PY_TYPES:
        v = PY_TYPES.get(type(v))(v)
        # print(hasattr(v,'toAttoString'),type(v),v.toAttoString())

    if False:
        pass
    elif hasattr(v,'toAttoString'):
        lst = v.toAttoString()        
#    elif type(v) in [int, float,bool] or isinstance(v,basestring):
    elif isinstance(v,basestring):
#    if not isinstance(basestring):            
#         s = unicode(v)
#        lst = unicode(v)

        if not isinstance(v,cls):
            v = cls.new(v,validate=True)
        lst = v

                
#         if next(re.finditer('[%s]'%INVALID,v),None) is not None:
#             raise RuntimeError('"{0}" contains invalid characters {1}'.format(v,list(INVALID)))
#         lst = unicode(v)
#        if ' ' in lst or 
#         lst = None
    elif isinstance(v, list):
        lst = [ this_func(cls, _v, **kw)
#                       .rewrap()
                     for _v in v]
    
    elif isinstance(v, dict):
        lst = []
        for ele in v.iteritems():
            k,v = ele
            
            if strict:
                assert isinstance(k,basestring),'Key must be basestring when strict=True:"%s"'%k
            else:
                k = str(k)
                
            if strict:
                if isinstance(v,list) or isinstance(v,dict):
                    pass
                else:
                    assert isinstance(v,basestring),'Value must be basestring when strict=True:"%s"'%(ele,)
                
            _v = '{0}{1}{2}'.format(
                this_func(cls, k, **kw),
                COLON,
                this_func(cls, v, **kw),
            )
            lst.append(_v)
            

    else:
        TYPE = type(v)
        msg = 'Dont know how to intepret type:{TYPE}'.format(**locals())
        raise Exception(msg)    
        
#     print(s,type(s),type(v),getattr(s,'rewrap',lambda:'')())

    if isinstance(lst, list):
        s = SEP.join(lst)
        s = cls.new(s,force=1)
        s = s.rewrap()    
    else:
        assert isinstance(lst,basestring), lst
        assert isinstance(lst,cls), lst
        s =  lst
#         s = cls.new(lst,f)
#     print(type(s),s.rewrap())
    return s

AttoString.fromContainer = classmethod(wrap1__fromContainer)    

# class DEFAULT_CLASS(list):
#     pass
# DEFAULT_TYPE = type(DEFAULT_CLASS)
DEFAULT_TYPE = object()

# import pymisca.atto_util
# print(pymisca.atto_util)
def getDefaultRegistry():
    import pymisca.atto_util
    d = pymisca.atto_util.TYPE_REGISTRY.copy()

#     d.update({v.__name__: v for v in PY_TYPES.values()})
    d.update(PY_TYPES)

    return d

def _toContainer( s, 
                 type_registry=None, 
                 **kw):
    '''
    '''
    _this_func = _toContainer
    # SEP = kw['SEP']
    # COLON = kw['COLON']
    # if isinstance(s,AttoString):
    SEP = s.SEP
    COLON = s.COLON
    NULL_STRING_LIST = s.NULL_STRING_LIST
    BRA,KET = s.BRA(), s.KET()
    debug = kw.get('debug',0)
    
    if type_registry is None:
        type_registry = getDefaultRegistry()
        
    ##### inferType 
    TYPE = DEFAULT_TYPE
    # TYPE = ''
    if s:            
        #### check whether initialise as custom class
        it = s._elementWithLevel(s,level=0)
        buf = ''
        for x in it:
            if x[0]=='PLA':
                buf += x[-2] ## msg,level,_a,_s
            else:
                if x[0] == 'BRA':
                    if buf:
                        TYPE = type_registry[buf]
#                     s = x[-1]
                break
    # if TYPE:
    if TYPE is not DEFAULT_TYPE:
        v = TYPE.fromAttoString(s)
        
    elif s.fullmatch():

            s = s.dewrap()
            it = string__list__members(s,SEP=SEP,BRA=BRA,KET=KET,debug=debug)

            if debug:
                return list(it)
        #         
            lst = []
            TYPE = list
            for i, _sp in enumerate(it):
                if not i:
                    ##### get type
                    ele =  list(string__list__members(_sp, COLON, BRA, KET,debug=debug))
                    if len(ele) <=1:
                        TYPE = list
                    elif len(ele) == 2:
                        TYPE = _DICT_CLASS
                    else:
                        assert 0, (len(ele),ele,type(ele),_sp,s)
        #                 print(TYPE,ele,_sp)

                if TYPE is _DICT_CLASS:
                    ele = list(string__list__members( _sp, COLON, BRA, KET,debug=debug))
                    assert len(ele) == 2, (ele,_sp)
                    ele[1] = _this_func( s.new(ele[1]), **kw)
                else:
                    ele = _this_func( s.new(_sp), **kw)


                lst.append(ele)
            assert callable(TYPE),(TYPE,lst)
            v = lst = TYPE(lst)

    else:
        
        if s in s.NULL_STRING_LIST:
            v = None
        else:
            v = s
        v = cast__empty__element(v)
        
        if isinstance(v,AttoString):
            v = unicode(v)
        # if isinstance(v,basestring):
        #     v 
    return v

# AttoString.toContainer = (_toContainer)    



def string__list__members(s,SEP,BRA,KET,level=0,debug=0):
    class _list(list):
        pass
    llst = lst = _list()
    ele = None
    lastLevel = level
    it = string__iter__elementWithLevel(s,SEP,BRA,KET,level,debug=0)
    msg = None
    for msg,level,_a,_s in it:
        if level == 0:
#             if msg in ['PLA','SEP']:
            if msg =='PLA':
                if ele is None:
                    ele = ''
                ele += _a
                
            if msg == 'SEP':
                if ele is None:
                    ele = ''
            
            if msg in ['SEP','BRA','KET',]:
                if ele is not None:
                    lst.append(ele)
                ele = None
        else:
            if ele is None:
                ele = ''
            ele += _a
            
    if msg == 'SEP':
        if ele is None:
            ele = ''
            
    if ele is not None:
        lst.append(ele)
    ele = None
                        
    assert lst is llst, (s,msg, lst,llst)
    return llst

class _AttoStringTestClass( AttoString):
#    SEP = '_::_'
#    COLON = ':-:'
    NULL_STRING_LIST = ['NA','None','null']
    _DICT_CLASS = _DICT_CLASS 
#    DBRA = {'BRA':'@--','KET':'--@'}
    
from collections import OrderedDict
import json
#testDataDict = [{'output': u'@--1_::_@--4_::_a--@_::_2_::_3--@', 'input': [u'1', [u'4', u'a'], u'2', u'3']}, {'output': u'@--1_::_2_::_3--@', 'input': [u'1', u'2', u'3']}, {'output': u'@--@--1_::_2_::_3--@--@', 'input': [[u'1', u'2', u'3']]}, {'output': u'', 'input': ''}, {'output': u'@--_::_123--@', 'input': ['', u'123']}, {'output': u'@--_::_123_::_--@', 'input': ['', u'123', '']}, {'output': u'@--123_::_--@', 'input': [u'123', '']}, {'output': u'@--1:-:2_::_3:-:@--4:-:5--@--@', 'input': OrderedDict([(u'1', u'2'), (u'3', OrderedDict([(u'4', u'5')]))])}, {'output': u'@--1:-:NA_::_b:-:ccccc--@', 'input': OrderedDict([(u'1', None), (u'b', u'ccccc')])}]
#testDataDict = [{'output': _AttoStringTestClass(u'@--1_.._@--4_.._a--@_.._2_.._3--@'), 'input': [u'1', [u'4', u'a'], u'2', u'3']}, {'output': _AttoStringTestClass(u'@--1_.._2_.._3--@'), 'input': [u'1', u'2', u'3']}, {'output': _AttoStringTestClass(u'@--@--1_.._2_.._3--@--@'), 'input': [[u'1', u'2', u'3']]}, {'output': _AttoStringTestClass(u''), 'input': ''}, {'output': _AttoStringTestClass(u'@--_.._123--@'), 'input': ['', u'123']}, {'output': _AttoStringTestClass(u'@--_.._123_.._--@'), 'input': ['', u'123', '']}, {'output': _AttoStringTestClass(u'@--123_.._--@'), 'input': [u'123', '']}, {'output': _AttoStringTestClass(u'@--1.--.2_.._3.--.@--4.--.5--@--@'), 'input': OrderedDict([(u'1', u'2'), (u'3', OrderedDict([(u'4', u'5')]))])}, {'output': _AttoStringTestClass(u'@--1.--.NA_.._b.--.ccccc--@'), 'input': OrderedDict([(u'1', None), (u'b', u'ccccc')])}]
#testDataDict = [{'output': _AttoStringTestClass(u'@--1-..-@--4-..-a--@-..-2-..-3--@'), 'input': [u'1', [u'4', u'a'], u'2', u'3']}, {'output': _AttoStringTestClass(u'@--1-..-2-..-3--@'), 'input': [u'1', u'2', u'3']}, {'output': _AttoStringTestClass(u'@--@--1-..-2-..-3--@--@'), 'input': [[u'1', u'2', u'3']]}, {'output': _AttoStringTestClass(u''), 'input': ''}, {'output': _AttoStringTestClass(u'@---..-123--@'), 'input': ['', u'123']}, {'output': _AttoStringTestClass(u'@---..-123-..---@'), 'input': ['', u'123', '']}, {'output': _AttoStringTestClass(u'@--123-..---@'), 'input': [u'123', '']}, {'output': _AttoStringTestClass(u'@--1.__.2-..-3.__.@--4.__.5--@--@'), 'input': OrderedDict([(u'1', u'2'), (u'3', OrderedDict([(u'4', u'5')]))])}, {'output': _AttoStringTestClass(u'@--1.__.NA-..-b.__.ccccc--@'), 'input': OrderedDict([(u'1', None), (u'b', u'ccccc')])}]
#testDataDict = [{'output': _AttoStringTestClass(u'@--1.__.@--4.__.a__@.__.2.__.3__@'), 'input': [u'1', [u'4', u'a'], u'2', u'3']}, {'output': _AttoStringTestClass(u'@--1.__.2.__.3__@'), 'input': [u'1', u'2', u'3']}, {'output': _AttoStringTestClass(u'@--@--1.__.2.__.3__@__@'), 'input': [[u'1', u'2', u'3']]}, {'output': _AttoStringTestClass(u''), 'input': ''}, {'output': _AttoStringTestClass(u'@--.__.123__@'), 'input': ['', u'123']}, {'output': _AttoStringTestClass(u'@--.__.123.__.__@'), 'input': ['', u'123', '']}, {'output': _AttoStringTestClass(u'@--123.__.__@'), 'input': [u'123', '']}, {'output': _AttoStringTestClass(u'@--1-..-2.__.3-..-@--4-..-5__@__@'), 'input': OrderedDict([(u'1', u'2'), (u'3', OrderedDict([(u'4', u'5')]))])}, {'output': _AttoStringTestClass(u'@--1-..-NA.__.b-..-ccccc__@'), 'input': OrderedDict([(u'1', None), (u'b', u'ccccc')])}]
testDataDict = [{'output': _AttoStringTestClass(u'@--1.__.@--4.__.a__@.__.2.__.3__@'), 'input': [u'1', [u'4', u'a'], u'2', u'3']}, {'output': _AttoStringTestClass(u'@--1.__.2.__.3__@'), 'input': [u'1', u'2', u'3']}, {'output': _AttoStringTestClass(u'@--@--1.__.2.__.3__@__@'), 'input': [[u'1', u'2', u'3']]}, {'output': _AttoStringTestClass(u''), 'input': ''}, {'output': _AttoStringTestClass(u'@--.__.123__@'), 'input': ['', u'123']}, {'output': _AttoStringTestClass(u'@--.__.123.__.__@'), 'input': ['', u'123', '']}, {'output': _AttoStringTestClass(u'@--123.__.__@'), 'input': [u'123', '']}, {'output': _AttoStringTestClass(u'@--1.--.2.__.3.--.@--4.--.5__@__@'), 'input': OrderedDict([(u'1', u'2'), (u'3', OrderedDict([(u'4', u'5')]))])}, {'output': _AttoStringTestClass(u'@--1.--.NA.__.b.--.ccccc__@'), 'input': OrderedDict([(u'1', None), (u'b', u'ccccc')])}]
#testDataDict  =[{"output": "@--1_.._@--4_.._a--@_.._2_.._3--@", "input": ["1", ["4", "a"], "2", "3"]}, {"output": "@--1_.._2_.._3--@", "input": ["1", "2", "3"]}, {"output": "@--@--1_.._2_.._3--@--@", "input": [["1", "2", "3"]]}, {"output": "", "input": ""}, {"output": "@--_.._123--@", "input": ["", "123"]}, {"output": "@--_.._123_.._--@", "input": ["", "123", ""]}, {"output": "@--123_.._--@", "input": ["123", ""]}, {"output": "@--1.--.2_.._3.--.@--4.--.5--@--@", "input": {"1": "2", "3": {"4": "5"}}}, {"output": "@--1.--.NA_.._b.--.ccccc--@", "input": {"1": null, "b": "ccccc"}}]    
if __name__ == '__main__':
    

    # testDataDict = {'output': u'@:1_@@_@:4_@@_a:@_@@_2_@@_3:@', 'input': [u'1', [u'4', u'a'], u'2', u'3']}, {'output': u'@:1_@@_2_@@_3:@', 'input': [u'1', u'2', u'3']}, {'output': u'@:@:1_@@_2_@@_3:@:@', 'input': [[u'1', u'2', u'3']]}, {'output': u'', 'input': ''}, {'output': u'@:_@@_123:@', 'input': ['', u'123']}, {'output': u'@:_@@_123_@@_:@', 'input': ['', u'123', '']}, {'output': u'@:123_@@_:@', 'input': [u'123', '']}, {'output': u'@:1:-:2_@@_3:-:@:4:-:5:@:@', 'input': OrderedDict([(u'1', u'2'), (u'3', OrderedDict([(u'4', u'5')]))])}, {'output': u'@:1:-:NA_@@_b:-:ccccc:@', 'input': OrderedDict([(u'1', None), (u'b', u'ccccc')])}
    # testDataDict = [{'output': u'@--1_@@_@--4_@@_a--@_@@_2_@@_3--@', 'input': [u'1', [u'4', u'a'], u'2', u'3']}, {'output': u'@--1_@@_2_@@_3--@', 'input': [u'1', u'2', u'3']}, {'output': u'@--@--1_@@_2_@@_3--@--@', 'input': [[u'1', u'2', u'3']]}, {'output': u'', 'input': ''}, {'output': u'@--_@@_123--@', 'input': ['', u'123']}, {'output': u'@--_@@_123_@@_--@', 'input': ['', u'123', '']}, {'output': u'@--123_@@_--@', 'input': [u'123', '']}, {'output': u'@--1:-:2_@@_3:-:@--4:-:5--@--@', 'input': OrderedDict([(u'1', u'2'), (u'3', OrderedDict([(u'4', u'5')]))])}, {'output': u'@--1:-:NA_@@_b:-:ccccc--@', 'input': OrderedDict([(u'1', None), (u'b', u'ccccc')])}]
    # testDataDict = [{'output': u'@:1_@@_@:4_@@_a:@_@@_2_@@_3:@', 'input': [u'1', [u'4', u'a'], u'2', u'3']}, {'output': u'@:1_@@_2_@@_3:@', 'input': [u'1', u'2', u'3']}, {'output': u'@:@:1_@@_2_@@_3:@:@', 'input': [[u'1', u'2', u'3']]}, {'output': u'', 'input': ''}, {'output': u'@:_@@_123:@', 'input': ['', u'123']}, {'output': u'@:_@@_123_@@_:@', 'input': ['', u'123', '']}, {'output': u'@:123_@@_:@', 'input': [u'123', '']}, {'output': u'@:1:-:2_@@_3:-:@:4:-:5:@:@', 'input': OrderedDict([(u'1', u'2'), (u'3', OrderedDict([(u'4', u'5')]))])}, {'output': u'@:1:-:NA_@@_b:-:ccccc:@', 'input': OrderedDict([(u'1', None), (u'b', u'ccccc')])}]
    res = []
    for i,d in enumerate(testDataDict):
        d['_output'] = _AttoStringTestClass.fromContainer(d['input'])
        d['_input'] = _AttoStringTestClass.toContainer(d['_output'])
        print('[i]%s'%i,d['input'],d['_output'])
        assert d['_output']==d['output'],json.dumps(d,indent=4)
        assert d['_input']==d['input'],json.dumps(d,indent=4)

#         print('[i]%s'%i,json.dumps(d,))
	d['output'] = d.pop('_output')
	#d['input'] = 
	d.pop('_input')
        res.append(d)
    print(repr(res))
    
#     ContainerString()
#    print(json.dumps(res))


