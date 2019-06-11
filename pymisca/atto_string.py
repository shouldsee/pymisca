#!/usr/bin/env python2
'''
Author: Feng Geng (fg368@cam.ac.uk)
An alternative serilaiser that allows customised delimiter
'''
import pymisca.ptn
import collections
import regex
_DICT_CLASS = collections.OrderedDict

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
            




# class AttoString( pymisca.ptn.WrapString):
class AttoString( unicode ):
# class AttoString( unicode):
#     SEP = '_@@_'
#     COLON = '@-@'
#     NULL_STRING_LIST = ['NA','None','null']
#     _DICT_CLASS = _DICT_CLASS     
    
    SEP = '-..-'
    COLON = '.__.'
    NULL_STRING_LIST = ['NA','None','null']
    _DICT_CLASS = _DICT_CLASS 
    DBRA = {'BRA':'@--','KET':'--@'}    
 
    @classmethod
    def new(cls, *a,**kw):
        self = cls.__new__(cls,*a,**kw )
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

    def toAttoString(self):
        return attoMeta__toAttoString(self,[unicode(self)])

    @classmethod
    def fromAttoString(cls,v,**kw):
        return cls.new(attoMeta__fromAttoString(cls,v),**kw).dewrap()
        # .__repr__())

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
        s = self.new(self[len(self.BRA()):-len(self.KET())])
        return s
    
    def rewrap(self):
        s = self
        s = pymisca.ptn.quote(s, self.__class__.DBRA_STRIPPED())
        s = self.new(s)
        return s

def attoMeta__toAttoString(self, v):
    return '%s%s'%( self.__class__.__name__, AttoString.fromContainer( v ))
def attoMeta__fromAttoString(cls,v):
    cname = cls.__name__
    assert  v.startswith(cname),"'{1}' must starts with {0}".format(cname,v)
    return AttoString.new(v[ len(cname): ])


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
        return attoMeta__toAttoString(self,[str(self.v)])

    @classmethod
    def fromAttoString(cls,v):
        return float(attoMeta__fromAttoString(cls,v).toContainer()[0])


PY_TYPES = {bool:PY_BOOL, float:PY_FLOAT, int:PY_INT}
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
        INVALID = ' /'
        if next(re.finditer('[%s]'%INVALID,v),None) is not None:
            raise RuntimeError('"{0}" contains invalid characters {1}'.format(v,list(INVALID)))
        lst = unicode(v)
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
        s = cls.new(s,)
        s = s.rewrap()    
    else:
        assert isinstance(lst,basestring), lst
        s = cls.new(lst,)
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
    d.update({v.__name__:v for v in PY_TYPES.values()})
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
AttoString.toContainer = (_toContainer)    



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
testDataDict = [{'output': _AttoStringTestClass(u'@--1_.._@--4_.._a--@_.._2_.._3--@'), 'input': [u'1', [u'4', u'a'], u'2', u'3']}, {'output': _AttoStringTestClass(u'@--1_.._2_.._3--@'), 'input': [u'1', u'2', u'3']}, {'output': _AttoStringTestClass(u'@--@--1_.._2_.._3--@--@'), 'input': [[u'1', u'2', u'3']]}, {'output': _AttoStringTestClass(u''), 'input': ''}, {'output': _AttoStringTestClass(u'@--_.._123--@'), 'input': ['', u'123']}, {'output': _AttoStringTestClass(u'@--_.._123_.._--@'), 'input': ['', u'123', '']}, {'output': _AttoStringTestClass(u'@--123_.._--@'), 'input': [u'123', '']}, {'output': _AttoStringTestClass(u'@--1.--.2_.._3.--.@--4.--.5--@--@'), 'input': OrderedDict([(u'1', u'2'), (u'3', OrderedDict([(u'4', u'5')]))])}, {'output': _AttoStringTestClass(u'@--1.--.NA_.._b.--.ccccc--@'), 'input': OrderedDict([(u'1', None), (u'b', u'ccccc')])}]

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
#    print(json.dumps(res))


