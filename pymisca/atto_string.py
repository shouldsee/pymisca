'''
Author: Feng Geng (fg368@cam.ac.uk)
'''
import pymisca.ptn
import collections
import regex

_DICT_CLASS = collections.OrderedDict
class AttoString(pymisca.ptn.WrapString):
    SEP = '_@@_'
    COLON = '@-@'
    NULL_STRING_LIST = ['NA','None','null']
    _DICT_CLASS = _DICT_CLASS 
    
    @classmethod
    def fromDict(cls,v):
        assert 0,'Use "{0}.fromContainer()" instead'.format(cls.__name__)
        
    def toDict(s):
        cls = type(s)
        assert 0,'Use "{0}.toContainer()" instead'.format(cls.__name__)
#         {}

# @classmethod
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
    
    if type(v) in [int,float,] or isinstance(v,basestring):
        s = unicode(v)
        lst = None
    elif isinstance(v, list):
        lst = ( this_func(cls, _v, **kw)
#                       .rewrap()
                     for _v in v)

#         s = wrap__fromContainer(cls, v, **kw).rewrap()        
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
                    
#                 if type(v) not in [lis]
#                 %v
                
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
        
    if lst is not None:
        s = SEP.join(lst)
        s = cls.new(s,).rewrap()    
    else:
        s = cls.new(s,)
    return s
AttoString.fromContainer = classmethod(wrap1__fromContainer)



def wrap1__toContainer( s, **kw):
    '''
    '''
    _this_func = wrap1__toContainer
    
    
    
    SEP = s.SEP
    COLON = s.COLON
    NULL_STRING_LIST = s.NULL_STRING_LIST
    BRA,KET = s.DBRA_STRIPPED['BRA'], s.DBRA_STRIPPED['KET']
    debug = kw.get('debug',0)
    ##### get type
#     if regex.fullmatch(
#         PTN_BRAKETED.format(**s.DBRA), 
#         s):
    def getNextElement(s):
        posSep = s.find(s.SEP)
        if posSep == -1:
            s = None
        else:
            s = s[posSep:]
        return s
#         s = s[posSep:]
#         posBra = s.find(s.DBRA_STRIPPED['BRA'])
#         if posSep <= posBra:
#             s = s
        
    if s.fullmatch():
        s = s.dewrap()
        def iter__elements(s,SEP=SEP):
            i = i_old = 0
            level = 0
            while True:
                _s = s[i:]
                if not _s:
                    break
                if _s.startswith(BRA):
                    level += 1
                elif _s.startswith(KET):
                    level -= 1
                elif _s.startswith(SEP):
                    if level ==0:
                        yield s[i_old:i]
                        i_old = i + len(SEP)
                    
                i += 1
            yield s[i_old:i]
                
        it = iter__elements(s)
                
        if debug:
            return list(it)
        
        lst = []
        for i, _sp in enumerate(it):
            if not i:
                ele =  list(iter__elements(_sp,SEP=COLON))
                if len(ele) ==1:
                    TYPE = list
                elif len(ele) == 2:
#                 if s.find(s.COLON) < s.find(BRA):
                    TYPE = _DICT_CLASS
                else:
                    assert 0, (len(ele),ele)
#                     TYPE = list
                
            if TYPE is _DICT_CLASS:
                ele = list(iter__elements( _sp,SEP=COLON))
                assert len(ele) == 2, (ele,_sp)
                ele[1] = _this_func( s.new(ele[1]), **kw)
            else:
                ele = _this_func( s.new(_sp), **kw)
            lst.append(ele)
        
        v = lst = TYPE(lst)
        
    else:
        if s in s.NULL_STRING_LIST:
            v = None
        else:
            v = s
        v = cast__empty__element(v)

    return v
AttoString.toContainer = (wrap1__toContainer)

        
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

    
    
#     strict = kw.get('strict',1)
if __name__ == '__main__':

#     testTuplesList = [
#         list('123'),
#         [list('123')],
#         [],
#         ['','123'],
#         ['','123',''],
#         ['123',''],
#         {"1":"2", "3":{"4":"5"}},
#         #{"1":"2", "3":{"4":5}}, #### 
#         {"1":None,'b':'ccccc'},
#                ]
    from collections import OrderedDict
    testTupleList = \
    [(u'[1_@@_2_@@_3]', [u'1', u'2', u'3']),
     (u'[[1_@@_2_@@_3]]', [[u'1', u'2', u'3']]),
     (u'', ''),
     (u'[_@@_123]', ['', u'123']),
     (u'[_@@_123_@@_]', ['', u'123', '']),
     (u'[123_@@_]', [u'123', '']),
     (u'[1@-@2_@@_3@-@[4@-@5]]',
      OrderedDict([(u'1', u'2'), (u'3', OrderedDict([(u'4', u'5')]))])),
     (u'[1@-@NA_@@_b@-@ccccc]', OrderedDict([(u'1', None), (u'b', u'ccccc')]))]

    testList = [x[0] for x in testTupleList]
    testList = map(container__castEmpty, testList)

    # import IPython.display as ipd
    import json
    def _print(x):
        print(x)

    res0 = res = (testList)
    _print(res)

    res1 = res = map(AttoString.fromContainer,testList)
    _print((res))

    res2 = res = map(lambda x:wrap1__toContainer(x,debug=0),res)
    _print(res)
    for expect,x in testTupleList:
#         out = (AttoString.fromContainer(x,strict=0)).toContainer()
        out = (AttoString.fromContainer(x,strict=0))
        back = out.toContainer()
#         .toContainer()
#         if isinstance(x,dict):
#             pass

        assert out == expect,(x,out,expect)
        assert back==  x, (x,out,back)
    it = zip(res0,res2,res1)
#     for e in zip(res0,res1,res2):
    print(json.dumps(it,indent=4))
    
    v = testList[0]
    print
    try:
        s = AttoString.fromDict(v)
    except Exception as e:
        print e
        
    try:
        s = res1[0].toDict()
    except Exception as e:
        print e
    