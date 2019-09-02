import re,glob
import regex
import slugify,unicodedata

import simpleeval
import inspect


import itertools,collections
import pymisca.header
# import pymisca.shell as pysh
import pandas as pd
fastqLike = ".(fastq|fq)(.gz$|$)"

baseSpace0424 = ptn = r'(?P<ALIAS>.*_S(?P<SAMPLE_ID_INT>\d{1,3}))(?P<HASH_LIKE>)?_L(?P<chunk>\d+)_R(?P<read>[012])_(?P<trail>\d{1,4})\.(?P<ext>.+)'
firthLike0424 = ptn2 = r'(?P<ALIAS>.*_(?P<SAMPLE_ID_INT>\d{1,2}))-(?P<HASH_LIKE>\d{8})'


calculon0606 = ptn = r'[H\-\/](?P<HASH_LIKE>\d{8})/.*L(?P<chunk>\d+)_R(?P<read>[012])\.(?P<ext>.+)'

import pymisca.tree

import IPython


import jinja2
import pymisca.header

_DICT_CLASS = collections.OrderedDict
        
    
def template__format(template, context=None, toFormat= False):
    functions = {'list':list}
#     context['_CONTEXT'] = context
    functions.update( simpleeval.DEFAULT_FUNCTIONS) 
    functions.update(context)

    if context is None:
        context = pymisca.header.get__frameDict(level=1)
        
    ptn =  '([^{]?){([^}]+)}'
    class counter():
        i = -1

    def count(m):
        counter.i += 1
        return m.expand('\\1{%d}'%counter.i)
    
    s = template
    template = re.sub(ptn,string=s, repl= count)
    exprs = [x[1] for x in re.findall(ptn,s)]
    
    
    vals = map(simpleeval.SimpleEval(names=context,
                                    functions=functions).eval,exprs)
#     if vals
        
    if len(vals)==1 and not toFormat:
        res = vals[0]
    else:
        res = template.format(*vals)
    del context
    return res

def template__format(template, context=None):
    functions = {'list':list}
#     functions.update( simpleeval.DEFAULT_FUNCTIONS) 

    if context is None:
        import __main__
#         context = vars(__main__)
        context = pymisca.header.get__frameDict(level=1)
        
    ob = IPython.utils.text.DollarFormatter()
    res = ob.vformat(template,args=[],kwargs = context)
    del context
    return res

from IPython.utils import py3compat
from pymisca.header import name__lookup,ast__eval,frame__default
import inspect
# _self = IPython.utils.text.DollarFormatter()
def template__format(format_string,
                     kwargs= None,
                     context=None,
                     frame = None,
                     level=-1,
                     self = IPython.utils.text.DollarFormatter(),
#                      self=_self
#             args, kwargs,
#             evaler
           ):
    frame = frame__default(frame)
    if context is not None:
        #### legacy renaming
        assert kwargs is None
        kwargs = context 
        
    if kwargs is None:
        kwargs = {}
#     self = IPython.utils.text.DollarFormatter()
#     frame = inspect.currentframe().f_back
    result = []
    for literal_text, field_name, format_spec, conversion in \
            self.parse(format_string):

        # output the literal text
        if literal_text:
            result.append(literal_text)

        # if there's a field, output it
        if field_name is not None:
            # this is some markup, find the object and do
            # the formatting

            if format_spec:
                # override format spec, to allow slicing:
                field_name = ':'.join([field_name, format_spec])

            # eval the contents of the field for the object
            # to be formatted
#                 obj = eval(field_name, kwargs)
#             obj = evaler(field_name)
            obj = ast__eval(field_name, kwargs=kwargs, level=level,frame=frame)
    
#             obj = kwargs.get(field_name, name__lookup(field_name,level=level))
            # do any conversion on the resulting object
            obj = self.convert_field(obj, conversion)

            # format the object and append to the result
            result.append(self.format_field(obj, ''))

    return u''.join(py3compat.cast_unicode(s) for s in result)
f = fformat = template__format

def jf2(template, context=None,undefined=jinja2.StrictUndefined):
    if context is None:
        context = pymisca.header.get__frameDict(level=1)
    res = jinja2.Template(template, undefined=undefined).render(context)
    return res
template__jinja2Format = jf2

def getCounts__bowtie2log(buf):
    '''
    Example:
        32008 reads; of these:
        32008 (100.00%) were unpaired; of these:
        32008 (100.00%) aligned 0 times
        0 (0.00%) aligned exactly 1 time
        0 (0.00%) aligned >1 times
        0.00% overall alignment rate
    '''
#     'grep reads | grep align'
    ptn = re.compile(r'([\d\.\>]+)')
    res = pd.DataFrame(
        map(ptn.findall,buf.splitlines()),
        columns=['count','per','how']).set_index('how')
    return res.to_dict(orient='index')

def getCounts__bowtie1log(buf):
    '''
    Example:
    # reads processed: 32008
    # reads with at least one reported alignment: 3420 (10.68%)
    # reads that failed to align: 28588 (89.32%)
    Reported 3420 alignments
    '''
    buf = buf.replace('at least one','align >=1')
    buf = buf.replace('failed to align','align 0')
    ptn = re.compile(r'([\d\.\>\=]+)')
    res = map(ptn.findall,buf.splitlines())
    res = [ x for x in res if len(x)==3]
    res = pd.DataFrame(
         res,
        columns=['how','count','per',]).set_index('how')
    return res.to_dict(orient='index')



def path__canonlise(s):
    s = unicode(s)
    s = (unicodedata.normalize('NFKD', s)
                    .encode('ascii', 'ignore')
        )
    s = re.sub('\s+',' ', s)
    s = s.strip()
    return s
    

    
def path__norm(s,DIRSEP='.__.'):
    '''
    Adapted from python-slugify

    '''
    
    '''
    Original: re.sub(r'[-\s]+', '-',
            unicode(
                re.sub(r'[^\w\s-]', '',
                    unicodedata.normalize('NFKD', string)
                    .encode('ascii', 'ignore'))
                .strip()
                .lower()))    
                
    '''
    s = path__canonlise(s).lower()
    s = unicode(s)
    s = re.sub('[_]','-',s)
    s = re.sub('[/]',DIRSEP,s)
    s = slugify.slugify(s).strip('-')
    return s

BRAKET = [r'\[',r'\]']
# BRAKET = [r'\{',r'\}']
BRA=BRAKET[0]
KET=BRAKET[1]
DBRA = DICT_BRAKET=dict(BRA=BRA,KET=KET)
DBRA_STRIPPED = {k:v.strip('\\') for k,v in DBRA.items()}

PTN_BRAKETED='{BRA}((?>[^{BRA}{KET}]+|(?R))*){KET}'
def opt__freeze(s):
#     s = re.sub('"([^"]+)"',
#                '{BRA}\\1{KET}'.format(**D_BRAKET),
#                s)
    s = regex.sub(PTN_BRAKETED.format(BRA='"',KET='"'),
                 '{BRA}\\1{KET}'.format(**DBRA_STRIPPED),
                 s)
#     s = re.sub('"([^"]+)"','[\1]',s)
    return s

def opt__defreeze(s):
    s = regex.sub(PTN_BRAKETED.format(**DBRA),
                 '"\\1"',
                 s)
#     s = regex.sub('{BRA}([^{BRA}{KET}]+){KET}'.format(**D_BRAKET),
#                '"\\1"',
#                s)

#     s = re.sub('"([^"]+)"','[\1]',s)
    return s

def opt__brackFreeze(s):
    s = opt__freeze(s)
    s = '{BRA}{s}{KET}'.format(s=s,**DBRA_STRIPPED)  
    return s

from pymisca.atto_string import quote
# def quote(s, BRAKET = None):
#     if BRAKET is None:
#         BRAKET = dict(BRA='"',KET='"')
#     s = u'{BRA}{s}{KET}'.format(s=s,**BRAKET)
#     
#     return s

def dequote(s,BRAKET=None):
    if BRAKET is None:
        BRAKET = dict(BRA='"',KET='"')
    s = regex.sub( PTN_BRAKETED.format(**BRAKET),
                 '\\1',
                 s)    
    return s    
def opt__defreezeDebrack(s):
    s = dequote(s, BRAKET=DBRA)
    s = opt__defreeze(s)
    return s

def opt__serialise(s):
#     s = re.sub('[_]','-',s)
#     s = re.sub('__','____', s)
    s = re.sub('\s+','__', s)
    s = re.sub('[/]+',':',s)
#     s = re.sub('[\'\"]','?',s)
    return s

def opt__deserialise(s):
#     s = s[1:-1]
    s = re.sub('__',' ',s)
    s = re.sub(':','/',s)
#     s = opt__defreezeQuote(s)
    return s
    
    
def opt__toWrap(s):
    s = path__canonlise(s)
    s = s.strip()
    s = opt__serialise(s)
    
    s = opt__freeze(s)
#     s = quote(s,DBRA_STRIPPED)
#     s = opt__brackFreeze(s)
    return s

def wrap__rewrap(s,BRAKET=DBRA_STRIPPED):
    s = quote(s,BRAKET)
    return s
def wrap__dewrap(s,BRAKET=DBRA):
    s = dequote(s,BRAKET)
    return s

def wrap__toOptString(s):
    s = path__canonlise(s)
    s = s.strip()
    s = opt__defreeze(s)
    s = opt__deserialise(s)
#     s = dequote(s,DBRA)
#     s = opt__defreezeDebrack(s)
    return s



class WrapString(unicode):
    '''
    A subclassed string for serialisation keyword arguments
    '''
    SEP = '__'
    PTN_BRAKETED = PTN_BRAKETED 
#     DBRA = BRAKET
    DBRA = DBRA
    #     DBRA_STRIPPED = {k:v.strip('\\') for k,v in self.DBRA.items()}
#     BRAKET= DBRA
    @classmethod
    def DBRA_STRIPPED(cls):
        res = {k:v.strip('\\') for k,v in cls.DBRA.items()}
        return res
    @classmethod
    def new(cls, *a,**kw):
        self = cls.__new__(cls,*a,**kw )
        return self
        
    def __new__(cls, s, VERSION=None, sep = None,
                BRAKET = None):
        # optionally do stuff to value here
        s = opt__toWrap(s)
        self = super(WrapString, cls).__new__(cls, s)
        if VERSION is None:
            VERSION = '20190528'
        self.VERSION = VERSION
        if sep is None:
            sep = cls.SEP
        self.SEP = sep 
#         self.DBRA = BRAKET
#         self.DBRA_STRIPPED = {k:v.strip('\\') for k,v in self.DBRA.items()}
        
#         self.sep = sep
        # optionally do stuff to self here
        return self
    
    def fullmatch(s,):
        res = regex.fullmatch(
            s.PTN_BRAKETED.format(**s.DBRA), 
            s)      
        return res is not None
    
    def rewrap(self):
        s = self
        s = quote(s, self.DBRA_STRIPPED())
        s = self.__class__(s)
        return s
    
    def dewrap(self, BRAKET = None):
        s = self
        s = dequote(s, self.DBRA )
        s = self.__class__(s)        
        return s
    
    def toOptString(self):
        s = self
        s = wrap__toOptString(s)
        return s
    
#     def toUnicode(self):
#         s = unicode(s)
#         return s
    
#     def toString(self):
#         s = self
#         s = wrap__asOpt(s)
#         s = unicode(s)
#         return s
    
    def __init__(self,s):
        pass
    
def wrap__fromDict(cls, d, 
                   **kw):
    '''
    Serialise a dictionary into a WrapString
    '''
    if not isinstance(d,dict):
        d = _DICT_CLASS(d)
    
    sep = kw.get('sep',None)
    if sep is None:
# #         sep = '__'
        sep = cls.SEP
#         sep = '__'
    lst = [None]*len(d)
    for i,(k,v) in enumerate(d.items()):
#         if isinstance(v,)
        
        if type(v) in [int,float,] or isinstance(v,basestring):
#         if isinstance(v,basestring):
            v = WrapString.new(v, **kw).rewrap()
            k = '--%s'%k
        elif isinstance(v, dict):
            v = wrap__fromDict(cls, v, **kw).rewrap()
#             v = dict__asWrap(v).rewrap()
            k = '--%s'%k
        elif isinstance(v, list):
            v = collections.OrderedDict(zip(v,[None]*len(v)))
            v = wrap__fromDict(cls, v, **kw).rewrap()
            k = '--%s'%k

        elif v is None:
            pass
#             k = '%s'%k
        else:
            TYPE = type(v)
            msg = 'Dont know how to intepret type:{TYPE}'.format(**locals())
            raise Exception(msg)
#         lst.append((k,v))
        lst[i] = (k,v)
    
    lst = sum(lst,tuple())
    lst = [x for x in lst if x]
#     print (lst)
    s = sep.join(lst)
    s = WrapString.new(s,**kw)
    return s
WrapString.fromDict = classmethod(wrap__fromDict)

def wrap__getBracketIter(s):
    '''
    Capture the BRAKETs into an iterator
    '''
#     it = regex.finditer( ('([^{BRA}]+)' + PTN_BRAKETED + '([^{KET}]+)').format(**DBRA),s )
    it = regex.finditer( PTN_BRAKETED.format(**DBRA),s )
    return it

def ptn__toSafe(ptn):
    ptn = re.sub('([\[\]])','[\\1]',ptn)
#     s = s.replace('[','[[]').replace(']','[]]')
    return ptn

def safeGlob(ptn):
    ptn = ptn__toSafe(ptn)
    res = glob.glob(ptn)
    return res

def wrap__toDict(self, debug=0, level=-1, _level = 0,
#                  truncate = None
                ):
    '''
    Opposite of wrap__fromDict 
    level: controls how much deseriealisation will be done. For example, it is not 
    '''
    #### initialisation
    s = self    
    _parse_prefix = lambda prefix: [ x for x in  prefix.split(self.SEP) if x] 
    
    
    it = wrap__getBracketIter(s)
    lastSpan = 0,0
    res = lst = []
    i = -1
    
    if debug:
        print('[wrap__toDict]')
        print('[s]%s'%s)
    if (level==-1) or (_level < level): 
        for m in it:
            i+=1
            span = m.span()

            prefix =  s[lastSpan[1]:span[0]]
            wrap = self.__class__( s[slice(*span)] ).dewrap()
            if debug:
                print('[prefix,wrap]{prefix},{wrap}'.format(**locals()))

            
            if wrap.startswith('?'):
                pass
            else:
                wrap = wrap__toDict(wrap,level=level,_level=_level+1)
                
            lst.extend( _parse_prefix( prefix) )
            lst.append(wrap )

            lastSpan = span

    if lastSpan[1]==0:
        lst.append(s)    
        s.isLeaf = True        
    elif lastSpan[1]!=len(s):
        lst.extend(_parse_prefix(s[lastSpan[1]:]))
    ### "lst" conatins the output here
    
    #### converting nested list into a treeDict
    d = collections.OrderedDict()    
    skipNext = 0
    it =pymisca.header.it__window(res,n=2,step=1,fill=None,keep=1)
    it = list(it)
    if debug:
        print(s,it)
    for x0,x1 in it:
#         print(x0,x1)
        if skipNext:
            skipNext = 0
            continue
        if x0.startswith('--') and not getattr(x0,'isLeaf',False):
            assert x1 is not None
            d[x0[2:]] = x1
            skipNext = 1
        else:
            d[x0] = None
            

    return d
WrapString.toDict = (wrap__toDict)




def wrapTreeDict__canonlise(d):
    s = WrapString.fromDict(d)
    d = WrapString.toDict(s)
    return d

class WrapTreeDict(pymisca.tree.TreeDict):
    def canonlise(d):
        d = d.__class__(wrapTreeDict__canonlise(d))
        return d
    
    def __new__(cls,d):
        d = wrapTreeDict__canonlise(collections.OrderedDict(d))
#         d = cls.canonlise(d)
        self= super(WrapTreeDict, cls).__new__(cls, d)
        return self

    def toOptString(self):
        s = WrapString.fromDict(self).toOptString()
        return s
    def toWrapString(self):
        return WrapString.fromDict(self)
    def toWrap(self):
        return self.toWrapString()
# WrapTreeDict(canonalise)




#         s = opt__asWrap(s)
#         print ('[hi]',unicode(s))
#         super(WrapString,self).__init__(s)
if __name__=='__main__':
    
    class pyext(object):
        WrapString = WrapString
    res = []
    def _work(s):
        print s,isinstance(s,pyext.WrapString),type(s); res.append(s)
    OPTS = '--OPTS "--seedlen 15 -k2"'
    s = OPTS
    _work(OPTS)
    s = pyext.WrapString(s)
    _work(s)
    s = pyext.WrapString(s)
    _work(s)
    s = s.rewrap()
    _work(s)
    s = s.dewrap()
    _work(s)

    s = s.toOptString()
    _work(s)

    EXPECTED =[
        '--OPTS "--seedlen 15 -k2"',
 u'--OPTS__[--seedlen__15__-k2]',
 u'--OPTS__[--seedlen__15__-k2]',
 u'[--OPTS__[--seedlen__15__-k2]]',
 u'--OPTS__[--seedlen__15__-k2]',
 u'--OPTS "--seedlen 15 -k2"']
    assert res==EXPECTED,res