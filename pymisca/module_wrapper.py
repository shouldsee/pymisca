import os,sys,re
### importing appropriate modules
import pymisca.ext as pyext
import pymisca.ptn
import pymisca.blocks
import pymisca.tree

import pymisca.atto_util 
import pymisca.atto_string

import importlib
import datetime

import traceback
# import wraptool
_this_mod = sys.modules[__name__]
# import fastqDB.ptn
_DICT_CLASS  =_DICT = pyext.collections.OrderedDict

# _getPathStack = pyext.getPathStack
_getPathStack = pyext.getAttoDirectory

import copy
import __builtin__
import pymisca.atto_string
_REG = pymisca.atto_string.getDefaultRegistry()
def type__resolve(typeName):
    _this_func = type__resolve
    
    if not isinstance(typeName, basestring):
        return typeName    
    
    res = getattr(__builtin__, typeName,None)
    if res is not None:
        return _this_func(res)
        
    res = _REG.get(typeName,None)
    if res is not None:
        return _this_func(res)
    
    assert res is not None,pyext.f('Cant resolve type: {typeName}')
    


    
def basename__toDict(full_name,VERSION):
    bname = os.path.basename(full_name)
    if VERSION >= '20190608':
        if bname.startswith("AttoString"):
            d = pymisca.atto_string.AttoString.new(bname).toContainer()
        else:
            d = {'INPUTDIR_BASENAME':bname}
    else:
        s = pymisca.ptn.WrapString(os.path.basename(bname))
        d = s.toDict()
    return d





def _bookkeep(DB_WORKER):
    def _main(OUTDIR,**kw):
        with _getPathStack([OUTDIR],force=1,
                                printOnExit=DB_WORKER['RUNTIME']['printOnExit']):
            
            d = {k:v for k,v in DB_WORKER.items() if not k.startswith('_')}
            d['DB_SCRIPT'] = {k:v for k,v in d['DB_SCRIPT'].items() if not k.startswith('_')}
            pyext.util__fileDict.main(
                argD=d,
                ofname='FILE.json')
        
    _ = _main(**DB_WORKER)
    DB_WORKER['RUNTIME']['MSG'] = {
        'CONTENT':
        '[_bookkeep] finished [RUN_MODULE:"{RUN_MODULE}"] at [OUTDIR:"{OUTDIR}"]'.format(**DB_WORKER),
        'VERBOSITY':1 }
    
    if DB_WORKER['VERSION'] >= '20190721':
        ### [20190721] casting AttoDirectory
        DB_WORKER["OUTDIR"] = pyext.AttoDirectory__toDir(DB_WORKER["OUTDIR"])
        
    return DB_WORKER

def getDefault__OUTDIR(DB_WORKER,):
    '''
    DEPRECATED
    '''
    if DB_WORKER['VERSION'] <= '20190528':
        OUTDIR = pymisca.blocks.compute__OUTDIR(
            inplace = DB_WORKER.get('INPLACE',1),
            pathLevel = DB_WORKER.get('pathLevel',None),
            INPUTDIR=DB_WORKER['INPUTDIR'], baseFile=0,
            SELFALI=DB_WORKER['DB_SCRIPT']['MODULE_NAME'].rsplit('.',1)[-1] 
        )()
    else:
#     elif DB_WORKER['VERSION'] <= '20190607':
#         DB_SCRIPT = _DICT(DB_WORKER['DB_SCRIPT'])
        
#         ### get Non-default parameters
#         for k,v in DB_SCRIPT['PARAMS_TRACED'].items():
#             if k in DB_WORKER['RUN_PARAMS']:
#                 if DB_WORKER['RUN_PARAMS'][k] == v:
#                     del DB_SCRIPT['PARAMS_TRACED'][k]

#             else:
#                 DB_WORKER['RUN_PARAMS'][k] = DB_SCRIPT['PARAMS_TRACED'].pop(k)
                
        DB_SCRIPT = _DICT(DB_WORKER['DB_SCRIPT'])
                
        
                
        pyext.dict__key__func(DB_WORKER, 'INPUTDIR', pymisca.atto_util.AttoPath)
        

        if DB_WORKER['VERSION'] < '20190619':
            _keys = DB_SCRIPT['PARAMS_TRACED'].keys()
        else:
            _keys = DB_SCRIPT['PARAMS_TRACED'].keys()
#             _keys = [x[0] for x in DB_SCRIPT['PARAMS_TRACED']]
            
            
        #### [TBD]
        if DB_WORKER['VERSION'] <'20190610':
            d = _DICT(RUN_MODULE=DB_WORKER['RUN_MODULE'])
            d.update(pyext.dictGetList(
                    DB_WORKER['RUN_PARAMS'],
                    _keys
                      )
                    )
        elif True:
                
            d = pyext.dictGetList(DB_WORKER,
                    ['RUN_MODULE','RUN_PARAMS']
                    )
                
            d['RUN_PARAMS'] = pyext.dictGetList(
                    DB_WORKER['RUN_PARAMS'],
                    _keys
                      )
            if not d['RUN_PARAMS']:
                del d['RUN_PARAMS']
                                        
        else:
            assert 0
        
        
        if DB_WORKER['VERSION'] >= '20190608':
            SELFALI = pymisca.atto_string.AttoStringDict(d).toAttoString()
#            SELFALI = pymisca.atto_string.AttoString.fromContainer(d)
#             .toAttoString()
        else:
            SELFALI = pyext.WrapString.fromDict(d,VERSION=DB_WORKER['VERSION'])
#         print(SELFALI)
#         if OUTDIR
        
        OUTDIR = pymisca.blocks.compute__OUTDIR(
            inplace = DB_WORKER.get('INPLACE',1),
            pathLevel = None,  #### to-be-deprecated
            INPUTDIR=DB_WORKER['INPUTDIR'], baseFile=0,   
            SELFALI=SELFALI,
#             SELFALI=DB_WORKER['DB_SCRIPT']['MODULE_NAME'].rsplit('.',1)[-1] 
        )()
#     else:
#         pass
        
    return OUTDIR



def getDefault__SUFFIX(DB_WORKER):
    DB_SCRIPT = _DICT(DB_WORKER['DB_SCRIPT'])
    pyext.dict__key__func(DB_WORKER, 'INPUTDIR', pymisca.atto_util.AttoPath)
    _keys = DB_SCRIPT['PARAMS_TRACED'].keys()
#             _keys = [x[0] for x in DB_SCRIPT['PARAMS_TRACED']]
    #### [TBD]
    if 1:
        d = pyext.dictGetList(DB_WORKER,
                ['RUN_MODULE','RUN_PARAMS']
                )
            
        d['RUN_PARAMS'] = pyext.dictGetList(
                DB_WORKER['RUN_PARAMS'],
                _keys
                  )
        if not d['RUN_PARAMS']:
            del d['RUN_PARAMS']
    
    SELFALI = pymisca.atto_string.AttoStringDict(d).toAttoString()
    
    return SELFALI        
        
# from pymisca.atto_string import dict__rule__cast
    
def tree__castType(tree, typs):
    _this_func = tree__castType
    _type, typs = typs[0],typs[1:]
    _type = type__resolve(_type)
    
#     if isinstance(tree, list):
#     if isinstance(tree, list):

    if issubclass(_type,list):
        assert not isinstance( tree, basestring), (tree,_type)
        tree = [ _this_func(x,typs) for x in tree ] 
        
#     elif isinstance(tree, dict):
    elif issubclass(_type, dict):
        assert not isinstance( tree, basestring), (tree,_type)
        tree = pyext._DICT_CLASS([ (x,_this_func(tree[x],typs) ) for x in tree ])
        
    if _type != object:
        tree = _type(tree)
    return tree

def dict__rule__cast(DICT_RUN,DICT_RULE):
#     DICT_RUN = DB_WORKER['RUN_PARAMS']
#     DICT_RULE = DB_SCRIPT['PARAMS_TRACED']
    ##### v is now a tuple instead of default
    for k,v in DICT_RULE.items():
        typs = v[0].split(':')
        vdef = v[1]

        if vdef is not None:
            vdef = tree__castType(vdef, typs)

    #                 if k in DB_WORKER['RUN_PARAMS']:
        vrun = DICT_RUN.get(k,None)
        if vrun is not None:
            vrun = tree__castType(vrun, typs)

        if vdef == vrun \
            :
    #                         or (vdef is not None and vrun is None):
            del DICT_RULE[k]

        elif (vrun is None and vdef is not None):
            del DICT_RULE[k]
    #                         value = worker__value__interpret(DB_WORKER,vdef)
            DICT_RUN[k] = vdef
        else:
            assert vrun is not None
    #                         value = worker__value__interpret(DB_WORKER,vrun)
            DICT_RUN[k] = vrun    
    return DICT_RUN

def _callbefore__0529(DB_WORKER):    
    
    def _main(INPUTDIR,**kw):
        DB_WORKER['CWD'] = pyext.os.getcwdu()
        
        #### get DATA_DICT from pathname
        DB_WORKER['DATA_DICT'] = basename__toDict(INPUTDIR,VERSION=DB_WORKER['VERSION'])
        
        if 'inplace' in DB_WORKER:
            #### legacy
            DB_WORKER['INPLACE'] = DB_WORKER.pop('inplace')

        
        OUTDIR = None \
            or DB_WORKER.get('OUTDIR',None) \
            or DB_WORKER.get('_OUTDIR',None) \
            or DB_WORKER['DB_SCRIPT'].get('OUTDIR',None) \
            or DB_WORKER['DB_SCRIPT'].get('_OUTDIR',None) \
        
        DB_SCRIPT = _DICT(DB_WORKER['DB_SCRIPT'])
        
        
        VERSION = DB_WORKER['VERSION']
        vals = DB_SCRIPT['PARAMS_TRACED'].values()
        if len(vals):
            ### get Non-default parameters        
            if (not isinstance(vals[0],basestring)) \
                and (vals[0] is not None) \
                and (len(vals[0])==2): 
                
#                 DICT_RUN = 
                DB_WORKER['RUN_PARAMS'] = dict__rule__cast(DB_WORKER['RUN_PARAMS'],DB_SCRIPT['PARAMS_TRACED'])
#                 DICT_RUN = DB_WORKER['RUN_PARAMS']
#                 DICT_RULE = DB_SCRIPT['PARAMS_TRACED']
#                 ##### v is now a tuple instead of default
#                 for k,v in DICT_RULE.items():
#                     typs = v[0].split(':')
#                     vdef = v[1]

#                     if vdef is not None:
#                         vdef = tree__castType(vdef, typs)

#     #                 if k in DB_WORKER['RUN_PARAMS']:
#                     vrun = DICT_RUN.get(k,None)
#                     if vrun is not None:
#                         vrun = tree__castType(vrun, typs)

#                     if vdef == vrun \
#                         :
# #                         or (vdef is not None and vrun is None):
#                         del DICT_RULE[k]
    
#                     elif (vrun is None and vdef is not None):
#                         del DICT_RULE[k]
# #                         value = worker__value__interpret(DB_WORKER,vdef)
#                         DICT_RUN[k] = vdef
#                     else:
#                         assert vrun is not None
# #                         value = worker__value__interpret(DB_WORKER,vrun)
#                         DICT_RUN[k] = vrun
                 
                
                for k in DB_WORKER['RUN_PARAMS']:
                    value = DB_WORKER['RUN_PARAMS'][k]
                    DB_WORKER['RUN_PARAMS'][k] = type(value)(worker__value__interpret(DB_WORKER,value))
            else:
    #         if DB_WORKER['VERSION'] <'20190619':
                for k,v in DB_SCRIPT['PARAMS_TRACED'].items():
                    if k in DB_WORKER['RUN_PARAMS']:
                        if DB_WORKER['RUN_PARAMS'][k] == v:
                            del DB_SCRIPT['PARAMS_TRACED'][k]
                            fillThis = 0
                        elif DB_WORKER['RUN_PARAMS'][k] is None:
                            fillThis = 1
                        else:
                            fillThis = 0
                    else:
                        fillThis = 1

                    if fillThis:
                        DB_WORKER['RUN_PARAMS'][k] = DB_SCRIPT['PARAMS_TRACED'].pop(k)
                    
        if OUTDIR is None:
            OUTDIR = '!{INPUTDIR}/{SUFFIX}'
            
        if callable(OUTDIR):
            OUTDIR = OUTDIR(DB_WORKER)
        elif isinstance(OUTDIR,basestring):
            if '{SUFFIX}' in OUTDIR:
                DB_WORKER['SUFFIX'] =  getDefault__SUFFIX(DB_WORKER)
            DB_WORKER['OUTDIR'] = OUTDIR
            OUTDIR = worker__key__interpret(DB_WORKER, 'OUTDIR') 

        else:
            assert 0,(type(OUTDIR),OUTDIR)
                
#         BASENAME = os.path.basename(OUTDIR)
            
        if DB_WORKER['VERSION'] >= '20190607':
            OUTDIR = os.path.realpath(OUTDIR)



        DB_WORKER['OUTDIR'] = OUTDIR
        assert 'OUTDIR' in DB_WORKER
        assert isinstance(DB_WORKER['OUTDIR'],basestring),DB_WORKER['OUTDIR']

        DB_WORKER['RUNTIME']['MSG'] = {
            'CONTENT':'[_callbefore] starting [RUN_MODULE:"{RUN_MODULE}"] at [OUTDIR:"{OUTDIR}"]'.format(**DB_WORKER),
            'VERBOSITY':0 }
        
        
        
        if DB_WORKER['VERSION'] <'20190607':
            #### get mapper ,to-be-deprecated
            pass
            _lastDirIndex = pyext.dir__indexify(INPUTDIR,OPTS='-maxdepth 1')
            DB_WORKER['LAST_DIR_FILEACC2FULLPATH'] = _lastDirMapper = pyext.df__asMapper( _lastDirIndex,'FILEACC','FULL_PATH')
        
    _ = _main(**DB_WORKER)
    return DB_WORKER

_callbefore = _callbefore__0529


def _check_output(DB_WORKER):
    
    v = DB_WORKER['DB_SCRIPT']['OUTPUT_FILEACC']
    if isinstance(v, dict):
        v = v.values()
        
    assert isinstance(v,list), pyext.ppJson(DB_WORKER['DB_SCRIPT'])
    OUTPUT_FILE_LIST = v
    
    with _getPathStack([DB_WORKER['OUTDIR']],force=1,
                            printOnExit=DB_WORKER['RUNTIME']['printOnExit']):
        OUTPUT_SET = set(OUTPUT_FILE_LIST)
        if OUTPUT_SET:
            _notEmpty = True

            for FNAME in OUTPUT_SET:
                _notEmpty &= pyext.file__notEmpty(FNAME)
        else:
            _notEmpty = False
        
        DB_WORKER['RUNTIME']['_check_output']['allExist'] = _notEmpty
        if DB_WORKER['RUNTIME']['_check_output']['allExist']:
            if  (False \
                or (DB_WORKER).get('FORCE',False) \
                or (DB_WORKER['RUNTIME']).get('FORCE',False))==False:
                
                DB_WORKER['RUNTIME']['EXIT_MSG'] = {
            'CONTENT':'[_check_output] output exists at {OUTDIR} and lacks "DB_WORKER.FORCE=1", break call to {RUN_MODULE}'.format(**DB_WORKER),
            'VERBOSITY':0 }
                
#                 \
#                 ('[DB_WORKER] output exists at {OUTDIR} and lacks "DB_WORKER.FORCE=1", break call to {RUN_MODULE}'.format(**DB_WORKER))
#                 ('[DB_WORKER] output exists at {DB_WORKER["OUTDIR"]} and lacks "DB_WORKER.FORCE=1", break call to {DB_WORKER["RUN_MODULE"]}'.format(**locals()))
    return DB_WORKER

def worker__stepWithModule(DB_WORKER, module = None, **kw):
    if module is None:
        module = DB_WORKER['RUN_MODULE']
    assert module
    if isinstance(module,basestring):
        module = module.replace(':','.')
        module = importlib.import_module(module)
    else:
        pass
    try:
        reload(module)
    except Exception as e:
        print(e)
    
    _res = worker__step(DB_WORKER, 
                 this_func = module.THIS_FUNC, 
                 DB_SCRIPT = module.DB_SCRIPT,
                             **kw)
    DB_WORKER = _res
    return DB_WORKER

def worker__msg__verbose(DB_WORKER, CONTENT, VERBOSITY):

    _verbose = DB_WORKER['RUNTIME'].get('VERBOSE',0)
    if VERBOSITY <= _verbose:
        print(CONTENT)
        
def dbscript__validate(DB_SCRIPT, strict=0):
    d = DB_SCRIPT.copy()
    
#     if DB_WORKER['VERSION'] < '20190619':
    if 1:
        it = [
            ('PARAMS_TRACED', _DICT(), 'optional' ),
#             ('INPLACE', 1, 'optional' ),
        ]
#     else:
#         it = [
#             ('PARAMS_TRACED', list(), 'optional' ),
#         ]
    
    for k, DFT, required in it:
        
        if k not in d:
            if required!='optional':
                if strict:
                    raise Exception(k)
            d[k] = DFT
        else:
            v = d[k]
            if not isinstance(v, type(DFT)):
                d[k] = type(DFT)(v)
                
#         d[k] = d.get(k, DFT)
        
    return d


# def worker__key__interpret(DB_WORKER, key):
#     value = DB_WORKER[key]
#     if isinstance(value,basestring):
#         if value.startswith('!'):
#             value = value[1:]
#             value = pyext.template__format(value,
#                                            context = dict(DB_WORKER=DB_WORKER,**DB_WORKER))
#     DB_WORKER[key] = value
#     return value


# def worker__value__interpret(DB_WORKER, value):
#     if isinstance(value,basestring):
#         _type = type(value)
#         if value.startswith('!'):
#             value = value[1:]
#             value = pyext.template__format(value,
#                                            context = dict(DB_WORKER=DB_WORKER,**DB_WORKER))
#         value = _type(value)

#     return value
import simpleeval
import pymisca.header
def _template__format(template0, context = None, formatResult= False):
    template = template0
    
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
    
    vals = map(simpleeval.EvalWithCompoundTypes(
#     vals = map(simpleeval.SimpleEval(
        names=context,
        functions=functions).eval,exprs)
#     if vals
    _template = template0[1:]
    if _template.startswith('{') \
        and _template.endswith('}') \
        and len(vals)==1 \
        and not formatResult:
        res = vals[0]
    else:
        res = template.format(*vals)
    del context
    return res


def worker__value__interpret(DB_WORKER, value,formatter=None, formatResult=False):
    if formatter is None:
        formatter = _template__format
    if isinstance(value,basestring):
        _type = type(value)
        if value.startswith('!'):
            value = value[1:]
            value = formatter(value,context = dict(DB_WORKER=DB_WORKER,**DB_WORKER),
                             formatResult=formatResult)
            if isinstance(value,basestring):
                value = _type(value)

    return value

def worker__key__interpret(DB_WORKER, key):
    value = DB_WORKER[key]
    DB_WORKER[key] = value = worker__value__interpret(DB_WORKER,value)
    return value

def tree__worker__interpret(node, context):
    _this_func = tree__worker__interpret
    if isinstance(node,basestring):
        res = worker__value__interpret(context,node,formatResult=True)
    elif isinstance(node,list):
        res = [_this_func(_node,context) for _node in node ]
    elif isinstance(node,dict):
        context = context.copy()
        context.update(node)
        res = _DICT_CLASS([(_key, _this_func( _node, context)) 
                          for _key, _node in node.iteritems()])
    else:
        res = node
        pass
#         assert 0,(type(node),)
    return res
        
# def worker__interpret__param(DB_WORKER, 
def worker__step(DB_WORKER, this_func ,DB_SCRIPT,strict=0, VERSION=None, copy = True):
    if copy:
        DB_WORKER = DB_WORKER.copy()
    DB_WORKER['PWD'] = os.getcwdu()
    worker__key__interpret(DB_WORKER, 'INPUTDIR')                                
    pyext.dict__key__func( DB_WORKER, 'INPUTDIR', pyext.path__toSafePath)
#     DB_WORKER = kwargs = DB_WORKER.copy()    
#     DB_WORKER['INPUTDIR'] = pyext.path__toSafePath(DB_WORKER['INPUTDIR'])
    VERSION = VERSION or DB_WORKER.get('VERSION',VERSION)
    if VERSION is None:
        VERSION = datetime.date.today().strftime('%Y%m%d')        
    DB_WORKER['VERSION'] = VERSION

    DB_WORKER['DB_SCRIPT'] = dbscript__validate(DB_SCRIPT,strict=strict)
    DB_WORKER.setdefault('RUNTIME',_DICT())
    DB_WORKER.setdefault('RUN_PARAMS',_DICT())
#     DB_WORKER['RUNTIME'] = pymisca.tree.TreeDict( DB_WORKER.get('RUNTIME',{}) )
    DB_WORKER = pymisca.tree.TreeDict(DB_WORKER)
        

    
    for _func in [
        _callbefore,
        _check_output,
        this_func,
        _bookkeep,
    ]:
        #####
#         print(_func,
#               )
        DB_WORKER['_FUNC'] = _func
        
        msg = pyext.ppJson({k:str(v) for k,v in  DB_WORKER.items()})
        worker__msg__verbose(DB_WORKER, msg,2)
        
        _res = _func(DB_WORKER)
        
        if DB_WORKER['RUNTIME'].get('MSG',None):
            worker__msg__verbose(DB_WORKER, 
                                 **DB_WORKER['RUNTIME'].pop('MSG')
                                )
            
        if DB_WORKER['RUNTIME'].get('EXIT_MSG',{}):
            worker__msg__verbose(DB_WORKER, 
                                 **DB_WORKER['RUNTIME'].pop('EXIT_MSG')
                                )
#             if not DB_WORKER['RUNTIME']['SILENT']:
#                 print(DB_WORKER['RUNTIME']['EXIT_MSG'])
            break
#             return DB_WORKER
        
    DB_WORKER.pop('_FUNC',None)
    
    if VERSION >= '20190604':
        DB_WORKER['LAST_DIR'] = DB_WORKER.pop('OUTDIR',None)
#         DB_WORKER['LAST'] = DB_WORKER.pop('RUN_PARAMS',None)
        
    return DB_WORKER

def worker__safeStepWithModule(DB_WORKER,traceback_limit = 10,**kw):
    try:
        DB_WORKER = worker__stepWithModule(DB_WORKER,**kw)
        DB_WORKER['RUN_RESULT'] = {'SUCCESS':True,}
    except Exception as e:
#         try:
#             d.update(pyext.getErrorInfo())
# # #             d = {k:str(v) for k,v in d.items()}
#         except:
#             pass
#             d = {}
        d = _DICT_CLASS()
        d['SUCCESS'] = 0

        d['TRACEBACK_STACK'] = traceback.extract_tb(  sys.exc_info()[-1], limit=traceback_limit)
        d['ERR_MSG'] = str(e)
#     [::-1]
        DB_WORKER['RUN_RESULT'] = d
#         {'SUCCESS':False,'ERROR': d}
#         DB_WORKER['RUN_RESULCT'] = False
    finally:
        return DB_WORKER
    
def worker__prune(res):
    keys = [
#        'LAST_DIR',
#       'OUTDIR',
      'INPUTDIR',
      'RUN_MODULE',
      'RUN_PARAMS',
      'RUNTIME',
      'NCORE',
      'OUTDIR',
      'LAST_DIR'
                         ]
    keys = [k for k in keys if k in res]
#     if 'OUTDIR' in res:
#         keys += ['OUTDIR']
#     if 'LAST_DIR' in res:
#         keys += ['LAST_DIR']
    res = pyext.dict__getList(res,
                         keys)
    return res
def worker__archive(res):
        
    if 'OUTDIR' in res:
        res['LAST_DIR'] = res['OUTDIR']
    if 'LAST_DIR' in res:
        res['OUTDIR'] = res['LAST_DIR']
        
    res = worker__prune(res)
    #### limit the information that is preseved
    res.setdefault('NCORE',1)    
    _res = pyext.dict__getList(res,
                             ['LAST_DIR',
                              'NCORE',
                              'RUNTIME',])
    _res['LAST'] = res
    res.pop('LAST',None)
    return _res
    
def worker__step__reducer(l,r,**kw):
#     hasModule = lambda x: bool( (x or {}).get('LAST',{}).get('RUN_MODULE',{}) )
# #     hasModule = lambda x: bool(x) and bool(x.get('RUN_MODULE',None) or x.get('LAST',{}).get('RUN_MODULE'))
#     if not hasModule(l):
#         if not hasModule(r):
#             return None
#         else:
#             _l = pyext._DICT_CLASS()
#             if isinstance(l,dict):
#                 _l = _l.update(l)
#             l = _l
#     else:
#         print(repr(l))
            
    if l is None:
        if r is None:
            return None
        else:
            l = pyext._DICT_CLASS()
    l.update(r)
    
#     _last = l.pop('LAST',None)
#     print(pyext.ppJson(l))
#     l['LAST'] = _last
    
    res = _this_mod.worker__stepWithModule(l,**kw)
    res = worker__archive(res)

    
    
    return res




def worker__reduce__steps(DB_JOB,**kwargs):
    DB_JOB  = copy.copy(DB_JOB)

    if DB_JOB[0].get('RUN_MODULE',None):
        DB_JOB.insert(0,{})
    DB_JOB[0].update(kwargs)
        
    res =  reduce(pymisca.module_wrapper.worker__step__reducer, DB_JOB)    
    return res

# VERSION = 
worker__step__0528 = pyext.functools.partial(worker__step, VERSION='20190528')
# worker__step = worker__step__0528
# worker__stepWithModule = worker__stepWithModule__0528 
worker__stepWithModule__0528  = worker__stepWithModule
