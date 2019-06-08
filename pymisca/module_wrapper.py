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
# import fastqDB.ptn
_DICT_CLASS  =_DICT = pyext.collections.OrderedDict
def basename__toDict(full_name,VERSION):
    bname = os.path.basename(full_name)
    if VERSION >= '20190608':
        d = pymisca.atto_string.AttoString.new(bname).toContainer()
    else:
        s = pymisca.ptn.WrapString(os.path.basename(bname))
        d = s.toDict()
    return d


def _bookkeep(DB_WORKER):
    def _main(OUTDIR,**kw):
        with pyext.getPathStack([OUTDIR],force=1,
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
    return DB_WORKER


def getDefault__OUTDIR(DB_WORKER):
    if DB_WORKER['VERSION'] <= '20190528':
        OUTDIR = pymisca.blocks.compute__OUTDIR(
            inplace = DB_WORKER.get('INPLACE',1),
            pathLevel = DB_WORKER.get('pathLevel',None),
            INPUTDIR=DB_WORKER['INPUTDIR'], baseFile=0,
            SELFALI=DB_WORKER['DB_SCRIPT']['MODULE_NAME'].rsplit('.',1)[-1] 
        )()
    else:
#     elif DB_WORKER['VERSION'] <= '20190607':
        DB_SCRIPT = _DICT(DB_WORKER['DB_SCRIPT'])
        
        ### get Non-default parameters
        for k,v in DB_SCRIPT['PARAMS_TRACED'].items():
            if k in DB_WORKER['RUN_PARAMS']:
                if DB_WORKER['RUN_PARAMS'][k] == v:
                    del DB_SCRIPT['PARAMS_TRACED'][k]

            else:
                DB_WORKER['RUN_PARAMS'][k] = DB_SCRIPT['PARAMS_TRACED'].pop(k)
                
                
        
                
        pyext.dict__key__func(DB_WORKER, 'INPUTDIR', pymisca.atto_util.AttoPath)
        
        d = _DICT(RUN_MODULE=DB_WORKER['RUN_MODULE'])
        d.update(pyext.dictGetList(
                DB_WORKER['RUN_PARAMS'],
                DB_SCRIPT['PARAMS_TRACED'].keys()
                  )
                )
        
        if DB_WORKER['VERSION'] >= '20180608':
            SELFALI = pymisca.atto_string.AttoString.fromContainer(d)
        else:
            SELFALI = pyext.WrapString.fromDict(d,VERSION=DB_WORKER['VERSION'])
#         print(SELFALI)
        
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
        
        
        if OUTDIR is None:
            OUTDIR = getDefault__OUTDIR(DB_WORKER)
            
        else:
            if callable(OUTDIR):
                OUTDIR = OUTDIR(DB_WORKER)
            elif isinstance(OUTDIR,basestring):
                DB_WORKER['OUTDIR'] = OUTDIR
                OUTDIR = worker__key__interpret(DB_WORKER, 'OUTDIR') 
                
#             elif isinstance(OUTDIR,basestring):
#                 OUTDIR = OUTDIR.lstrip('!')
#     #             DB_WORKER['OUTDIR'] = DB_WORKER['OUTDIR'].format(DB_WORKER=DB_WORKER,DB_SCRIPT=DB_WORKER['DB_SCRIPT'])
#                 OUTDIR = pyext.template__format(OUTDIR,
#                                                              context = dict(DB_WORKER=DB_WORKER,**DB_WORKER))
            else:
                assert 0,(type(OUTDIR),OUTDIR)
            
        if DB_WORKER['VERSION'] >= '20190607':
            OUTDIR = os.path.realpath(OUTDIR)



        DB_WORKER['OUTDIR'] = OUTDIR
        assert 'OUTDIR' in DB_WORKER
        assert isinstance(DB_WORKER['OUTDIR'],basestring),DB_WORKER['OUTDIR']

        DB_WORKER['RUNTIME']['MSG'] = {
            'CONTENT':'[_callbefore] starting [RUN_MODULE:"{RUN_MODULE}"] at [OUTDIR:"{OUTDIR}"]'.format(**DB_WORKER),
            'VERBOSITY':1 }
        
        
        
        if DB_WORKER['VERSION'] <'20190607':
            #### get mapper ,to-be-deprecated
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
    
    with pyext.getPathStack([DB_WORKER['OUTDIR']],force=1,
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
                
                DB_WORKER['RUNTIME']['EXIT_MSG'] = \
                ('[DB_WORKER] output exists at {OUTDIR} and lacks "DB_WORKER.FORCE=1", break call to {RUN_MODULE}'.format(**DB_WORKER))
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
    
    it = [
        ('PARAMS_TRACED', _DICT(), 'optional' ),
    ]
    
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


def worker__key__interpret(DB_WORKER, key):
    value = DB_WORKER[key]
    if isinstance(value,basestring):
        if value.startswith('!'):
            value = value[1:]
            value = pyext.template__format(value,
                                           context = dict(DB_WORKER=DB_WORKER,**DB_WORKER))
    DB_WORKER[key] = value
    return value
        
# def worker__interpret__param(DB_WORKER, 
def worker__step(DB_WORKER, this_func ,DB_SCRIPT,strict=0, VERSION=None, copy = True):
    if copy:
        DB_WORKER = DB_WORKER.copy()
    DB_WORKER['PWD'] = os.getcwdu()
    worker__key__interpret(DB_WORKER,'INPUTDIR')                                
    pyext.dict__key__func( DB_WORKER, 'INPUTDIR', pyext.path__toSafePath)
#     DB_WORKER = kwargs = DB_WORKER.copy()    
#     DB_WORKER['INPUTDIR'] = pyext.path__toSafePath(DB_WORKER['INPUTDIR'])
    DB_WORKER['DB_SCRIPT'] = dbscript__validate(DB_SCRIPT,strict=strict)
    DB_WORKER.setdefault('RUNTIME',_DICT())
    DB_WORKER.setdefault('RUN_PARAMS',_DICT())
    DB_WORKER = pymisca.tree.TreeDict(DB_WORKER)
    if VERSION is None:
        VERSION = datetime.date.today().strftime('%Y%m%d')        
        
    DB_WORKER['VERSION'] = VERSION

    
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
            if not DB_WORKER['RUNTIME']['SILENT']:
                print(DB_WORKER['RUNTIME']['EXIT_MSG'])
            break
#             return DB_WORKER
        

    if VERSION >= '20190604':
        DB_WORKER['LAST_DIR'] = DB_WORKER.pop('OUTDIR',None)
        
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

# VERSION = 
worker__step__0528 = pyext.functools.partial(worker__step, VERSION='20190528')
# worker__step = worker__step__0528
# worker__stepWithModule = worker__stepWithModule__0528 
worker__stepWithModule__0528  = worker__stepWithModule