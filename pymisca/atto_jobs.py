import pymisca.atto_string
import pymisca.tree
import pymisca.shell
import json

import os,shutil,sys
# from pymisca.shell import shellexec, file__notEmpty
_this_mod = sys.modules[__name__]

import imp
import pymisca.header

# import pymisca.atto_jobs
import collections

import pymisca.ptn
_jf2 = pymisca.ptn.jf2
# import pymisca.atto_jobs

import pymisca.date_extra
# import pymisca.bio
import filelock

from pymisca.atto_string import AttoJobResult
_getPathStack = pymisca.tree.getAttoDirectory

import collections
#### wrap shell
from pymisca import shell as _shell

import inspect
def cls__fullname(o):
  # o.__module__ + "." + o.__class__.__qualname__ is an example in
  # this context of H.L. Mencken's "neat, plausible, and wrong."
  # Python makes no guarantees as to whether the __module__ special
  # attribute is defined, so we take a more circumspect approach.
  # Alas, the module name is explicitly excluded from __qualname__
  # in Python 3.
  if not inspect.isclass(o):
    o = o.__class__
  module = o.__module__
  if module is None or module == str.__module__:
    return o.__name__  # Avoid reporting __builtin__
  else:
    return module + '.' + o.__name__


class ShellLogger(object):
    _parent = type('_NULL_PARENT',(object,),{})
    _jsonFile = None
    def __init__(self, atto_job, parent_shell=None):
        self._atto_job = atto_job
        self.logListLocal = []
        if parent_shell is not None:
            logLists = parent_shell.logLists[:]
        else:
            logLists = []
        self.logLists = logLists + [ self.logListLocal ] 
        
    def file__link(
            self,*a,**kw):
        return _shell.file__link(*a,**kw)
    
    def file__notEmpty(self,*a,**kw):
        return _shell.file__notEmpty(*a,**kw)
    
    def shellexec(
            self,
            CMD,
            silent=1,
            **kw):
        
        logLists = self.logLists
        cls_name = cls__fullname(self._atto_job)
#         .__class__.__name__
        
        tup = (cls_name,
               os.path.realpath(os.getcwd()),
               CMD)
        for logList in logLists:
            logList.append(tup)
        return _shell.shellexec(CMD,silent=silent,**kw)    
    
    def setJsonFile(self,jsonFile):
        self._jsonFile = jsonFile
        
    def loadCmd__fromJson(self, OUTPUT_FILE=None):
        OUTPUT_FILE = self._jsonFile if OUTPUT_FILE is None else OUTPUT_FILE
        assert OUTPUT_FILE 
        
        if self.file__notEmpty( OUTPUT_FILE ):
            with open(OUTPUT_FILE,"r") as f:
                res = json.load(f, object_pairs_hook=collections.OrderedDict)
        else:
            res = []
        [ x.extend(res) for x in self.logLists ]
#         self.logListLocal[:] = res
        return res
        
    
    def dumpCmd__asBash(self, OUTPUT_FILE):
        with open(OUTPUT_FILE, "w") as f:
            for i,(cls_name, cwd, cmd) in enumerate(self.logListLocal):
                lst = [
                    u'#'*30,'\n',
                    u'### [step:%05d]\n'%i,
                    u'### [shell._atto_job.class_name]:%s\n' % cls_name,
                    u'### [shell.wkdir]:\n',
                    u'mkdir -p %s\n' % cwd, ### temporary hack
                    u'cd %s\n' % cwd,
                    u'### [shell.cmd]:\n',
                    u'%s\n'%cmd,
                    u'\n',
                ]
                map(f.write,lst)
                
    def dumpCmd__asJson(self, OUTPUT_FILE=None):
        OUTPUT_FILE = self._jsonFile if OUTPUT_FILE is None else OUTPUT_FILE
        assert OUTPUT_FILE 
        
        tups = self.logListLocal
        with open(OUTPUT_FILE, "w") as f:
            json.dump( tups, f, indent=4 )
        self.dumpCmd__asBash( OUTPUT_FILE+".sh")
                
            
class _NULL_PARENT(object):
    shell=None
    pass
class AttoJob(pymisca.atto_string.AttoCaster, ):
    _parent = _NULL_PARENT;
#     _parent = type('_NULL_PARENT',(object,),{"shell":None})
    
#     _shellexec =  attoJob__dec__shellexec(pymisca.shell.shellexec)
#     _parent = 
#     PARAMS_TRACED = 



    @classmethod
    def _lock( cls, BASENAME=None):
        if BASENAME is None:
            BASENAME = 'RUN'
        return filelock.FileLock(BASENAME + '.lock')
    
    def __init__(self, kw):
        self._result = None
        kw = self._DICT_CLASS(kw)
        kw = self._cast(kw)
        self._data =  kw
        
        self._parent = kw.get('PARENT',self.__class__._parent)
        self.shell = ShellLogger( self, self._parent.shell)
        
        with pymisca.date_extra.ScopeTimer(
            data=self._data,
            key='TIME_DICT',) as timer:
            
            # timer.update(self._run)
            self._run()

        if "OUTDIR" in self._data:
            self._data["LAST_DIR"] = self._data.pop("OUTDIR")



class ModuleJob(AttoJob):
    PARAMS_TRACED = [
        ('MODULE_FILE',('AttoPath','')),
        ('MODULE_ATTR',('AttoPath','')),
        ('OUTPUT_BASH_SCRIPT',('AttoPath','')),
        ('DATA',('dict:object:object:object:object:object',{}))
    ]
    def _run(self,):
        kw = self._data
        for key in ["MODULE_FILE",
#                     "MODULE_ATTR"
                   ]:
            assert kw[key],("Must specify %s"%key,)
        kw['MODULE_FILE'] = MODULE_FILE = kw['MODULE_FILE'].realpath()
        if not kw['MODULE_ATTR']:
            dft = MODULE_FILE.rsplit('/')[-1].rsplit('.')[0]
            kw['MODULE_ATTR'] = type(kw['MODULE_ATTR'])( dft )
        MODULE_ATTR = kw['MODULE_ATTR'];

        
        MODULE_NAME = None
        if MODULE_NAME is None:
            MODULE_NAME = pymisca.header.get__anonModuleName()
        mod = pymisca.atto_jobs.AttoModule({
            "NAME":MODULE_NAME,
            "INPUT_FILE":MODULE_FILE})['MODULE']
        cls = getattr(mod, MODULE_ATTR)
        
        kw['DATA_RESULT'] = res = cls(dict(PARENT=self, **kw['DATA']))
        if kw['OUTPUT_BASH_SCRIPT']:
            self.shell.dumpCmd__asBash(kw['OUTPUT_BASH_SCRIPT'].realpath())
#             self.shell.dumpCmd__asJson()
#             setJsonFile()

@pymisca.header.setAttr(_this_mod, "dicts__toSeries")
class dicts__toDict(AttoJob):
    PARAMS_TRACED = [
        ('DATA_LIST',("list:dict:AttoPath",[])),
        ('OFNAME',("AttoPath",'')),
    ]
    def _run(self):
        kw = self._data
        kw['OUTPUT'] = collections.OrderedDict()
        for d in kw['DATA_LIST']:
            kw['OUTPUT'][d['NAME']]=d['VALUE']
        if kw['OFNAME']:
            pymisca.shell.real__dir(fname=kw['OFNAME'])
#             d = kw['OUTPUT'].to_dict()
            d = kw['OUTPUT']
            with open(kw['OFNAME'],'w') as f:
                json.dump(d,f,indent=4)
            
        return             

class bam__toBigWig(_this_mod.AttoJob):
    PARAMS_TRACED = [
#         ('MODULES',('dict:object', None)),
        ('MODULE',('dict:object',None)),
        ('INPUT_FILE',('AttoPath',None)),
        ('OFNAME',('AttoPath',None)),
        ('FORCE',('int',0)),
    ]
    def _run(self):
        kw = self._data
        if not kw.get("MODULE",{}):
            kw['MODULE'] = pymisca.header.get__defaultModuleDict()['bcbio.bam2bw']
        self._work(None,None,**kw)
        
    @staticmethod
    def _work(self, key, INPUT_FILE, OFNAME, MODULE, FORCE):
        mod = MODULE['MODULE']
        if not FORCE and pymisca.shell.file__notEmpty(OFNAME):
            pass
        else:
            if os.path.isfile(OFNAME):
                os.remove(OFNAME)
            mod.main(INPUT_FILE,outfile=OFNAME)
        return OFNAME
        
class bamList__merge( AttoJob):
    PARAMS_TRACED = [
        ('INPUT_FILES',('list:AttoPath',None)),
        ('OFNAME',('AttoPath',None)),
        ('FORCE',('int',0)),
        ('NCORE',('int',1)),
    ]
    
    def _outNode(self):
        return self._data['OFNAME']
        
    def _run(self):
        kw = self._data
        INPUT_FILES = kw['INPUT_FILES']
        FORCE = kw['FORCE']
        NCORE = kw['NCORE']
        OFNAME = kw['OFNAME']

        pymisca.shell.real__dir(fname=OFNAME)
        if not FORCE and pymisca.shell.file__notEmpty(OFNAME):
            return 'SKIP'
        else:
            CMD  = '''
            samtools merge -f --threads {{NCORE}} {{OFNAME}} {{' '.join(INPUT_FILES)}} 
            samtools index -@ {{NCORE}} {{OFNAME}} {{OFNAME}}.bai        
            '''
            self._data['CMDS'] = CMDS = _jf2(CMD).splitlines()
#             .format(**locals()).splitlines()
            res = map(pymisca.shell.shellexec, CMDS)
            return 'FINISH'    
    
    
class bamFile__toFastq(AttoJob):
    PARAMS_TRACED=[
        ('INPUT_FILE',('AttoPath',None)),
        ('OUTDIR',('AttoPath','')),
#         ('OUTPUT_FILE',('list:AttoPath',[])),
        ('OPTS_LIST',('list:AttoPath',['-f','0x4']
                     )),
        ('IS_PAIRED',('int',0)),
        ('DRY',('int',0)),
        ('FORCE',('int',0)),
        
    ]
    def _run(self):
        kw = self._data
        kw['INPUT_FILE'] = INPUT_FILE = kw['INPUT_FILE'].realpath()
#         kw['OUTPUT_FILE'] = OUTPUT_FILE = \
#                 kw['OUTPUT_FILE'] / ( INPUT_FILE.rsplit('.',1)[0] + '.fastq') \
#                 if not kw['OUTPUT_FILE'] \
#                 else kw['OUTPUT_FILE'].realpath()
        kw['OUTDIR'] = OUTDIR = \
            INPUT_FILE.dirname() \
            if not kw['OUTDIR'] \
            else kw['OUTDIR'].realpath()
    
        kw['OFNAME'] = OFNAME = INPUT_FILE.basename().rsplit('.',1)[0] + '.fastq'
        
        OPTS_LIST = kw['OPTS_LIST']
        IS_PAIRED = kw['IS_PAIRED']
        DRY = kw['DRY']
        FORCE = kw['FORCE']
        
        _LIST = ['samtools','bam2fq']
        if IS_PAIRED:
            assert 0,('Not implemented',)

        else:
            if '-0' not in _LIST:
                _LIST += ['-0',  OFNAME+'.partial',
        #                   '-s','RS.fastq'
                         ] 
        _LIST += OPTS_LIST
        _LIST += [ INPUT_FILE ]                
        CMD = ' '.join(_LIST)
        
        with pymisca.tree.getAttoDirectory([OUTDIR],force=1):
            if not FORCE and pymisca.shell.file__notEmpty(OFNAME):
                return "SKIP"
            else:
                if DRY:
                    return "RUN"
                else:
                    with self._lock(OFNAME):
                        res = pymisca.shell.shellexec(CMD)
                        shutil.move(OFNAME+'.partial', OFNAME )
#                 pass
##### [TBC] how to merge nodes to create bigger nodes?        
class bamFile__filter(AttoJob):
    PARAMS_TRACED = [
        ("INPUT_FILE",("AttoPath",None)),
        ("OUTDIR",("AttoPath",None)),
#         ("SAMTOOLS_FLAG_LIST",("list:unicode",["-F0x4","-F0x100",]))
        ("SAMTOOLS_FLAG_LIST",("list:unicode",[])),
        ("SORTED",("int",1)),
        ('FORCE', ('int', 0)),
        ('DRY', ('int', 0)),
        ('NCORE', ('int', 1)),
        
    ]
    
    def _run(self,):
        kw = self._data
        assert kw['OUTDIR']
        kw['OUTDIR'] = OUTDIR = kw['OUTDIR'].realpath()
        assert kw["SAMTOOLS_FLAG_LIST"]
        INPUT_FILE = kw['INPUT_FILE']
        FORCE = kw['FORCE']
        DRY = kw['DRY']
        NCORE = kw['NCORE']
        SORTED = kw['SORTED']
        kw['NODES']= NODES = []
        node = Shellexec({
            "OUTDIR":OUTDIR,
            "CMD_LIST":["samtools","view","-bh",kw["SAMTOOLS_FLAG_LIST"],
                                  [INPUT_FILE,],
                                  [">ALIGNMENT.bam",],
                                  ["&&","echo","DONT-COUNT",">COUNTS.json"],
                                  ],
            "OFNAME_LIST":["ALIGNMENT.bam"],
            "FORCE":FORCE,
            "DRY":DRY,
            "NCORE":NCORE,
            })
        NODES.append(node)
        
        node = job__samtools__qc({
            "INPUT_FILE":node["LAST_DIR"] / "ALIGNMENT.bam",
            "OUTDIR":node["LAST_DIR"],
            "SORTED":SORTED,
            "NCORE":NCORE,
            "FORCE":FORCE,
            "DRY":DRY,
        })
        NODES.append(node)    
    
class bamFile__toBigwig(AttoJob):
    PARAMS_TRACED  = [
        ('INPUT_FILE',('AttoPath','')),
        ('OUTDIR',('AttoPath','')),
        ('FORCE',('int',0)),
        ('DRY',('int',0)),
        ('SCALE_LOG10',('float',0.)),
        ('FILTER_FLAG_DICT',('dict:list:AttoPath',
                             {
                                 'FWD.bigwig':['-F','0x10','-F','0x100','-F','0x4'],
                                 'REV.bigwig':['-f','0x10','-F','0x100','-F','0x4'],
                                 'MAPPED.bigwig':['-F','0x100','-F','0x4'],
                             }
                            )),
    ]

    def _export(self):
        kw = self._data
        out = []
        OUTDIR = kw.get('LAST_DIR') or kw['OUTDIR']
        DRY = kw['DRY']
        
        for key in self._data['FILTER_FLAG_DICT']:
#             OUT_BIGWIG = OUTDIR / '%s.bigwig' % key
            OUT_BIGWIG = OUTDIR / key
            if not DRY:
                assert pymisca.shell.file__notEmpty(OUT_BIGWIG),(OUT_BIGWIG,)
            out += [(OUT_BIGWIG, key )]
        return out
    
    def _run(self):
        kw = self._data
        FILTER_FLAG_DICT = kw['FILTER_FLAG_DICT']
        kw['INPUT_FILE'] = INPUT_FILE =  kw['INPUT_FILE'].realpath()
        OUTDIR = kw['OUTDIR']
        DRY = kw['DRY']
        FORCE = kw['FORCE']
        SCALE_LOG10 = kw['SCALE_LOG10']
        
        def _worker( (key, FILTER_FLAG) ):
            TEMP_BAM =  '%s.temp.bam' % key
            OUT_BIGWIG = key
#             OUT_BIGWIG = '%s.bigwig' % key
            if not FORCE and pymisca.shell.file__notEmpty(OUT_BIGWIG):
                return "SKIP"
            else:
                if DRY:
                    return "RUN"      
                else:
                    lst = ["samtools","view","-bS", FILTER_FLAG , INPUT_FILE , ">", TEMP_BAM, 
                           "&&" , "samtools", "index", TEMP_BAM , 
                           "&&", "bam2bw", 
                           ["--scale-log10",str(SCALE_LOG10)] if SCALE_LOG10 else [],  ###wrong version
#                            ["--scale_log10",str(SCALE_LOG10)] if SCALE_LOG10 else [], 
                           "-i", TEMP_BAM, "-o", OUT_BIGWIG +'.partial',
                           "&&", "rm", "-f", TEMP_BAM, TEMP_BAM+'.bai',
                           "&&", "mv", OUT_BIGWIG +'.partial', OUT_BIGWIG,
                          ]
                    cmd = []
                    for x in lst:
                        if isinstance(x,basestring):
                            cmd.append(x)
                        else:
                            cmd.extend(x)
                    CMD = " ".join(cmd)
                    res = pymisca.shell.shellexec(CMD)
                    return "RUN"
        
        with pymisca.tree.getAttoDirectory([OUTDIR],force=1):
            with self._lock() as lock:
                it = FILTER_FLAG_DICT.items()
                res = map(_worker, it)
                return res

    
    
class AttoModule(AttoJob):
    PARAMS_TRACED = [
        ('NAME',('AttoPath',None)),
        ('INPUT_FILE',('AttoPath',None)),
        ('VERBOSE',('int',0)),
        
    ]
    def _run(self):
        kw = self._data
        NAME = kw['NAME']
        kw['INPUT_FILE'] = INPUT_FILE = kw['INPUT_FILE'].realpath()
        VERBOSE = kw["VERBOSE"]
        SILENT = 1 - VERBOSE
        
        ###### check the loaded module has the same path as self
        _mod  = sys.modules.get(NAME,None)
        if _mod is not None and \
            hasattr( _mod, '__file__'):
            OLD_FILE = os.path.realpath( _mod.__file__ )
            #### consider .py -> .pyc
            assert INPUT_FILE in [OLD_FILE, OLD_FILE[:-1] ], json.dumps(( str(NAME), OLD_FILE, INPUT_FILE),indent=4)
            

        if INPUT_FILE.endswith('.py'):
            with pymisca.header.Suppress(SILENT,SILENT):
                kw['MODULE'] = imp.load_source(NAME,INPUT_FILE)
        elif INPUT_FILE.endswith('.pyc'):
            with pymisca.header.Suppress(SILENT,SILENT):
                kw['MODULE'] = imp.load_compiled(NAME,INPUT_FILE)
        else:
            assert 0,("dont know how to load module file:",INPUT_FILE)
        return
    
            
class Cleaner(AttoJob):
    PARAMS_TRACED = [
        ('INPUTDIR',('AttoPath', None)),
        ('OUTDIR',('AttoPath', None)),
        ('LAST_DIR',('AttoPath', None)),
#         ('SORTED',('int',0)),
        ('FORCE',('int',0)),
        ('NCORE',('int',1)),
        ('VERBOSE',('int',0)),
    ]
    def __init__(self,kw):
# #         print kw
#         sys.stderr.write(json.dumps(kw,indent=4)+'/n')
        super(Cleaner,self).__init__(kw)
        
    def _run(self,):
        kw = self._data
        kw["OUTDIR"] = kw["OUTDIR"].abspath()

        with pymisca.tree.getAttoDirectory([kw["OUTDIR"]],force=1) as stack:
            if kw['VERBOSE'] >= 1:
                sys.stderr.write(str(("cleaning",stack.d))+'/n')

            with self._lock() as lock:
                ### https://stackoverflow.com/a/1073382/8083313
                for root, dirs, files in os.walk('.'):
                    for f in files:
                        os.unlink(os.path.join(root, f))
                    for d in dirs:
                        shutil.rmtree(os.path.join(root, d))    
class Shellexec(AttoJob):
    PARAMS_TRACED = [
        ('LAST_JOB',('AttoCaster',None)),
#         ('INPUTDIR',('AttoPath',None)),
        ('OUTDIR',('AttoPath',None)),
        ('CMD_LIST',('list:object:object:object',[])),
        ('OFNAME_LIST',('list:AttoPath',[])),
        ('FORCE',('int',0)),
        ('DRY',('int',0)),
    ]
    def _run(self):
        kw = self._data
        kw['OUTDIR'] = OUTDIR = kw['OUTDIR'].realpath()
        kw['CMD_LIST'] = CMD_LIST = pymisca.header.stringList__flatten(kw['CMD_LIST'])

#         kw['INPUTDIR'] = INPUTDIR = kw['INPUTDIR'].realpath()
        
        with _getPathStack([OUTDIR],force=1):
            with self._lock():
                if not kw['FORCE'] and kw['OFNAME_LIST'] \
                and all(map(pymisca.shell.file__notEmpty,kw['OFNAME_LIST'])):
                    return "SKIPPED"
                else:
                    if kw['DRY']:
                        pass
                    else:
                        assert len(CMD_LIST),(CMD_LIST,)
                        CMD = ' '.join(CMD_LIST)
                        pymisca.shell.shellexec(CMD)
                        
                    return "RUNNED"
###############                
##### do not over-subclassing
# class bamFile__uniqMapped(Shellexec):
#     def _run(self):
#         kw = self._data
#         LAST_JOB = kw["LAST_JOB"]
#         BAM_BASENAME = kw.get("BAM_BASENAME", "ALIGNMENT-sorted.bam")
#         kw["RUN_PARAMS"] = kw["LAST_JOB"].get("RUN_PARAMS",{})
#         kw["OFNAME_LIST"] = [BAM_BASENAME]
#         kw["CMD_LIST"] = [['samtools',
#                           ['view','-bh','-F0x4','-F0x100',],
#                           LAST_JOB['LAST_DIR']/BAM_BASENAME,
#                           ">%s.partial"%BAM_BASENAME],
#                           "&&",["mv","%s.partial"%BAM_FILE, BAM_FILE]]
#         super(bamFile__uniqMapped,self)._run()
        
                    
# class job__samtools__qc(pymisca.atto_string.AttoCaster):
class job__samtools__qc(AttoJob):
    PARAMS_TRACED = [
        ('INPUT_FILE',('AttoPath', None)),
        ('OUTDIR',('AttoPath', None)),
        ('LAST_DIR',('AttoPath', None)),
        ('SORTED',('int',0)),
        ('FORCE',('int',0)),
        ('NCORE',('int',1)),
    ]

#     def __init__(self,kw):
#         self._result = None
#         kw = self._DICT_CLASS(kw)
#         kw = self._cast(kw)
#         self._data =  kw
#         self._run()
#         self._data["LAST_DIR"] = self._data.pop("OUTDIR")
        
    def _run(self,):
        shell = self.shell
        shellexec = shell.shellexec
        
        kw = self._data
        kw['INPUT_FILE']  = INPUT_FILE = kw["INPUT_FILE"].realpath()
        assert shell.file__notEmpty(INPUT_FILE),(INPUT_FILE,)
        FORCE = kw['FORCE']
#         shell.setCacheJson(kw["OUTDIR"]/INPUT_FILE.basename())
        ### populate result
#         self._result = self._data.copy()
#         self["TIME_DICT"] = pyext._DICT_CLASS()
#         shellexec = pyext.shellexec
#         shellexec = pymisca.shell.shellexec
        
    
        with pymisca.tree.getAttoDirectory([kw["OUTDIR"]],force=1):
            with self._lock() as lock:
                with self._timer(OFNAME="TIME_DICT.json") as timer:
                    with pymisca.tree.getAttoDirectory(['.temp'],force=1):
                        pass
                    
                    if not kw['SORTED']:
                        BASENAME = INPUT_FILE
                        OFNAME = 'ALIGNMENT.bam'
                        if not FORCE and shell.file__notEmpty(OFNAME):
                            pass
                        else:
                            shell.file__link( BASENAME,OFNAME,force=1)

                        BASENAME = "ALIGNMENT.bam"
                        OFNAME = 'ALIGNMENT-sorted.bam'
                        if not FORCE and shell.file__notEmpty(OFNAME):
                            pass
                        else:
                            shell.shellexec("samtools sort -o {OFNAME}.temp -T .temp/$$ {BASENAME} \
                            && mv {OFNAME}.temp {OFNAME}".format(**locals()) )
                    else:
                        BASENAME = INPUT_FILE
                        OFNAME = 'ALIGNMENT-sorted.bam'
                        if not FORCE and shell.file__notEmpty(OFNAME):
                            pass
                        else:
                            shell.file__link( BASENAME,OFNAME,force=1)

                    BASENAME = "ALIGNMENT-sorted.bam"
                    OFNAME = 'ALIGNMENT-sorted.bam.bai'
                    if not FORCE and shell.file__notEmpty(OFNAME):
                        pass
                    else:
                        shell.shellexec(("samtools index {BASENAME} {OFNAME}.temp \
                        && mv {OFNAME}.temp {OFNAME}").format(**locals()))

                    OFNAME = "COUNTS.json"
                    DATA_DICT = self._data
                    if not FORCE and shell.file__notEmpty(OFNAME):
                        pass
                    else:

                        DATA_DICT['counts'] = self._DICT_CLASS()
                        DATA_DICT['counts']['UNMAPPED'] = int(shellexec('samtools view -c -f 0x4 ALIGNMENT-sorted.bam').strip())
                        DATA_DICT['counts']['FWD'] = int(shellexec('samtools view -c -F 0x10 -F 0x100 -F 0x4 ALIGNMENT-sorted.bam').strip())
                        DATA_DICT['counts']['REV'] = int(shellexec('samtools view -c -f 0x10 -F 0x100 -F 0x4 ALIGNMENT-sorted.bam').strip())
                        DATA_DICT['counts']['SUM'] = DATA_DICT['counts']['FWD'] + DATA_DICT['counts']['REV']
                        DATA_DICT['counts']['MAPPED'] = DATA_DICT['counts']['SUM']
#                         DATA_DICT['texts'] = pyext._DICT_CLASS()
#                         DATA_DICT['texts']['flagstat'] = shellexec('samtools flagstat ALIGNMENT-sorted.bam')
#                         DATA_DICT['texts']['quickcheck'] = shellexec('samtools quickcheck ALIGNMENT-sorted.bam 2>&1')
                        with open(OFNAME,'w') as f:
                            json.dump(DATA_DICT['counts'], f,indent=4)
            
                    self.shell.setJsonFile("%s.CMD.json"%self.__class__.__name__)
                    if self.shell.logListLocal:
                        self.shell.dumpCmd__asJson()
                    else:
                        self.shell.loadCmd__fromJson()

        self._data=  self._cast( self._data )
        return 
    

