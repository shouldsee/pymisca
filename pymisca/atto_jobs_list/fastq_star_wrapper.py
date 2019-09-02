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


import pymisca.atto_jobs
from pymisca.atto_job import AttoJob
_getPathStack = pymisca.tree.getAttoDirectory
from pymisca.atto_jobs import bamFile__toFastq

import json
# from pymisca.header import ppJson,dppJson
# @pymisca.header.setAttr("MAIN")
# class fastq__alignWithStar(AttoJob):
# class fastq__alignWithStar(AttoJob):
class fastq_star_wrapper(AttoJob):
    PARAMS_TRACED = [
       ('INPUTDIR',('AttoPath','')),
       ('OUTDIR',('AttoPath','')),
       ('OPTS', ('list:AttoPath',[])),
       ('RAM_BAMSORT',('int',25E9)),
       ('GENOMEDIR',('AttoPath',None)),
       ('FORCE',('int',0)),
       ('DRY',('int',0)),
       ('NCORE',('int',1)),
    ]

    def _run(self):
        shell  =self.shell
        kw = self._data
        kw['INPUTDIR'] = INPUTDIR = kw['INPUTDIR'].realpath()
        assert kw['OUTDIR']
        kw['OUTDIR'] = OUTDIR = kw['OUTDIR'].realpath()
        OPTS = kw['OPTS']
        RAM_BAMSORT = kw['RAM_BAMSORT']
        GENOMEDIR = kw['GENOMEDIR'].realpath()
        assert GENOMEDIR.isdir(),(GENOMEDIR,)
        FORCE = kw['FORCE']
        DRY = kw['DRY']
        NCORE = kw['NCORE']
        
        ####
        
        
        _LST = ['STAR']
        _LST += OPTS

        #### [extra_casting] GENOMEDIR -> DIR_STAR_INDEX
        if GENOMEDIR.endswith('STAR_INDEX'):
            DIR_STAR_INDEX = GENOMEDIR
            pass
        else:
            res = GENOMEDIR.glob("*STAR_INDEX/")
            assert len(res)==1,(GENOMEDIR,)
#             GENOMEDIR = res[0]
            DIR_STAR_INDEX = res[0]
#         DIR_STAR_INDEX = GENOMEDIR
            
        with _getPathStack([ DIR_STAR_INDEX ],force=0) as stack:
            for suff in ['Genome','SA','SAindex']:
                _f = ( stack.d / suff )
                assert pymisca.shell.file__notEmpty(_f),('STAR_INDEX/ folder is broken', _f,)
                
            ### flagfile for GTF existence
            if pymisca.shell.file__notEmpty( stack.d / "geneInfo.tab"):
                _LST += ['--quantMode', 'TranscriptomeSAM',]
        
        #### casting LST_FILES
        with _getPathStack([INPUTDIR]) as stack:
            INPUTs = []
            if not INPUTs:
                INPUTs = sorted(stack.d.glob("*.fastq"))
#                 assert len(INPUTs) in [1,2],(INPUTDIR, INPUTs)
            if not INPUTs:
                FNAME = stack.d / ("ALIGNMENT-sorted.bam")
                assert pymisca.shell.file__notEmpty(FNAME),(FNAME,)
                def INPUTs(stack=stack,FNAME=FNAME):
                    ###### [TBC] "needs to remove this node explicitly"
                    toRemove = res = bamFile__toFastq({
                    "INPUT_FILE":FNAME,
                    "OUTDIR": (stack.d/"UNMAPPED"),
                    "OPTS_LIST":["-f","0x4"],
                })
                    FILES = sorted(res["LAST_DIR"].glob("*.fastq"))
                    assert FILES,(res._data["LAST_DIR"],"*.fastq" )

                    return FILES
                    
#                 INPUTs = lambda : bamFile__toFastq({
#                     "INPUT_FILE":FNAME,
#                     "OUTDIR": (stack.d/"UNMAPPED"),
#                     "OPTS_LIST":["-f","0x4"],
#                 })["LAST_DIR"].glob("*.fastq")
                
            if not callable(INPUTs):
                assert pymisca.shell.file__notEmpty( INPUTs[0] ), (INPUTDIR,INPUTs)
#             LST_FILES = sorted(INPUTs)
            LST_FILES = INPUTs
            
            
        DIR_TEMP = 'TEMP'
        DEFAULT_ARGS = [
            '--runMode',['alignReads',],
            '--runThreadN',[str(NCORE),],
    #         '--outSAMtype',['BAM','Unsorted'],
           '--outSAMtype',['BAM','SortedByCoordinate'],  # STAR crashes when sorting bam

            '--outWigType',['wiggle'],
            '--outWigStrand',['Stranded'],
    #         '--outWigNorm',['RPM'],
            '--outWigNorm',['None'],

            '--outFilterMismatchNmax',[str(2),],
            '--outFilterIntronMotifs', ['RemoveNoncanonicalUnannotated',],
            '--outMultimapperOrder',['Random',],
            '--genomeLoad', ['NoSharedMemory',],        
            '--outFileNamePrefix',['./',],
            '--outReadsUnmapped',['Fastx',],
            '--limitBAMsortRAM', ['%d'%RAM_BAMSORT,],
            '--readFilesIn', [LST_FILES],
            '--genomeDir',[DIR_STAR_INDEX],
            '--outTmpDir',[DIR_TEMP,],  ###[TBC]
            '--outSAMunmapped',['Within',],
            '--seedSplitMin',['15',],
        ]            
        
        DEFAULT_ARGS = list(pymisca.header.it__window(DEFAULT_ARGS,2,2))
        for k, lst in DEFAULT_ARGS:
            if k not in _LST:
                _LST += [k] + lst        
                
        kw['CMD'] = CMD = ' '.join(map(repr,_LST)) ## for caching only
        kw['RUN_PARAMS'] = kw.copy() #### [legacy-keyword] 
        with _getPathStack([ OUTDIR ], force=1) as stack:
            OFNAME = 'ALIGNMENT-sorted.bam'
            
            self.shell.setJsonFile(OUTDIR / "%s.CMD.json"%self.__class__.__name__)
            if not FORCE and pymisca.shell.file__notEmpty(OFNAME):
                self.shell.loadCmd__fromJson()
#                 return 
                res = "SKIP"
                pass
            else:
                if DRY:
                    res = "RUN"
                else:
                    with self._lock(OFNAME) as lock:
                        with _getPathStack(['STAR_OUT'],force=1) as stack1:
                            if (stack1.d / DIR_TEMP).isdir():
                                shutil.rmtree(DIR_TEMP)
                            _LST = pymisca.header.list__call(_LST)

                            CMD = ' '.join(pymisca.header.stringList__flatten(_LST)) 
#                             res = pymisca.shell.shellexec(CMD)
                            res = self.shell.shellexec(CMD)
    
                        renamer= [
                            ('STAR_OUT/Aligned.sortedByCoord.out.bam','ALIGNMENT-sorted.bam'),
#                             ('STAR_OUT/Aligned.sortedByCoord.out.bam','ALIGNMENT.bam',),
                            ('STAR_OUT/Log.final.out','LOG'),
                            ('STAR_OUT/Unmapped.out.mate1','UNALIGNED_R1.fastq',),
                            ('STAR_OUT/Unmapped.out.mate2','UNALIGNED_R2.fastq',),
                        ]
                        if hasattr(renamer,'items'):
                            renamer = renamer.items()
                        for k,v in renamer:
                            self.shell.file__link( k, v, force=1)   
                            
                            
            
#                         with open("CMD.json","w") as f:
#                             json.dump( self.shell.logListLocal, f, indent=4)
#                         pyext.printlines([pyext.ppJson(self._logListLocal)],"LOG.json")

                        ########
                        res = pymisca.atto_jobs.job__samtools__qc(
                            {"INPUT_FILE":  "ALIGNMENT-sorted.bam",
                             "OUTDIR": OUTDIR,
                             "PARENT": self,
                             "FORCE":FORCE,
                             "DRY":DRY,
                             "SORTED":1,
                             "NCORE":NCORE,
                            }
                        )
                        kw['CHILDREN'] = [res] ###del
                        self.shell.dumpCmd__asJson()

