from pymisca.atto_job import AttoJob

import pymisca.ext as pyext
import time
import functools
import distutils.spawn


class rjob_quadron(AttoJob):
    PARAMS_TRACED = [
        ('INPUT_FILE',('AttoPath','')),
        ('OUTPUT_FILE',('AttoPath','')),
#         ('FASTA_GLOB',("unicode",'*.fasta')),
        ('SeqPartitionBy',("int",1E6)),
        ('NCORE',('int',1)),
        ('FORCE',('int',0)),
    ]
    def _run(self):
        kw = RUN_PARAMS = self._data
        assert kw['INPUT_FILE'] and kw['OUTPUT_FILE']
        kw['INPUT_FILE'] = INPUT_FILE = kw['INPUT_FILE'].realpath() 
        kw['OUTPUT_FILE'] = OUTPUT_FILE = kw['OUTPUT_FILE'].realpath()
        NCORE = kw['NCORE']
        FORCE = kw['FORCE'] 
        
        SeqPartitionBy = kw['SeqPartitionBy']
        DIR_QUADRON = distutils.spawn.find_executable("Quadron.lib")
        assert DIR_QUADRON is not None
        
        CMD = '''    
    R --slave  <<EOF &>LOG
#    if (!exists("Quadron")){load(system('which Quadron.lib ', intern = T))}
if (!exists("Quadron")){load("{{DIR_QUADRON}}")};

system.time({
Quadron(
        FastaFile = "{{INPUT_FILE}}",
        OutFile  = "{{OUTPUT_FILE}}",
        nCPU     = {{NCORE}},
        SeqPartitionBy = {{SeqPartitionBy}}
)    
})
EOF
    '''
        if not FORCE and pyext.file__notEmpty(OUTPUT_FILE):
            pass
        else:
            CMD = pyext.jf2(CMD)
            res = pyext.shellexec(CMD)
            
        
if __name__ == '__main__':
    rjob_quadron({"INPUT_FILE":"/work/mapped-data/G00000003/OUTPUT/GENOME.fasta",
                  'OUTPUT_FILE':"test.quadron.txt"})