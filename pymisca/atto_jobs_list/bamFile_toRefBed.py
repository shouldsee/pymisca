import pymisca.ext as pyext

import pysam
import shutil

from pymisca.atto_job import AttoJob
import pymisca.shell
import pymisca.pyhawk

class bamFile_toRefBed(AttoJob):
    PARAMS_TRACED = [
        ('INPUT_FILE',('AttoPath','')),
        ('OFNAME',('AttoPath','')),
        ('FILTER_FUNCS',('list:object',None)),
        ('FORCE',('int',0)),
    ]
    def _run(self):
        kw = self._data
        assert kw['INPUT_FILE']
        kw['INPUT_FILE'] = INPUT_FILE = kw['INPUT_FILE'].realpath()
        assert kw['OFNAME']
        kw['OFNAME'] = OFNAME = kw['OFNAME'].realpath()
        FORCE = kw['FORCE']
        FILTER_FUNCS = kw['FILTER_FUNCS']
        
        if not FORCE and pymisca.shell.file__notEmpty(OFNAME):
            pass
        else:
            pymisca.shell.real__dir(fname=OFNAME)
            with open(OFNAME+'.partial',"w") as fout:
                with pysam.AlignmentFile(INPUT_FILE,"r") as bam:
                    it =zip(bam.references,['0']*len(bam.references),bam.lengths)
                    def stdin(it=it):
                        for ele in it:
                            line = '\t'.join(map(str,ele))
                            yield '%s\n'%line
                        
                    it = stdin()
                    d = {
                        "lambdaFuncs":FILTER_FUNCS,
                         "FS":"\t",
                        "OFS":"\t",
                    }
                    pymisca.pyhawk.main(
                        d, 
                        stdinFunc = lambda :next(it,None),
                        stdoutFunc = fout.write,
                        stderrFunc = lambda x:None,
                   )
                    
                shutil.move(OFNAME+'.partial',OFNAME)
    
    


if __name__ =='__main__':
    bamFile_toRefBed({"INPUT_FILE":"ALIGNMENT-sorted.bam",
                      "FORCE":1,
                      "OFNAME":"test.bed",
                     "FILTER_FUNCS":[lambda line:line[0]!="chr10"]})
    import pymisca.ext as pyext
    d = dict(func=lambda: bamFile_toRefBed({
        "INPUT_FILE":"examples/test.sam",
         "FORCE":1,
          "OFNAME":"out/test.bed",
         "FILTER_FUNCS":[lambda line:line[0]=="chr10"]})
        )
    d['func']()
    assert list(open("out/test.bed","r")) == [u'chr10\t0\t130694993\n']    