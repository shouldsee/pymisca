import itertools
import pymisca.ext as pyext
from pymisca.atto_job import AttoJob
from pymisca.shell import real__dir,file__notEmpty


from htseq_ext.htseq_extra import GenomicIntervalDeque,ValuedIterator
from htseq_ext.htseq_extra import ivdqq__ivdqr__splice

import shutil

class bedFile_genepredFile_liftOver(AttoJob):
    '''
    This does not consider opposite strand yet
    '''
    PARAMS_TRACED = [
        ('INPUT_FILE_DICT',('dict:AttoPath',{'BED':'','GENEPRED':''})),
        ('OUTPUT_FILE',('AttoPath','')),
                    ]

    def _run(self):
        kw = self._data
        assert kw['OUTPUT_FILE']
        kw['OUTPUT_FILE'] = OUTPUT_FILE = kw['OUTPUT_FILE'].realpath()
        _d = INPUT_FILE_DICT = kw['INPUT_FILE_DICT']
        for k,v in _d.items():
            assert v,(k,)
            _d[k] = v.realpath()

#     OUTPUT_FILE = './OUT.bed'
#     BED_FILE = FNAME = "/work/mapped-data/G00000003/WORKDIR/quadron_bed/OUT.bed"
#     file_genepred = "/work/mapped-data/G00000003/WORKDIR/init/genepred/ANNOTATION.genepred"
    

        it = pyext.readData(INPUT_FILE_DICT['BED'],'it')
        def _it(it=it):
            for line in it:
                x = line.strip()
                if not x:
                    continue
#                 assert 0
                x = x.strip('\t')
                x = x.split('\t')
                x = GenomicIntervalDeque.bedTuple2SamTuple(x) 
                x = GenomicIntervalDeque.fromTuples([x])
                yield x
#                 print x[0].__dict__
#                 break
        it = _it()
        it,_it = itertools.tee(it)
        it_bed = it = ValuedIterator(it)
        tups1 = tups = list(enumerate(_it))[:10]
        tups

        # ivrs_bed = ivrs = GenomicIntervalDeque.fromTuples(tups)
        # ivrs

        it = pyext.readData( INPUT_FILE_DICT['GENEPRED'],'it')
        it = (GenomicIntervalDeque.fromGenePredLine(x) for x in it if x.strip())
        it,_it = itertools.tee(it)
        it = ValuedIterator(it)
        it_genpred = it


        ivqit = it_bed
        ivrit = it_genpred
        i = 0
        out = []
        N = -1


        def iterPairs(ivqit,ivrit):
            ivqit = list(ivqit)
            ivrit = list(ivrit)
#             print pyext.ppJson([ivqit[0],ivqit[0].__dict__,ivqit[0][0].__dict__],default=repr)
            
            for ivqs in ivqit:
                for ivrs in ivrit:
                    yield ivqs,ivrs
                    
        real__dir(fname=OUTPUT_FILE)            
        with open(OUTPUT_FILE+'.partial', "w") as f:
            def callback(i,res,f=f):
#                 assert len(res)==1
                if len(res) > 1:
                    print(res)
                    return
                else:
                    f.write(res[0].toBedLine())
                    out.append(res)
            #         f.write("out.append(GenomicIntervalDeque.fromTuples(%s))\n"%repr(res.toTuples()))

            for ivqs,ivrs in iterPairs(ivqit,ivrit):
                res = ivdqq__ivdqr__splice( ivqs.flipped_to_plus(),ivrs.flipped_to_plus())            
        
        ###  rev strand
        #             if not res:
        #                 res = ivdqq__ivdqr__splice( ivqs, ivrs.flipped_to_plus())
                if not res:
                    continue
                res = res.flipped_to_genome()            
                callback(i,res)
        shutil.move(OUTPUT_FILE+'.partial',OUTPUT_FILE)
         

if __name__ == '__main__':
    
    with pyext.getPathStack([
        pyext.os.path.dirname(pyext.os.path.realpath(__file__)),
        "bedFile_genepredFile_liftOver.test_data/"]):
        
        OFNAME = "OUT.bed"
        bedFile_genepredFile_liftOver({
            "INPUT_FILE_DICT":{
                "BED":  'INPUT.bed',
                "GENEPRED":"INPUT.genepred",
    #             "IPYNB":"G00000002-mm10.ipynb",
            },
            "FORCE":1,
            "OUTPUT_FILE":OFNAME,
        })
        import filecmp
        assert filecmp.cmp( OFNAME, OFNAME+'.expect')
        print "DONE"    
        