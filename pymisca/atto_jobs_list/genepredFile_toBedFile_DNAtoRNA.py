import itertools
import pymisca.ext as pyext
from pymisca.atto_job import AttoJob
from pymisca.shell import real__dir,file__notEmpty


from htseq_ext.htseq_extra import GenomicIntervalDeque,ValuedIterator
from htseq_ext.htseq_extra import ivdqq__ivdqr__splice

class genepredFile_toBedFile_DNAtoRNA(AttoJob):

# class bedFile_genepredFile_liftOver(AttoJob):
    '''
    This does not consider opposite strand yet
    '''
    PARAMS_TRACED = [
        ('INPUT_FILE',('AttoPath','')),
#         ('INPUT_FILE_DICT',('dict:AttoPath',{'BED':'','GENEPRED':''})),
        ('OUTPUT_FILE',('AttoPath','')),
        ('USE_EXON',('int',1)),
                    ]

    def _run(self):
        kw = self._data
        assert kw['OUTPUT_FILE']
        kw['OUTPUT_FILE'] = OUTPUT_FILE = kw['OUTPUT_FILE'].realpath()
        kw['INPUT_FILE'] = INPUT_FILE = kw['INPUT_FILE'].realpath()
        USE_EXON = kw['USE_EXON'] 

#     OUTPUT_FILE = './OUT.bed'
#     BED_FILE = FNAME = "/work/mapped-data/G00000003/WORKDIR/quadron_bed/OUT.bed"
#     file_genepred = "/work/mapped-data/G00000003/WORKDIR/init/genepred/ANNOTATION.genepred"
    

        def tuples_toIvdqs(
                           name, 
                           chrom ,
                           strand,
                           txStart,
                           txEnd,
                           cdsStart,
                           cdsEnd,
                           exonCount,
                           exonStarts,
                           exonEnds,
                           *a):
#             print(cdsStart,cdsEnd,txStart,txEnd,)
            cds  = [chrom,cdsStart,cdsEnd,strand,'%s.CDS'%name]
            lutr = [chrom,txStart, cdsStart,strand,'%s.5UTR'%name] 
            rutr = [chrom,cdsEnd, txEnd ,strand,  '%s.3UTR'%name]
            cdna = [chrom,txStart, txEnd ,strand,  '%s'%name]
            if strand == "-":
                lutr,rutr = rutr, lutr
                lutr[-1],rutr[-1] = rutr[-1],lutr[-1]
            elif strand== "+":
                pass
            else:
                assert 0,(strand,)
            return  [GenomicIntervalDeque.fromTuples([x]) for x in [lutr,cds,rutr,cdna]]
        
        it = pyext.readData( INPUT_FILE,'it')
        it = (x for x in it if x.strip())
        it = (x.strip().split('\t') for x in it)
        def _it(it=it):
            for x in it:
                y = tuples_toIvdqs(*x)
                
                if USE_EXON:
                    transcript = GenomicIntervalDeque.fromGenePredTuples(*x)
                else:
                    transcript = y[-1]
                yield (transcript, y[:3])
        it = _it()
#         it = (
#             (
#                 GenomicIntervalDeque.fromGenePredTuples(*x) ,
#                 tuples_toIvdqs(*x),
#               ) for x in it)        
#         it = (GenomicIntervalDeque.fromGenePredLine(x) for x in it)
        it,_it = itertools.tee(it)
#         it = ValuedIterator(it)
        it_genpred = it


#         ivqit = it_bed
        ivrit = it_genpred
        i = 0
        out = []
        N = -1


        def iterPairs(ivrit=ivrit):
            for ele in ivrit:
                yield ele
# #             ivqit = list(ivqit)
#             ivrit = list(ivrit)
#             for ivrs in ivrit:
#                 yield ivrs,ivrs
                    
        real__dir(fname=OUTPUT_FILE)            
        with open(OUTPUT_FILE, "w") as f:
            def callback(i,res,f=f):
                if len(res) > 1:
                    for i,x in enumerate(list(res)[:-1]):
                        y = res[i+1]
#                         assert x.end == y.start,(x,y,res)
                        assert x.end == y.start,(x,y,)
                    res = res.iv_merged()
                else:
                    assert len(res)==1,(res,)
                
                f.write(res[0].toBedLine())
                out.append(res)
            
            for ivrs,(lutr,cds,rutr) in iterPairs():
                ivrs = ivrs.flipped_to_plus()
                for x in [lutr,cds,rutr]:
#                     print x[0].chrom
#                     if not len(x[0]):
#                         continue
                    x = ivdqq__ivdqr__splice( x.flipped_to_plus(), ivrs)
#                     if not x:
#                         continue
                    callback(i, x.flipped_to_genome())

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
        