import pymisca.atto_jobs
import pymisca.ext as pyext
import lazydict
import pysam
import itertools
pd = pyext.pd
bam = template= lazydict.LazyDictionary()
bam['INPUT_FILE'] = None

bam['N_HEAD'] = 100000
# bam['N_HEAD'] = -1
bam['file/open'] = lambda self,key, INPUT_FILE: pysam.AlignmentFile( INPUT_FILE,'rb') 
bam['file/header']  = lambda self,key,file_open: file_open.header
bam['file/close'] = lambda self,key, file_open: file_open.close()

# @pyext.setItem
bam['flag/to_hex'] =  {'is_unmapped':'0x4',
                                         'is_secondary':'0x100',}
bam['filter/inc'] = []
bam['filter/exc'] = ['is_unmapped','is_secondary']
bam['filter/inc/hex'] =  lambda self,key,flag_to_hex,filter_inc: map(flag_to_hex.__getitem__, filter_inc)
bam['filter/exc/hex'] =  lambda self,key,flag_to_hex,filter_exc: map(flag_to_hex.__getitem__, filter_exc)
# bam['list/seq'] = lambda  self
@pyext.setItem(bam, 'iter/seq')
def _func(bam,key, file_open, 
          N_HEAD,filter_inc, filter_exc ):
    it = file_open
    def _func(it):
        for seq in it:
            keep = 1
            for _filter_inc in filter_inc:
                if not getattr(seq,_filter_inc):
                    keep= 0 
                    break
            for _filter_exc in filter_exc:
                if getattr(seq, _filter_exc):
                    keep = 0
                    break
            if keep:
                yield seq
    it = _func(it)
    if N_HEAD >=0:
        it = itertools.islice(it,0,N_HEAD)
    return list(it)

@pyext.setItem(bam,'list/seq/length')
def _func(bam,key, iter_seq):
    res = [len(x.__getattribute__('seq')) for x in iter_seq]
    return res

@pyext.setItem(bam,'seq/reference_name/list')
def _func(bam,key, iter_seq):
    res = [x.__getattribute__('reference_name') for x in iter_seq]
    return res

@pyext.setItem(bam,'seq/reference_name/count/series/pysam')
def _func(bam,key, 
          file_header, seq_reference_name_list ):
    # refNames = bam['file/header']
    refNames = file_header.references
    ct = pyext.OrderedCounter(refNames)
    ct.subtract(refNames)
    ct.update(seq_reference_name_list)
    ct = pd.Series(ct)
    return ct

@pyext.setItem(bam, 'seq/reference_name/count/series')
@pyext.setItem(bam, 'seq/reference_name/count/series/samtools')
def _func(bam,key,INPUT_FILE,filter_exc_hex,filter_inc_hex,N_HEAD):
    CMD = [
        'cat',INPUT_FILE,
        '|samtools','view','-h',
        ['-F %s'% x for x in filter_exc_hex],
        ['-f %s'% x for x in filter_inc_hex],
        ['|head','-n','%d'%N_HEAD] if N_HEAD >0 else [],
        ['|samtools','idxstats','-'],
        ['|cut','-f1,3'],

    ]


    CMD = pyext.stringList__flatten(CMD)
    CMD = ' '.join(CMD)

    res= pyext.shellexec(CMD)
    res = pyext.read__buffer(res,ext='tsv',header=None).iloc[:-1,0]
    return res