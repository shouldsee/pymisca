import pymisca.ext as pyext
# import requests
# import pymisca.atto_string as pyatt
# e = exons[0]
from HTSeq import GenomicInterval
from pyRiboSeq.htseq_extra import GenomicIntervalDeque
from pyRiboSeq.htseq_comp import ivdq__fastaDict__getSequence, iv__intersect, iv__null

# import pymisca.bio
from pymisca.bio import fasta__toDataFrame

# DB_SCRIPT = {
#     'MODULE_NAME':__name__,
#     'PARAMS_TRACED':[
#         ('FILEGLOB_GENEPRED',('AttoPath','GENEPRED')),
#         ('FILEGLOB_FASTA',('AttoPath','GENOME.fasta')),
#         ('OPTION_FASTA_EMPTY',('AttoPath','single_n')),
#         ('OPTION_FASTA_HEADER',('AttoPath','simple')),
#         ('OPTION_OUTPUT_FORMAT',('AttoPath','fasta')),
#         ('DEBUG',('int',0)),
#     ],

# }
# #  files
# #### [TBC] implement file size checking

from pymisca.atto_jobs import AttoJob

class emptyFuncs(object):
    '''
    Options to deal with empty fasta sequences
    '''
    @classmethod
    def single_n(cls,lines):
        lines[1] = 'N\n'
        return lines
    
class headerFuncs(object):
    '''
    Options to add headers
    '''
    @classmethod
    def type_name(cls,**kw):
        return '>{type}::{name}\n'.format(**kw)
    
    @classmethod
    def simple(cls,**kw):
        return '>{name}\n'.format(**kw)
    
    
class outputFuncs(object):
    @classmethod
    def fasta(cls,**kw):
        it = kw['it']
        FASTA_DICT= kw['FASTA_DICT']
        headerFunc = kw['headerFunc']
        def _toFileIter(it):
            for d in it:
                for TYPE, ivdq in d['regions'].items():
                    ### TYPE in ['CDS','5UTR','3UTR',]
        #             TYPE = 'CDS'
                    header = headerFunc(name=d['name'],type=TYPE)
                    assert list(ivdq)

        #             seq = ivdq__fastaDict__getSequence(d['regions'][TYPE],FASTA_DICT) + '\n'
                    seq = ivdq__fastaDict__getSequence( ivdq,FASTA_DICT) + '\n'
                    lines = [header, seq]
                    if not seq:
                        lines = emptyFunc(lines)

#                     if lines:
                    yield (TYPE+'.fasta', lines)

        _it = _toFileIter(it)
        res = pyext.itGrouped__toFiles(_it, newline='')
        return res
    @classmethod
    def npy(cls,**kw):
        it = kw['it']
        np.save('OUT.npy',it)
        return 
    
    @classmethod
    def bed(cls,**kw):
        it = kw['it']
        FASTA_DICT= kw['FASTA_DICT']
        def _toFileIter(it):
            for d in it:
                for TYPE, ivdq in d['regions'].items():
                    lines = []
                    for iv in ivdq :
                        ### ['chrom', 'start', 'end', 'acc', 'score', 'strand', 'FC', 'neglogPval', 'neglogQval', 'summit', 'img']
                        if iv.length:
                            line = '\t'.join(map(str,[iv.chrom, iv.start, iv.end, TYPE, '1', iv.strand, ])) + '\n'
                            lines+=[line]
                    yield ('OUT.bed',lines)
                    
                        
#                     ### TYPE in ['CDS','5UTR','3UTR',]
#         #             TYPE = 'CDS'
#                     header = headerFunc(name=d['name'],type=TYPE)
#                     assert list(ivdq)

#         #             seq = ivdq__fastaDict__getSequence(d['regions'][TYPE],FASTA_DICT) + '\n'
#                     seq = ivdq__fastaDict__getSequence( ivdq,FASTA_DICT) + '\n'
#                     lines = [header, seq]
#                     if not seq:
#                         lines = emptyFunc(lines)

# #                     if lines:
#                     yield (TYPE+'.fasta', lines)

        _it = _toFileIter(it)
        res = pyext.itGrouped__toFiles(_it, newline='')
        return res
    
_getPathStack  = pyext.getAttoDirectory    
class genepred_to_other(AttoJob):
    PARAMS_TRACED = [
        ('INPUT_FILE_DICT',('dict:AttoPath',
                            {'GENEPRED':'','FASTA':''
                            })),
#         ('INPUTDIR',('AttoPath',''))
        ('OUTDIR',('AttoPath','')),
        ('FILEGLOB_GENEPRED',('AttoPath','GENEPRED')),
        ('FILEGLOB_FASTA',('AttoPath','GENOME.fasta')),
        ('OPTION_FASTA_EMPTY',('AttoPath','single_n')),
        ('OPTION_FASTA_HEADER',('AttoPath','simple')),
        ('OPTION_OUTPUT_FORMAT',('AttoPath','fasta')),
        ('DEBUG',('int',0)),
        ('FORCE',('int',0)),
    ]
    def _run(self):
        kw = DB_WORKER = RUN_PARAMS = self._data
#         kw['INPUTDIR'] = INPUTDIR = kw['INPUTDIR'].realpath()
#         assert INPUTDIR
        kw['OUTDIR'] = OUTDIR = kw['OUTDIR'].realpath()
        assert OUTDIR
        kw['INPUT_FILE_DICT'] = INPUT_FILE_DICT = {k:v.realpath() for k,v in kw['INPUT_FILE_DICT'].items()}
        FORCE = kw['FORCE']

#         FILEGLOB_GENEPRED = RUN_PARAMS['FILEGLOB_GENEPRED']
#         FILEGLOB_FASTA = RUN_PARAMS['FILEGLOB_FASTA']
        OPTION_FASTA_EMPTY = RUN_PARAMS['OPTION_FASTA_EMPTY']
        OPTION_FASTA_HEADER = RUN_PARAMS['OPTION_FASTA_HEADER']
        OPTION_OUTPUT_FORMAT = RUN_PARAMS['OPTION_OUTPUT_FORMAT']
        DEBUG = RUN_PARAMS['DEBUG']
        emptyFunc  = getattr(emptyFuncs, OPTION_FASTA_EMPTY)
        headerFunc = getattr(headerFuncs, OPTION_FASTA_HEADER)
        outputFunc = getattr(outputFuncs, OPTION_OUTPUT_FORMAT)
    #     emptyOption = 'singleN'    
    #     headerOption = 'simple'
        with _getPathStack([DB_WORKER['OUTDIR']],force=1):
            pass






        with _getPathStack([DB_WORKER['OUTDIR']],force=1):
            OFNAME = "LOG.%s"%(OPTION_OUTPUT_FORMAT,)
            if not FORCE and pyext.file__notEmpty(OFNAME):
                pass
            else:

                df_GENEPRED = pyext.readData( INPUT_FILE_DICT['GENEPRED'],
                                             'tsv',columns=pyext.COLUMNS_GENEPREDEXT,header=None,guess_index=0)
                df = FASTA_DICT = fasta__toDataFrame(INPUT_FILE_DICT['FASTA'])
                d = pyext.df__asMapper(df,'ACC','SEQ')
                FASTA_DICT = d

                it = pyext.df__iterdict(df_GENEPRED)
                out = []
                for d in it:
                    res = worker__splitExons(d)
                    out.append(res)
                it = transcripts_splitted = out

                if DEBUG:
                    pyext.np.save('transcripts_splitted.npy',out)
                    it = transcripts_splitted = pyext.readData('transcripts_splitted.npy',allow_pickle=1).tolist()

                it = outputFunc(**locals())

                pyext.printlines(["DONE"],OFNAME)

def worker__splitExons(d):
        chrom = d['chrom']
        strand = d['strand']
        com = common = dict(chrom=chrom,strand=strand)
        
        d['regions'] = regions = {'5UTR':GenomicIntervalDeque([],**com),
                   '3UTR':GenomicIntervalDeque([],**com),
                   'CDS':GenomicIntervalDeque([],**com),                  
                   'CDNA':GenomicIntervalDeque([],**com),                  
                  }
        ivdqs = {}
        
#         iv_trans = HTSeq.GenomicInterval(start=d['txStart'],end=d['txEnd'],**com)
#         ivdq_trans = GenomicIntervalDeque([iv_trans])
        
        if d['cdsStart'] == d['cdsEnd'] == 0:
            d['cdsStart'] = d['cdsEnd'] = d['txEnd']
#             #### Process a gene without CDS
#             ivdqs['CDS'] =  GenomicIntervalDeque([iv__null()])
#             ivdqs['5UTR'] = GenomicIntervalDeque([iv__null()])
#             ivdqs['3UTR'] = GenomicIntervalDeque([iv__null()])
#             ivdqs['CDNA'] =  GenomicIntervalDeque([ GenomicInterval(start=d['txStart'],end=d['txEnd'],**com)])
#         else:
                
        ivdqs['CDS'] = ivdq_cds = GenomicIntervalDeque([
            GenomicInterval(start=d['cdsStart'],end=d['cdsEnd'],**com)
        ])
        ivdqs['5UTR'] = GenomicIntervalDeque([
            GenomicInterval(start=d['txStart'],end=d['cdsStart'],**com)
        ])
        ivdqs['3UTR'] = GenomicIntervalDeque([
            GenomicInterval(start=d['cdsEnd'],end=d['txEnd'],**com)
        ])

        if strand=='-':
            ivdqs['3UTR'],ivdqs['5UTR'] = ivdqs['5UTR'],ivdqs['3UTR']

        ### construct exons
        exons = GenomicIntervalDeque([],strand=strand,chrom=chrom)
        for es, en in zip(d['exonStarts'].split(','),d['exonEnds'].split(',')):
            if not es:
                continue
            es = int(es)
            en = int(en)
            exon = GenomicInterval(start=es,end=en,**com)
#             exons += [exon]
            exons.append(exon)
        assert len(exons)


        #### split exons
        for i,exon in enumerate(exons):
            exon = exons[i]
            L = exon.length
            _L = 0
            for k in ['5UTR','CDS','3UTR',
#                       'CDNA'
                     ]:
                ivdq = ivdqs[k]
                res = iv__intersect( exon, ivdq.iv)
                if res.length:
                    regions[k].append(res)
                    _L += res.length
                if _L == L:
                    break
            assert _L==L

        for TYPE in ['5UTR','CDS','3UTR']:
            regions['CDNA'].extend(regions[TYPE])

            if not regions[TYPE].length:
                regions[TYPE] = ivdqs[TYPE]


        return d

#             GenomicIntervalDeque.
#             GenomicIntervalDeque.append(exon)
#             (exons)
#         regions['5UTR'] = dict(start=d['txStart'],end=['cdsStart'])
#         regions['3UTR'] = dict(start=d['cdsEnd'],end=['txEnd'])
#         regions['5UTR'] = 