# -*- coding: utf-8 -*-
import Bio

from BCBio.GFF import GFFExaminer
import BCBio.GFF as bgff

import itertools
# import pymisca.util as pyutil


# import pymisca.vis_util as pyvis
# plt = pyvis.plt
class plt:
    pass
class pyvis:
    pass

import pymisca.ext as pyext
pd = pyext.pd


def bam__getHeader(FNAME,mode="rb",**kw):
    import pysam
    with pysam.AlignmentFile(FNAME,mode=mode,**kw) as f:
        return f.header    
###### this implmentation is much slower somehow
# class fastaIter(object):
#     def __init__(self, *a,**kw):
# #         super(fastaIter, self).__init__( *a,**kw)
#         self.a = a
#         self.kw = kw
#         self.handle = None
#     def __enter__(self,*a,**kw):
#         it = self.handle = open(*self.a,**self.kw)
        
#         it = itertools.groupby(it,lambda x:x.startswith('>'))
#         header = None
#         for isHeader, _it in it:            
#             if isHeader:
#                 _it = list(_it)
#                 assert len(_it) == 1
#                 header = _it[0]
#             else:
#                 assert header is not None
# #                 _itt = itertools.tee(_it)[1]
#                 _it = list(_it)
#                 yield (header, _it,   itertools.chain( [header] , list(_it) ) )
        
#     def __enter__(self,*a,**kw):
#         it = self.handle = open(*self.a,**self.kw)
        
        
#         def nextHeader(it):
#             for line in it:
#                 if line.startswith('>'):
#                     yield [line]
#                     break
#             yield None

#         def nextBuffer(it):
#             return itertools.takewhile(lambda x:not x.startswith('>'),
#                                       it)
        
#         def _main(it):
#             lastHeader = 0
#             _it = []
# #             while True:
#             isStart = 1
#             for i,line in enumerate(it):
#                 if line.startswith('>'):
#                     assert not lastHeader
# #                     print i
# #                     if i!=0:
#                     if not isStart:
#                         yield _it  #### empty
#                         _it = []
#                     yield [line]
#                     lastHeader = 1
#                     isStart = 0
#                 else:
#                     _it = itertools.chain(_it,[line])
#                     lastHeader = 0
#             yield _it
# #                 res = nextHeader(it)
# #                 if res is None:
# #                     break
# #                 else:
# #                     yield res
# #                 res = nextBuffer(it)
# #                 yield res
                
#         it = _main(it)
# #         pyext.nTuple(it)      

#         it = pyext.it__window(it,n=2,step=2)
    
#         for x in it:
#             yield x
   
    
#     def __exit__(self,*a,**kw):
#         self.handle.close()
#         pass
    
#     def __iter__(self):
#         return self
#     def __next__(self):
#         pass
    
    
class fastaIter(object):
    def __init__(self, *a,**kw):
#         super(fastaIter, self).__init__( *a,**kw)
        self.a = a
        self.kw = kw
        self.handle = None
        assert pyext.file__notEmpty(a[0]),('Cant parse empty file as FASTA:',a[0],)
#         assert os.path.isfile(a[0]) or os.pat
    def __enter__(self,*a,**kw):
        it = self.handle = open(*self.a,**self.kw)
        
        it = itertools.groupby(it,lambda x:x.startswith('>'))
        header = None
        for isHeader, _it in it:            
            if isHeader:
                _it = list(_it)
                assert len(_it) == 1
                header = _it[0]
            else:
                assert header is not None
#                 _itt = itertools.tee(_it)[1]
                _it = list(_it)
                yield (header, _it,   itertools.chain( [header] , list(_it) ) )
    
    def __exit__(self,*a,**kw):
        self.handle.close()
        pass
    
    def __iter__(self):
        return self
    def __next__(self):
        pass
    
def fasta__toDataFrame(FNAME):
    with fastaIter(FNAME) as f:
        df = [x[:2] for x in f]
        df = pyext.pd.DataFrame(df,columns = ['HEADER','SEQ_LINES'])
        df['ACC'] = df['HEADER'].str.lstrip('>').str.strip()
        df['SEQ'] = df['SEQ_LINES'].map(lambda lines:''.join([x.strip() for x in lines]))

    return df    

def parse__fastq(FNAME,buffering=-1,**kw):
    '''
    Return a iterator of length-4-lists, corresponding to the four lines of each fastq entry
    '''
    with open(FNAME,'r',buffering=buffering,**kw) as it:
        lst = []
        for i,x in enumerate(it):
            lst.append(x.rstrip())
            if not (i+1)% 4:
                yield lst
                lst = []


def resolve_from_sra(runID,dtype='sra',
                    host = 'ftp://ftp-trace.ncbi.nih.gov/sra/sra-instant/reads/ByRun/sra'):
    assert len(runID) > 6,runID
    head3 = runID[:3]
    head6 = runID[:6]
    assert head3 in ['DRR','ERR','SRR'],runID
    uri = None
    if dtype=='sra':
        if head3=='ERR':
            uri = '{host}/{head3}/{head6}/{runID}/{runID}.sra'.format(**locals())
    if uri is None:
        raise Exception('Not implemented')
        
    return uri


def gtf__guess_gid_name(fname,head=10):
    it = pyext.readData(pyext.file__header(fname,head),ext='it',)
    d = pyext.collections.Counter()
    # for line in it
    map(d.update,(x.split() for x in it));
    df = pd.DataFrame(d.items(),columns=['key','count'])
    df['up']= df.key.str.upper()
    dfc = df.query('count >= @head - 1')
    res =dfc.query('up.str.startswith("GENE")')
    assert len(res)==1,'Failed to identify a string starts with "GENE"'

    name = res.iloc[0,0]
    return name

def read_gtf(in_file, cache=1):
    if isinstance(in_file,str):
        in_handle = open(in_file)    
    else:
        in_handle = in_file
    it = bgff.parse(in_handle,)
    if cache:
        res = list( it )
        in_handle.close()
    else:
        res = it
    return res



def gene2transcript(g,force=0):
    '''Convert a SeqRecord from BCBio.GFF.parse() to a dictionary-like object
'''
    if isinstance(g,Bio.SeqRecord.SeqRecord):
        if not force:
            assert len(g.features)==1
        g = g.features[0]
        
    feats = g.sub_features
    d = {'parent':g}
    for i,f in enumerate(feats):
        if f.type in ['start_codon','stop_codon']:
            d[f.type]  = f
    return pyext.util_obj(**d)

def add_transcript(g,ax=None,ycent=0.25,adjust_xlim=1,
                   force=0):
    if type(g) in [Bio.SeqRecord.SeqRecord,
                   Bio.SeqFeature.SeqFeature]:
#     if isinstance(g, Bio.SeqFeature.SeqFeature):
        
        g = gene2transcript(g,force=force)
    
    if ax is None:
        ax = plt.gca()
#     if g.parent.strand==-1:
#         (g.parent.location.end,
#         g.parent.location.start) = (g.parent.location.start,
#                                     g.parent.location.end)
    tss = g.parent.location.start
    tend = g.parent.location.end
    if g.parent.strand==-1:
        tss,tend = tend,tss
    #### use parent if can't find codons due to truncation
    start_codon = g.__dict__.get('start_codon',None)
    stop_codon = g.__dict__.get('stop_codon',None)
#     trss =  start_codon.location.start
    trss = start_codon.location.start  if start_codon is not None else tss    
    trend = stop_codon.location.end  if stop_codon is not None else tend  
    tlim = tss,tend

#         ycent *= -1
    arrowArgs = {'ycent':ycent,'ax':ax}
        
#     print (tss,trss,trend,tend)
    ax.text(trss,ycent + 0.5,g.parent.id,
            horizontalalignment='center')
    patches=[pyvis.add_hbox(tss,trss,height=0.25,**arrowArgs),
        pyvis.add_harrow(trss,trend,height=0.5,**arrowArgs),
        pyvis.add_hbox(trend,tend,height=0.25,**arrowArgs)]
    if adjust_xlim:
        xlim = ax.get_xlim()
        ax.set_xlim(min(xlim[0],min(tss,tend)),
                    max(xlim[1],max(tss,tend)))   
        ax.set_ylim(-.5,.5)
        
    return tlim,patches,ax


def add_transcript(g,ax=None,ycent=0.25,
                   adjust_xlim=1,debug=0,
                   intronHeight = 0.25,
                    exonHeight = 1.0,
                   force=0):
#     if type(g) in [Bio.SeqRecord.SeqRecord,
#                    Bio.SeqFeature.SeqFeature]:
# #     if isinstance(g, Bio.SeqFeature.SeqFeature):
#         g = gene2transcript(g,force=force)
    
    if ax is None:
        ax = plt.gca()
        
    tss = g.location.start
    tend = g.location.end
    subfeats = g.sub_features
        
    arrowArgs = {'ycent':ycent,'ax':ax}
    

#     intronHeight = 0.25
#     exonHeight = 0.5
#     debug = 0
    def add_cds(start,end,strand=1):
#         if strand == -1:
#             end,start = start,end
        if abs(start - end)<=3:
            return None
        if debug:
            print ('adding exon',start,end)
        patch = pyvis.add_harrow(start,
                                   end,
                                 height = exonHeight,
#                                      head_length= (start_new- end)/2.,
                                 **arrowArgs)
        return patch
    def add_intron(start,end,strand=1):
#         if strand == -1:
#             end,start = start,end
        if abs(start - end)<=3:
            return None
        if debug:
            print ('adding intron',start,end)
        patch = pyvis.add_hbox(start,end,
                               height = intronHeight,head_length=0.,
                                   **arrowArgs)
        
        
        return patch
        
#     end = g.location.start
#     if g.strand == -1:
#         subfeats = subfeats[::-1]
#         tss,tend = tend,tss        
    def getStrandedLocation(f):
        start,end = f.location.start, f.location.end
        if f.strand== -1 :
            start,end = end,start
        return start,end
    tss,tend = getStrandedLocation(g)
    if g.strand == -1:
        subfeats = subfeats[::-1]        

    end = tss
    tmid = (tss+tend)//2
#     ax.text(end, ycent + intronHeight , g.id,
#             horizontalalignment='center')

    if adjust_xlim:
        xlim = ax.get_xlim()
        ax.set_xlim(min(xlim[0],min(tss,tend)),
                    max(xlim[1],max(tss,tend)))   
        ax.set_ylim(-.5,.5)
        ax.set_ylim(-2,2)

    xlim = ax.get_xlim()
#     textLoc = 
    xspan = xlim[-1] - xlim[0]
    textLoc = tss
    textLoc = max(xlim[0] + 0.1*xspan, textLoc)
    textLoc = min(xlim[1] - 0.1*xspan,textLoc)
    ax.text(textLoc, 
            ycent + intronHeight , 
            g.id,
            horizontalalignment='center')
#     print ('[strand]',g.strand)
    
    patches = []
    count_cds = 0
    
    for i,f in enumerate(subfeats):
        if f.type=='CDS':
#             start_new,end_new =　f.location.start, f.location.end
            start_new,end_new = getStrandedLocation(f)
            
            patch = add_intron(end,start_new,strand = 1)
            patches += [patch]            
            
            patch = add_cds(start_new,end_new,strand= 1)
            patches += [patch]
            
#             print end,start_new,end_new
            
            start,end = start_new,end_new
            count_cds += 1
            
        if i + 1 ==len(subfeats):
            start_new = tend
            ### this is actually utr
            patch = add_intron(end,start_new)
#             patch = pyvis.add_hbox(end,start_new,
#                                    height = intronHeight,head_length=0.,
#                                    **arrowArgs)
#             patch = pyvis.add_harrow(end,start_new,
#                                      height = intronHeight,
#                                      head_length= (start_new- end)/2.,
#                                      **arrowArgs)
#             pyvis.add_arrow([patch])
            patches += [patch]
            continue

    
    tmin = min(tss,tend)
    tmax = max(tss,tend)
        
#     if g.strand==-1:
#         tss,tend = tend,tss        
    return (tmin,tmax),patches,ax

