import re
# import pymisca.shell as pysh
import pandas as pd
fastqLike = ".(fastq|fq)(.gz$|$)"
baseSpace0424 = ptn = r'(?P<ALIAS>.*_S(?P<SAMPLE_ID_INT>\d{1,3}))(?P<HASH_LIKE>)?_L(?P<chunk>\d+)_R(?P<read>[012])_(?P<trail>\d{1,4})\.(?P<ext>.+)'
firthLike0424 = ptn2 = r'(?P<ALIAS>.*_(?P<SAMPLE_ID_INT>\d{1,2}))-(?P<HASH_LIKE>\d{8})'

def getCounts__bowtie2log(buf):
    '''
    Example:
        32008 reads; of these:
        32008 (100.00%) were unpaired; of these:
        32008 (100.00%) aligned 0 times
        0 (0.00%) aligned exactly 1 time
        0 (0.00%) aligned >1 times
        0.00% overall alignment rate
    '''
#     'grep reads | grep align'
    ptn = re.compile(r'([\d\.\>]+)')
    res = pd.DataFrame(
        map(ptn.findall,buf.splitlines()),
        columns=['count','per','how']).set_index('how')
    return res.to_dict(orient='index')

def getCounts__bowtie1log(buf):
    '''
    Example:
    # reads processed: 32008
    # reads with at least one reported alignment: 3420 (10.68%)
    # reads that failed to align: 28588 (89.32%)
    Reported 3420 alignments
    '''
    buf = buf.replace('at least one','align >=1')
    buf = buf.replace('failed to align','align 0')
    ptn = re.compile(r'([\d\.\>\=]+)')
    res = map(ptn.findall,buf.splitlines())
    res = [ x for x in res if len(x)==3]
    res = pd.DataFrame(
         res,
        columns=['how','count','per',]).set_index('how')
    return res.to_dict(orient='index')