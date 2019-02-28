from __future__ import absolute_import

from pymisca.header import *
import os,sys,subprocess,io
import json
import functools, itertools, copy,re
import collections
import urllib2,urllib,io    
import pymisca.fop as pyfop
import pymisca.pandas_extra as pypd
pd=pypd.pd
import pymisca.shell as pysh

pyext = sys.modules[__name__]

import pymisca.header as pyhead
# pyhead.mpl__setBackend(bkd='Agg')
pyhead.set__numpy__thread(vars(sys.modules['__main__']).get('NCORE',None))

import pymisca.numpy_extra as pynp
np = pynp.np

from pymisca.util__fileDict import *

import json
import pymisca.oop as pyoop

import StringIO, codecs

def iter__toSoft(it):
    for line in it:
        if line:
            if line[0] in list('!/'):
                yield line
def iter__toFile(it,fname,lineSep='\n'):
    with io.open(fname,'w',encoding='utf8') as f:
        for i,line in enumerate(it):
            if i ==0:
                _type = type(line)
            if lineSep is not None:
                line = line.strip() + _type(lineSep)
            f.write(line)
            
def unicodeIO(handle=None, buf=[],encoding='utf8',*args,**kwargs):
    if handle is None:
        handle = StringIO.StringIO()
    codecinfo = codecs.lookup(encoding)
    wrapper = codecs.StreamReaderWriter(
        handle, 
        codecinfo.streamreader, 
        codecinfo.streamwriter)
    if buf:
        wrapper.write(buf)
        wrapper.seek(0)
    return wrapper

def kwarg2dict(res,d = None,silent=1,lineSep='\n'):
    if d is None:
        d = collections.OrderedDict()
    for line in res.strip().split(lineSep):
        sp = line.strip().partition("=")
        if sp[1]=='=':
            (key, _, value) = sp
            key = key.strip()
            value =value.strip()
            d[key] = value
            if not silent:
                print (sp)
                print(key,'=',value)
    return d

def fileDict__load(fname):
    fdir = pyext.os.path.dirname(fname) 
#     with open(fname,'r') as f:
#         dd = json.load_ordered(f,)
#     if os.stat(fname).st_size == 0:
#         dd = collections.OrderedDict()
#     else:
    if 1:
        dd = pd.read_json(fname, typ='records'
                         ).to_dict(into=collections.OrderedDict)
        dd = collections.OrderedDict([(k,
                                       pyext.os.path.join(fdir,v)) 
                                      for k,v in dd.items()])
    dd = pyoop.util_obj(**dd)
    return dd

json.load_ordered  = functools.partial(json.load,
                                    object_pairs_hook=collections.OrderedDict,)
json.loads_ordered = functools.partial(json.loads,
                                    object_pairs_hook=collections.OrderedDict,)


def arr__l2norm(x,axis=None,keepdims=0):
    res = np.sum(x**2,axis=axis,keepdims=keepdims)**0.5
    return res


def list__paste0(ss,sep=None,na_rep=None,castF=unicode):
    '''Analogy to R paste0
    '''
    if sep is None:
        sep=u''
    
    L = max([len(e) for e in ss])
    it = itertools.izip(*[itertools.cycle(e) for e in ss])
    res = [castF(sep).join(castF(s) for s in next(it) ) for i in range(L)]
    res = pd.Series(res)
    return res
pasteB = paste0 = list__paste0


# def fileDict__save(fname=None,d=None,keys=None):
#     assert d is not None,'set d=locals() if unsure'
#     d = collections.OrderedDict(d)
#     if keys is not None:
#         d  = pyext.dictFilter(d,keys)
    
#     if isinstance(fname,basestring):
#         if os.path.exists(fname):
#             if os.stat(fname).st_size != 0:
#                 with open(fname,'r') as f:
#                     res = pyext.json.load_ordered(f,)
#                     res.update(d)
#                     d = res
                
#     res = pyext.json.dumps(d)
#     res = res + type(res)('\n')    
#     if isinstance(fname,basestring):
#         with open(fname,'w') as f:
#             f.write(res)
#         return fname
#     elif fname is None:
#         f = sys.stdout
#         f.write(res)
#     else:
#         f = fname
#         f.write(res)
#         return f
    
# def dictFilter(oldd,keys):
#     d =collections.OrderedDict( 
#         ((k,v) for k,v in oldd.iteritems() if k in keys) 
#     )
#     return d

class envDict(collections.OrderedDict):
    def __init__(self, *args,**kwargs):
        super(envDict,self).__init__(*args,**kwargs)
    def __getattribute__(self,name):
#         res = super(envDict,self).__getattribute__(name)
        try:
            res = super(envDict,self).__getattribute__(name)
        except AttributeError as e:
            res = self.get(name,None)
            if res is None:
                raise e
        return res
#     def __setattr__
#     def __setattr__(self,name,value):
#         super(envDict, self).__setitem__(name, value)
#         res = self.__dict__.get(name,None)
#         res = super(envDict,self).__dict__.get(name,None)
#         res = getattr( super(envDict,self), name, None)
#         .__getattr__(name)
#         if res is not None:
#             pass
#         else:
#             res = self.get(name,None)
#         return res

def unique(lst):
    """Generate unique items from sequence in the order of first occurrence.
    adapted from: https://stackoverflow.com/questions/12897374/get-unique-values-from-a-list-in-python
    """
    seen = list()
    for val in lst:
        yes = any((val is _val for _val in seen))
        if yes:
            continue
        seen.append(val)
        yield val
        
def index2frame(index,val=True,name=None):
    '''
    initialise frame from index with value
    '''
    index = pd.Index(index,name=name)
    track = pd.DataFrame([val]*len(index),index=index)
#     track = index.to_frame()
#     track.loc[:,:]=val
    return track    
  

def list__realise(lst, d=None, cls=basestring,check = 1):
    assert d is not None
    res = []
    for x in lst:
        if isinstance(x,cls):
            y = d.get(x,x)
            if (check) and (y is x):
                print ('[WARN] cannot realise:"%s"'% y)
        else:
            y = x 
        res.append(y)
    return res
#     res = [ d.get(x,x) for x in lst ]
    
def arr__polyFit(x,y, deg=1, **kwargs):
    x,y = map(np.ravel,np.broadcast_arrays(x,y))
    isNA = np.isnan(x) | np.isnan(y) 
    fit = np.polyfit(x[~ isNA], y[~isNA], deg=deg, **kwargs)
    return fit

def getBname(fname):
    '''
    Extract "bar" from "/foo/bar.ext"
    '''
    bname = fname.split('/')[-1].rsplit('.',1)[0]
    return bname

basename = getBname ###legacy


# def base__touch(fname):

# print ('[__main__]',__main__)

# def execBaseFile(fname,):
#     fname = pyext.base__file(fname)
#     g= vars(sys.modules['__main__'])
# #     g = __main__.globals()
#     res = execfile(fname, g, g)
# #     exec(open(fname).read(), g)
#     return

def sanitise__column(col):
    col = col.replace(' ','_')
    return col
def df__sanitiseColumns(dfc):
    cols = dfc.columns
    res = cols.map(sanitise__column)
    dfc = dfc.rename(columns=dict(zip(cols,res)))
    return dfc

def it__toExcel(it,ofname,engine='xlsxwriter',sheetFunc=lambda k:'__'.join(k),
                silent=0,**kwargs):
    f = pd.ExcelWriter(ofname,engine=engine,**kwargs)
    for k,mm in it:
        ali = sheetFunc(k)
        if not silent:
            print ('[writing]%s'%ali)

        mm.to_excel(f,sheet_name=ali,engine='xlsxwriter')
    f.close()            
    return ofname

def index__sortFactor(ind,level=None,inplace=False,dtype = None):
    '''sort a particular level of the multiIndex and change the labels accordingly
'''
#     ind.levels[i]
    ilvl = ind.levels[level]
    if dtype is not None:
        ilvl = ilvl.astype(dtype)
    silvl, od = ilvl.sort_values(return_indexer=1)
    np.argsort(od)
    iod = np.argsort(od)
    print  ind.labels[level]
    print (map(iod.__getitem__, ind.labels[level]))
    
    ind = ind.set_labels( map(iod.__getitem__, ind.labels[level]),level=level,inplace=inplace)
    ind = ind.set_levels(silvl, level=level,inplace=inplace)
    return ind

def index__set_level_dtype(ind,level=None, dtype=None,inplace=False):
    if dtype is None:
        pass
    else:
        ind.set_levels( ind.levels[level].astype(dtype),level=level,inplace=inplace)
    return ind


def file__callback(fname,callback,mode='r'):
    if isinstance(fname,basestring):
        with open(fname,mode) as f:
            res = callback(f)
    else:
        f = fname
        res = callback(f)
    return res

def file__link(IN,OUT,force=0):
    if os.path.exists(OUT):
        if force:
            assert os.path.isfile(OUT)
            os.remove(OUT)
        else:
            assert 'OUTPUT already exists:%s'%OUT
    else:
        pass
    os.link(IN,OUT)
    return OUT

def file__rename(d,force=0, copy=1):
    for k,v in d.items():
        file__link(k,v,force=force)
        if not copy:
            os.remove(k)
            
def read_json(fname,
             object_pairs_hook=collections.OrderedDict,
             **kwargs):
    
    res = pyext.file__callback(
        fname,
        functools.partial(
            json.load,
            object_pairs_hook=collections.OrderedDict,
            **kwargs
        ),
    )
    return res

def job__script(scriptPath, ODIR=None, opts='', ENVDIR = None, silent= 0,
#                 baseFile=0,
                prefix = 'results',
               ):
    base__check()
    BASE = os.environ['BASE'].rstrip('/')    
    baseLog = base__file('LOG',force=1)
    scriptBase = pyext.getBname(scriptPath,)    
    scriptFile = os.path.basename(scriptPath)
#     if ODIR is None:
#         ODIR = scriptBase
#     ODIR = ODIR.rstrip('/')
    prefix = prefix.rstrip('/')
    if ODIR is None:
        ODIR = '{BASE}/{prefix}/{scriptBase}'.format(**locals())
    CMD='''
mkdir -p {ODIR} || exit 1
time {{ 
    set -e;      
    cp -f {scriptPath} {ODIR}/{scriptFile}; 
    cd {ODIR}; 
    chmod +x {scriptFile} ;
    ./{scriptFile} {opts}; 
    touch {ODIR}/DONE; 
}} 2>&1 | tee {ODIR}/{scriptFile}.log | tee -a {baseLog};
exit ${{PIPESTATUS[0]}}; 
'''.format(**vars())
    
    if ENVDIR is not None:
        CMD = 'source {ENVDIR}/bin/activate\n'.format(**locals()) + CMD

    res, err  = pysh.shellpopen(CMD,silent=silent)
    success = (err == 0)
    return success, res

def is_ipython():
    try:
        get_ipython
        return 1
    except:
        return 0
hasIPD = is_ipython()

if hasIPD:
    import IPython.display as ipd

def showMD(s):
    if pyext.hasIPD:
        ipd.display(ipd.Markdown(s))
        
def MDFile(fname):
    s='[{0}]({0})'.format(fname)
    return showMD(s)


def printlines(lst,fname = None,
               callback=MDFile,encoding='utf8',
              lineSep='\n',castF=unicode):
    s = castF(lineSep).join(map(castF,lst))
    if fname is None:
        print(s)
    else:
# f = open('test', 'w')
# f.write(foo.encode('utf8'))
# f.close()
        with open(fname,'w') as f:
            print >>f,s.encode(encoding)
        if callback is not None:
            callback(fname)

def nTuple(lst,n,silent=1):
    """ntuple([0,3,4,10,2,3], 2) => [(0,3), (4,10), (2,3)]
    
    Group a list into consecutive n-tuples. Incomplete tuples are
    discarded e.g.
    
    >>> group(range(10), 3)
    [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    """
    if not silent:
        L = len(lst)
        if L % n != 0:
            print '[WARN] nTuple(): list length %d not of multiples of %d, discarding extra elements'%(L,n)
    return zip(*[lst[i::n] for i in range(n)])

def get__LeafFiles(DIR):
    ''' Drill down to leaf files of a directory tree if the path is unique.
    '''
    assert os.path.exists(DIR),'%s not exist'%DIR
    DIR = DIR.rstrip('/')
    if not os.path.isdir(DIR):
        return [DIR]
    else:
        cmd = 'ls -LR %s'%DIR
        res = subprocess.check_output(cmd,shell=1)
        res = re.split(r'([^\n]*):',res)[1:]
        it = pyext.nTuple(res,2,silent=0)
        DIR, ss = it[0];
        for dd,ss in it[1:]:
            NEWDIR, ALI = dd.rsplit('/',1)
            assert NEWDIR == DIR, 'Next directory %s not contained in %s'%(dd,DIR)
            DIR = dd 
        res = [ '%s/%s'%(DIR,x) for x in LinesNotEmpty(ss)]
        return res
LeafFiles  = get__LeafFiles  
def LeafDict(d):
    ''' Walking to leaves of a nested dictionary
    '''
    if isinstance(d,dict):
        res = [pyext.LeafDict(dd) for dd in d.values()]
        res = sum(res,[])
        return res
    else:
        if isinstance(d,list):
            pass
        else:
            d = [d]
        return d
    
def LinesNotEmpty(sub):
    sub = [ x for x in sub.splitlines() if x]
    return sub

def meta__label(geneMeta,geneInfo,indexName='GeneAcc'):
    geneInfo = geneInfo.copy()
    geneInfo.index = geneInfo.index.str.split('.').str.get(0)
    geneInfo = pyext.mergeByIndex( geneInfo, geneMeta,how='left')
    geneInfo.index.name = indexName
    return geneInfo

def dict2flat(dct,sep='_',concat='='): 
    ''' Pack a depth-1 dictionary into flat string
    '''
    s = sep.join(['%s%s%s'%(k,concat,v) for k,v in dct.items()])
    return s

def mergeByIndex(left,right,how='outer',as_index = 0, **kwargs):
    dcts = [{'name': getattr(left,'name','left')},
            {'name': getattr(right,'name','right')},
           ]
    suffixes = ['_%s'%x for x in map(pyext.dict2flat,dcts)]

    df = pd.DataFrame.merge(left,right,
                       how = how,
                        left_index=True,
                        right_index=True,
                       suffixes= suffixes
                      )
    if as_index:
        df = df.iloc[:,:1]
    return df

if __name__=='__main__':
    def test_d(d):
        try:
            return LeafFiles(d)
        except Exception as e:
            print e
    ##### To be refactored to work everywhere
    d = '/home/feng/test_envs/tdir/something/A/B/C/D/'
    test_d(d)
    d = '/home/feng/test_envs/tdir/something/'
    print test_d(d)
    d = '/home/feng/syno3/PW_HiSeq_data/ChIP-seq/Raw_data/182C/Bd_ELF3-44645602/FASTQ_Generation_2018-06-06_03_43_21Z-101158414/182C_721_L001-ds.e1f926b50b5f4efd99bcffeca5fb75a0'
    print test_d(d)
    d = '/home/feng/syno3/PW_HiSeq_data/ChIP-seq/Raw_data/182C/Bd_ELF3-44645602/FASTQ_Generation_2018-06-06_03_43_21Z-101158414/182C_721_L001-ds.e1f926b50b5f4efd99bcffeca5fb75a0/Bd-ELF3OX-SD-ZT16-_S2_L001_R1_001.fastq.gz'
    # print test_d(d)
    LeafFiles(d)

def log2p1(x):
    return np.log2(1. + x)

def log2p1_ln2(x):
    y  = np.log2(1 + np.log(2)*x)
    return y

def entropise(ct,axis=None, normed=0):
    ''' Transform count into -p *logP, summation of which is entropy
'''
    if not normed:
        p = ct/np.sum(ct,axis=axis,keepdims=True)
    else:
        p = ct
    logP = np.nan_to_num(np.log2(p))
    r = p * - logP    
    return r

def getH(proba,axis = None):
    part = pynp.logsumexp(proba, axis=axis,keepdims=1)
    proba = proba - part
    proba = np.exp(proba)
    H = entropise(proba,axis=axis,normed=1).sum(axis=axis)    
    return H

def canonic_npy(s):
    f = lambda s: s if s.endswith('.npy') else '%s.npy'%s
    res = pyfop.safeMap(f,s)        
    return res

def dict__wrap(d):
    for k in d.keys():
        d[k] = [d[k]]
    return d
def dict__combine(dcts):
    for i,d in enumerate(dcts): 
        if i==0:
            res = dict__wrap(d)
        else:
            dict__wrap(d)
            for k in d.keys():
                if k not in res.keys():
                    res[k] = d[k]
                else:
                    res[k] += d[k]
    return res

##### Multiprocessing map
import multiprocessing as mp
def mp_map(f,lst,n_cpu=1, chunksize= None, callback = None, 
           NCORE=None,
#            kwds = {}, 
           **kwargs):
    if NCORE is not None:
        n_cpu = NCORE
    if n_cpu > 1:
        p = mp.Pool(n_cpu,**kwargs)
        OUTPUT=p.map_async(f,lst, chunksize=chunksize, 
#                            kwds = kwds,
                           callback = callback).get(999999999999999999) ## appx. 7.6E11 years
#         OUTPUT = p.map(f,lst)
        p.close()
        p.join()
    else:
        OUTPUT = map(f,lst)
    return OUTPUT

def MapWithCache(f,it,ALI='Test',nCPU=1,force=0):
    print ('[MSG] Mapping function:   Under cahce alias: %s'%(ALI))
    fmt = '%s_%%05d'%(ALI)
    def cb(resLst):
        for i,res in enumerate(resLst):
            fname = canonic_npy([fmt%i])[0]
            np.save(fname,res)    
            print 'Saving to %s'%fname
        return resLst
    fname = canonic_npy([fmt%0])[0]
    print (fname)
    if os.path.exists(fname) and not force:
        res = np.load(fname).tolist()
        p = None
    else:        
        p = mp.pool.ThreadPool(nCPU)        
        res = p.map_async(f,it,callback=cb)
    return res,p