from __future__ import absolute_import

from pymisca.header import *
import os,sys,subprocess
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

def base__file(fname, BASE=None, HOST='BASE', force = False,silent= 1):
    '''find a file according under the directory of environment variable: $BASE 
    '''
    if not isinstance(BASE, basestring):
        BASE  = os.environ.get( HOST,None)
        assert BASE is not None
    BASE = BASE.rstrip('/')
    res = os.path.join(BASE,fname)
    if BASE.startswith('/'):
        existence = os.path.exists(res)
        if not force:
            assert existence,'BASE={BASE},res={res}'.format(**locals())
        else:
            if not existence:
                pysh.shellexec('touch {res}'.format(**locals()), silent=silent)
                
        with open('%s/TOUCHED.list' % BASE, 'a') as f:
            f.write(fname +'\n')
    return res
# def base__touch(fname):

# print ('[__main__]',__main__)
def execBaseFile(fname,):
    fname = pyext.base__file(fname)
    g= vars(sys.modules['__main__'])
#     g = __main__.globals()
    res = execfile(fname, g, g)
#     exec(open(fname).read(), g)
    return


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
def base__check():
    if os.environ.get('BASE',None) is None:
        PWD =  os.getcwd()
        print ('[WARN] $BASE not set,defaulting to PWD:{PWD}'.format(**locals()))
        os.environ['BASE'] = PWD

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