import os,sys
import re
import subprocess
import shutil
import StringIO
import warnings


import pymisca.io_extra
##### Shell util
import tempfile
import subprocess
import re
import collections
import functools,itertools
# import pymisca.header as pyheader

pysh=sys.modules[__name__]

import json
import datetime

from shutil import copystat
class Error(EnvironmentError):
    pass

# import os
import stat
def file__chmod_px(fname):
    st = os.stat(fname)
    os.chmod(fname, st.st_mode | stat.S_IEXEC)
    
def file__safeInode(x):
    if not os.path.exists(x):
        return -1
    else:
        return os.stat(x).st_ino
    
import zipfile  ##file__unzip

def dir__iterFiles(DIRNAME,start=1):
    _this_func = dir__iterFiles
    if start:
        DIRNAME = path.Path(DIRNAME)
    for root, dirs, files in os.walk(DIRNAME):
#         if root
        for f in files:
            yield DIRNAME / f
        for d in dirs:
            for f in _this_func(DIRNAME / d,0):
                yield f
                
                
def file__unzip(FNAME,DIRNAME=None):
    if DIRNAME is None:
        DIRNAME = FNAME + '.unzip'
    
    with zipfile.ZipFile( FNAME, 'r') as zip_ref:
        zip_ref.extractall(unicode(DIRNAME)+'.partial')    
    shutil.move(DIRNAME+'.partial',DIRNAME)
    return DIRNAME
    

def dir__getSize(start_path = '.'):
    '''
    Source:https://stackoverflow.com/a/1392549/8083313
    '''
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def dir__link(src,dst,
              symlinks=False, ignore=None,
              force = 0,
              copyFunc=None,
              **kw):
    assert copyFunc is None,("auto produced")
    
    res = dir__copy(src,dst, 
                    symlinks=symlinks, ignore = ignore,
                    copyFunc = lambda x,y:file__link(x,y,force=force,**kw),
                    **kw)
    return res

def dir__copy(src, dst, symlinks=False, ignore=None,
             copyFunc = shutil.copy2):
    _this_func = dir__copy
    """
    Adapted from python2 shutil.copytree()
    
    Recursively copy a directory tree 
    with an arbitrary copyFunc(src,dest) instead of copy2

    The destination directory must not already exist.
    If exception(s) occur, an Error is raised with a list of reasons.

    If the optional symlinks flag is true, symbolic links in the
    source tree result in symbolic links in the destination tree; if
    it is false, the contents of the files pointed to by symbolic
    links are copied.

    The optional ignore argument is a callable. If given, it
    is called with the `src` parameter, which is the directory
    being visited by copytree(), and `names` which is the list of
    `src` contents, as returned by os.listdir():

        callable(src, names) -> ignored_names

    Since copytree() is called recursively, the callable will be
    called once for each directory that is copied. It returns a
    list of names relative to the `src` directory that should
    not be copied.

    XXX Consider this example code rather than the ultimate tool.

    """
    
#     assert os.path.isdir(src),("Not a directory",src)
    names = os.listdir(src)
    if ignore is not None:
        ignored_names = ignore(src, names)
    else:
        ignored_names = set()
        
    if not os.path.isdir(dst):
        assert not os.path.isfile(dst),(dst,src)
        os.makedirs(dst)
        
    errors = []
    for name in names:
        if name in ignored_names:
            continue
        srcname = os.path.join(src, name)
        dstname = os.path.join(dst, name)
        try:
            if symlinks and os.path.islink(srcname):
                linkto = os.readlink(srcname)
                os.symlink(linkto, dstname)
            elif os.path.isdir(srcname):
                _this_func(srcname, dstname, symlinks, ignore, copyFunc)
            else:
                # Will raise a SpecialFileError for unsupported file types
                copyFunc(srcname, dstname)
        # catch the Error from the recursive copytree so that we can
        # continue with other files
        except Error, err:
            errors.extend(err.args[0])
        except EnvironmentError, why:
            errors.append((srcname, dstname, str(why)))
    try:
        copystat(src, dst)
    except OSError, why:
        if WindowsError is not None and isinstance(why, WindowsError):
            # Copying file access times may fail on Windows
            pass
        else:
            errors.append((src, dst, str(why)))
    if errors:
        raise Error, errors
        
def file__getDateTime(fname,attr):
    st = os.stat(fname)
    ts = getattr( st, attr)
    dt = datetime.datetime.fromtimestamp(ts)
    return dt

def file__getModifiedTime(fname,attr='st_mtime'):
    return file__getDateTime(fname,attr)

        
def file__link(IN,OUT,force=0, forceIn = False, link='link',relative=1):
    
    linker = getattr(os,link)
    
    #### Make sure input is not an empty file
    if not file__notEmpty(IN) and link !='symlink':
        if not forceIn:
            return IN
    if os.path.abspath(OUT) == os.path.abspath(IN):
        return OUT
        
#     if os.path.exists(OUT):
    if os.path.isfile(OUT) or os.path.islink(OUT):
        if force:
#             assert os.path.isfile(OUT) or os.path.islink(OUT)
            os.remove(OUT)
#             print ('removed[OUT]%s'%OUT)
        else:
            assert 'OUTPUT already exists:%s'%OUT
    else:
        pass
    
    
    try:
        if link=='symlink':
            IN  =os.path.realpath(IN)
            
#             OUT = os.path.realpath(OUT)
#             OUT = os.path
            if relative:
                IN = os.path.relpath( IN,os.path.dirname(OUT))
        linker(IN,OUT)
    except Exception as e:
        d = dict(PWD=os.getcwdu(),
                 IN=IN,
                 OUT=OUT)
        print(json.dumps(d,indent=4))
#         print ('[PWD]%s'%os.getcwdu())
#         print('l')
        raise e
        
    return OUT


def file__rename(d,force=0, copy=1, **kwargs):
    for k,v in d.items():
        DIR = os.path.dirname(v)
        if DIR:
            if not os.path.exists(DIR):
                os.makedirs(DIR)
        file__link(k,v,force=force,**kwargs)
        if not copy:
            if os.path.isfile(k):
                os.remove(k)        

def read__json(fname,
             object_pairs_hook=collections.OrderedDict,
              parser = json,
             **kwargs):
    kwargs.update(dict(object_pairs_hook=object_pairs_hook,))
#     if hasattr(fname,'read'):
#         func = json.loads
#         res = func(fname.read(),**kwargs)
#     else:
    if 1:
        func = parser.load
        res = file__callback(
            fname,
            functools.partial(
                func,**kwargs
            )
        )
    return res
read_json = read__json

def file__callback(fname,callback,mode='r'):
    if isinstance(fname,basestring):
        with open(fname,mode) as f:
            res = callback(f)
    else:
        f = fname
        res = callback(f)
    return res

def file__notEmpty(fpath):  
    '''
    Source: https://stackoverflow.com/a/15924160/8083313
    '''
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def module__getPath(module,
#                     callback = None,
                   ):
    if isinstance(module,basestring):
        module = sys.modules[module]
    d = vars(module)
    PKG_DIR = d['__path__'][0]
    return PKG_DIR

def job__shellexec(d):
    try:
        d['result'] = shellexec(d['CMD'])
        d['suc'] = True
    except:
        d['result'] = None
        d['suc'] = False
    return d

def dir__curr(silent = 1):
    res = shellexec('pwd -L',silent=silent).strip()
    return res


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



def xargsShellCMD(CMD, lst):
# def getHeaders(lst):
    with tempfile.TemporaryFile() as f:
        f = pymisca.io_extra.unicodeIO(handle=f)
        f.write(u' '.join(map(quote,lst)))
        f.seek(0)
        p = subprocess.Popen('xargs {CMD}'.format(**locals()),stdin=f,stdout=subprocess.PIPE,shell=True)
        res = p.communicate()[0]
    return res
def getHeaders(lst):
    res = xargsShellCMD('head -c2048',lst)
    lst = re.split('==> *(.+) *<==',res)[1:]
    lst = [x.strip() for x in lst] 
    res = collections.OrderedDict(nTuple(lst,2))
    return res
def getTails(lst):
    res = xargsShellCMD('tail -c2048',lst)
    lst = re.split('==> *(.+) *<==',res)[1:]
    lst = [x.strip() for x in lst] 
    res = collections.OrderedDict(nTuple(lst,2))
    return res
def getSizes(lst):
    res = xargsShellCMD('du',lst)
    res = collections.OrderedDict([x.split('\t')[::-1] for x in res.splitlines()])
    return res
########

def dict2kwarg(params):
    s = ' '.join('--%s %s' % (k,v ) for k,v in params.items())
    return s



def file__cat(files,ofname='temp.txt',silent=1,bufsize=1024*1024*10):
    with open(ofname,'wb') as wfd:
        for f in files:
            with open(f,'rb') as fd:
                shutil.copyfileobj(fd, wfd, bufsize)    
    return ofname

def file__header(fname,head = 10,silent=1,ofname = None, 
                 bufferClass = pymisca.io_extra.unicodeIO ):
    if ofname == 'auto':
        ofname = fname + '.head%d'%head
    cmd = 'head -n{head} {fname}'.format(**locals())
    if ofname is not None:
        cmd = cmd + '>{ofname}'.format(**locals())
    res = shellexec(cmd, silent=silent)
    if bufferClass is not None:
        res  = bufferClass(res)
#     res = bufferClass
    if ofname is not None:
        return ofname
    else:
        return res

def file__lineCount(fname):
    with open(fname,"r") as f:
        for i, _ in enumerate(f):
            pass
    return i + 1

def real__dir(fname=None,dirname=None,mode=0777):
    if dirname is None:
        assert fname is not None
        dirname = os.path.dirname(fname)
    else:
        assert fname is None
        
    if not os.path.exists(dirname) and dirname!='':
        os.makedirs(dirname,mode=mode)
    return dirname
dir__makeIfNeed = real__dir

def symlink(fname,ofname = None,
            relative = 0,
            silent=1,debug=0,**kwargs):
#     if ODIR is None
    if not os.path.exists(fname):
        warnings.warn('trying to symlink non-existent file:%s'%fname)
    if ofname is None:
        ofname = './.'
    ODIR = real__dir(fname=ofname)
    if relative:
        fname = os.path.abspath(fname)
    else:
        fname = os.path.relpath(fname,ODIR)
    
        
    cmd = 'ln -sf {fname} {ofname}'.format(**locals())
    shellexec(cmd,silent=silent,debug=debug)
    return ofname
def envSource(sfile,silent=0,dry=0,
              executable=None,outDict=None):
    if outDict is None:
        outDict = os.environ
#     import os
    '''Loading environment variables after running a script
    '''
    command = 'source %s&>/dev/null ;env -0 ' % sfile
    # print command
#     res = subprocess.check_output(command,stderr=subprocess.STDOUT,shell=1)
    res = shellexec(command,silent=silent,executable=executable)
    for line in res.split('\x00'):
        (key, _, value) = line.strip().partition("=")
        if not silent:
            print key,'=',value
        if not dry:
            outDict[key] = value
    return outDict

def real__shell(executable=None):
    if executable is None:
        executable = os.environ.get('SHELL','/bin/bash')
    return executable

def silentShellexec(cmd,silent=1,**kwargs):
    res = shellexec(cmd=cmd, silent=silent,**kwargs)
    return res

def shellexec(cmd,debug=0,silent=0,
              executable=None,
              encoding='utf8',error='raise',
#               env = None,
              shell = 1,
              getSuccessCode = False,
              **kwargs
             ):
#     _env = os.environ.copy()
#     _env = _env.update(env or {})
    
    executable = real__shell(executable)
    if silent != 1:
        buf = '[CMD]{cmd}\n'.format(**locals())
        if hasattr(silent,'write'):
            silent.write(buf)
        else:
            sys.stderr.write(buf)
#         print (cmd)
    if debug:
        return 'dbg'
    else:
        try:
            res = subprocess.check_output(cmd,shell=shell,
#                                           env=_env,
                                         executable=executable,
                                         **kwargs)

#             p.stdin.close()
            if encoding is not None:
                res=  res.decode(encoding)
            
            ## [PATCH]0819
#             res = res.strip()
            res = res.rstrip()
            suc = True

        except subprocess.CalledProcessError as e:
            errMsg = u"command '{}' return with error (code {}): {}\
                ".format(e.cmd, e.returncode, e.output)
            e = RuntimeError(
                    errMsg)
            if error=='raise':
                raise e
            elif error=='ignore':
                
#                 res = 'FAIL'
                res = (errMsg)
                suc = False
                
    
         #### allow name to be returned
        if getSuccessCode:
            return suc,res
        else:
            return res
    
def getMD5sum(fname,silent=1):
    res = shellexec('md5sum %s'%fname,silent=silent)[:32]
    return res

def pipe__getResult(p, input = None, encoding='utf8', check=False, errEmptyStdout=True,CMD=None):
    stdout,stderr = p.communicate(input=input)
    res = stdout
    suc = p.returncode==0
    if errEmptyStdout:
        suc = suc & (res!='')  
    
    if encoding is not None:
        res = res.decode(encoding)
        
    if not check:
        return suc, res
    
    if check:
        assert suc,\
        'Command "{CMD}" returned error or empty output:\n[stdout]:{stdout}\n[stderr]:{stderr}'.format(**locals())        
#         assert suc,'return code {p.returncode} != 0 or output is empty' .format(**locals())
        return res
    
def pipe__getSafeResult(p,check=True,**kw):
    return pipe__getResult(p,check=check,**kw)

def shellpopen(cmd,
                 stdin=subprocess.PIPE,
                 stdout=subprocess.PIPE,
                stderr = subprocess.PIPE,
                debug=0,silent=0,executable=None,
                 shell = 1,
                 bufsize= 1,
                 **kwargs):
    executable = real__shell(executable)
    if not silent:
        sys.stdout.write(u'[CMD] %s\n'%cmd)
    if debug:
        return 'dbg'
    else:
        p = subprocess.Popen(
                     cmd,
                     shell=shell,
                     bufsize=bufsize,
                     executable=executable,
                     stdin=stdin,
                     stdout = stdout,
                    stderr = stderr,

                    **kwargs)
        return p
#         res = p.communicate()[0]
#         return res,p.returncode




def pipe__cat(p=None,fname= None, **kwargs):
    if fname is None:
        fname = '-' 
    p = pysh.shellpopen('cat "{fname}"'.format(**locals()), **kwargs)
    return p



def pipe__wig2bed(p = None, stdin=None, fname = '-', ):
    assert fname is not None
    if p is not None:        
#         stdin = p.stdout
        pass
    else:
        p = pipe__cat(fname=fname)
    p = pysh.shellpopen('convert2bed -i wig',  stdin=p.stdout)
    return p


class ShellPipe(list):
    def __init__(self, 
                 p0 = None, 
                 fname=None):
        super(ShellPipe,self).__init__()
        if p0 is None:
            p0 = pysh.pipe__cat(fname=fname)
            cmd= 'cat %s'%fname
        else:
            assert fname is None,'conflict "fname" and "p0"'
            cmd = '' ### cannot capture cmd from an existing pipe
        self.p = self.p0 = p0
        self.addElement(p=self.p,
                        cmd=cmd)
        self.stdin = self.p0.stdin
        
    def addElement(self, p,cmd):
#         self.append(collections.OrderedDict(cmd=cmd))
        self.append(collections.OrderedDict(p=p, cmd=cmd))
        
    def checkResult(self, check=True, cmd='head',**kw):
        self.p0.stdin.close()
        if cmd:
            self.chain(cmd,)
        res =  pysh.pipe__getResult( self.p, check=check,**kw)            
        return res
    
    def readIter(self,it,
                 delay = False,
                 lineSep = '',
#                  mapper=map
                ):
        it = (u'%s%s'%(x,lineSep) for x in it)
        
        if delay:
            mapper = itertools.imap()
        else:
            mapper = map
        return mapper(self.p0.stdin.write,it)

    def chain(self, cmd, stdin=None,**kw):
        if stdin is None:
            stdin = self.p.stdout
        self.p = pysh.shellpopen(cmd=cmd, stdin=stdin,**kw)
        self.addElement(p=self.p,cmd=cmd)


# shellexec = shellopen
