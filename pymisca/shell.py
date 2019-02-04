import os,sys
import re
import subprocess
import shutil
import StringIO

def dict2kwarg(params):
    s = ' '.join('--%s %s' % (k,v ) for k,v in params.items())
    return s



def file__cat(files,ofname='temp.txt',silent=1,bufsize=1024*1024*10):
    with open(ofname,'wb') as wfd:
        for f in files:
            with open(f,'rb') as fd:
                shutil.copyfileobj(fd, wfd, bufsize)    
    return ofname

def file__header(fname,head = 10,silent=1,ofname = None):
    if ofname == 'auto':
        ofname = fname + '.head%d'%head
    cmd = 'head -n{head} {fname}'.format(**locals())
    if ofname is not None:
        cmd = cmd + '>{ofname}'.format(**locals())
    res = shellexec(cmd, silent=silent)
    res = StringIO.StringIO(res)
    if ofname is not None:
        return ofname
    else:
        return res

def real__dir(fname=None,dirname=None,mode=0777):
    if dirname is None:
        assert fname is not None
        dirname = os.path.dirname(fname)
    else:
        assert fname is None
        
    if not os.path.exists(dirname):
        os.makedirs(dirname,mode=mode)
    return dirname

def symlink(fname,ofname = None,silent=1,debug=0,**kwargs):
    real__dir(fname=ofname)
    fname = os.path.abspath(fname)
    if ofname is None:
        ofname = '.'
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

def shellexec(cmd,debug=0,silent=0,executable=None):
    executable = real__shell(executable)
    if silent != 1:
        buf = cmd +'\n'
        if hasattr(silent,'write'):
            silent.write(buf)
        else:
            sys.stdout.write(buf)
#         print (cmd)
    if debug:
        return 'dbg'
    else:
        try:
            res = subprocess.check_output(cmd,shell=1,
                                         executable=executable)

#             p.stdin.close()
            return res
        except subprocess.CalledProcessError as e:
#         except Execption as e:
#             print e.output
#             raise e
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
        return res
    
def getMD5sum(fname,silent=1):
    res = pyutil.shellexec('md5sum %s'%fname,silent=silent)[:32]
    return res

def shellpopen(cmd,debug=0,silent=0,executable=None):
    executable = real__shell(executable)
    if not silent:
        print (cmd)
    if debug:
        return 'dbg'
    else:
        p = subprocess.Popen(
                     cmd,
                     shell=1,
                     bufsize=1,
                     executable=executable,
                     stdout=subprocess.PIPE,
                     stdin=subprocess.PIPE)
        res = p.communicate()[0]

        return res,p.returncode
# shellexec = shellopen