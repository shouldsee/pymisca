# import pymisca.ext as pyext
import pymisca.header as pyheader
import pymisca.shell as pysh
import pymisca.util__fileDict
import os
import funcy

import pymisca.ext as pyext

'''
Everything here needs to be backwards compatible
'''

def compute__OUTDIR(**args):
    '''
    Compute the directory to output
    '''
    def func(**args):

        INPUTDIR = args['INPUTDIR']
        pathLevel = args['pathLevel']
        inplace  = args['inplace']
        prefix = args.get('prefix','')
        suffix = args.get('suffix','')

        if not inplace:
            CWD = ''
            OUTDIR = pyext.splitPath(INPUTDIR,pathLevel)[1]
        else:
            CWD = pyext.base__file(INPUTDIR)
            OUTDIR = pyext.os.path.basename(
                pyext.runtime__file(silent=0)
            ).rsplit('.',1)[0]
#             del CWD
        OUTDIR = os.path.join(prefix,OUTDIR)
        OUTDIR = os.path.join(OUTDIR,suffix)
        OUTDIR = os.path.join(CWD,OUTDIR)
        OUTDIR = os.path.normpath(OUTDIR)
        

        return OUTDIR
    return funcy.partial(func,**args)

def bookkeep(INPUTDIR=None,
             argD = {},
             ofname='FILE.json'):
    '''
    returns a function that create FILE.json in CWD by combining 
    argD > FILE.json > {INPUTDIR}/FILE.json
    
    2019/03/07
    '''
    if INPUTDIR is not None:
        INPUTDIR=pyheader.base__file(INPUTDIR)
        ifname=os.path.join(INPUTDIR,'FILE.json')
    else:
        ifname = None
    documentF = funcy.partial(
            pymisca.util__fileDict.main,
            ifname = ifname,
            ofname=ofname,
            argD = argD,
    )
    return documentF

def shellsafe(CMD=None, prefix= 'set -e; set -o pipefail;\n'):
    '''
    returns a function that executes CMD in CWD
    
    2019/03/07
    '''
    assert CMD is not None
    workF = funcy.partial(
        pysh.shellexec,
                  cmd =  prefix + CMD,    
                 )            
    return workF

def printPath(
    CMD='echo $BASE;echo $PWD;\
    realpath --relative-to="$BASE/" "$PWD" --no-symlinks', 
     prefix= 'set -e; set -o pipefail;\n'):
    '''
    returns a function that executes CMD in CWD
    
    2019/03/08
    '''
    return shellsafe(CMD=CMD,prefix=prefix)

def printRelPath(save=True):
    def workF():
#         CDIR = pyext.os.getenv('PWD')
        CDIR = os.getcwdu()
#         CDIR = pyext.shellexec('pwd -P')
#         print (CDIR)
        REL = os.path.relpath( 
            CDIR, 
            pyheader.base__file(),
#             pyext.shellexec('real $BASE')
        )
        print(REL)
        if save:
            pymisca.util__fileDict.main(
                argD = {'RELPATH': REL}
            )
        return REL
    
    return workF

def symlink(renamer,relative = True,**kwargs):
    assert hasattr(renamer,'items')
    def job():
        for k,v in renamer.items():
            pysh.symlink(fname=k,ofname=v,relative =relative,**kwargs)
        return renamer.values()
    return job

def job__script(*a,**kw):
#     def workF(*a,**kw):
#         return pyext.job__script(*a,**kw)
    return funcy.partial(pyext.job__script,*a,**kw)
        
