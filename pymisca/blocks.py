# import pymisca.ext as pyext
import pymisca.header as pyheader
import pymisca.shell as pysh
import pymisca.util__fileDict
import os
import funcy

'''
Everything here needs to be backwards compatible
'''

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
        ifname='{INPUTDIR}/FILE.json'.format(**locals())
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
            pyheader.base__file(''),
#             pyext.shellexec('real $BASE')
        )
        print(REL)
        if save:
            pymisca.util__fileDict.main(
                argD = {'RELPATH': REL}
            )
        return REL
    
    return workF


