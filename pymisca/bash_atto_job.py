#!/usr/bin/env python2
'''
Command line interface to pymisca.atto_job.AttoJob classes
'''
import pymisca.ext as pyext
from pymisca.pyhawk import parser,parser__getArgTree
import sys,os
parser.description = __doc__
parser.add_argument('--CLASS',type=unicode,default='pymisca.atto_job.ModuleJob')
parser.add_argument('--DEBUG',type=int,default=0)
parser.add_argument('--STRICT',type=int,default=0)
def bash_atto_job(argTree=None,**kw):
    if argTree is None:
        argTree = _getArgTree()
    argTree.update(kw)
    CLASS = argTree['CLASS']
    DEBUG = argTree['DEBUG']
    STRICT = argTree['STRICT']
    if not STRICT:
        #### Hacking through ModuleJob()
        if CLASS.endswith('ModuleJob'):
            argTree['DATA'] = argTree.get('DATA',argTree.copy())

    if DEBUG:
        sys.stderr.write('%s\n'%pyext.ppJson(argTree,default=repr))
    MODULE, CLASS = CLASS.rsplit('.',1)
    CLASS = getattr( __import__(MODULE, fromlist=[CLASS]),CLASS)
    res = CLASS(argTree)
    return res
main = bash_atto_job

def _getArgTree(parser=parser):
    return parser__getArgTree(parser)

if __name__ == '__main__':
    argTree = _getArgTree()
#     argTree = parser__getArgTree(parser)
    main(argTree)

#     assert 0,pyext.ppJson(argTree)