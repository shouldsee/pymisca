# reload(pyext.pymisca.shell)
# reload(pyext.pymisca.header)
# reload(pyext)

import os,sys,re,glob
import functools,itertools
import pymisca.shell as pysh
import pymisca.header
# import pymisca.ext as pyext
import inspect

class Worker(object):
    '''
    An environmental object that knows how to execute commands
    '''
    def __enter__(self):
        pass
    def __exit__(self):
        pass
    
    
    @property
    def scope(self):
        '''
        A crucial invention that make sure the namespace is up to date
        '''
        scope = self.frame.f_back.f_locals
        return scope
    
    def __del__(self):
        ### Clear circular reference
        del self.frame
        super(Worker,self).__del__()
        
    def __init__(self,
                 frame = None,
                 scope=None, 
                 BASE = None,
                 baseDir= None, silent=0, check=True,
                stack = None):
        '''
        Reference for access parent namespace: https://stackoverflow.com/questions/9130453/access-parent-namespace-in-python
        '''
        if frame is None:
            frame = inspect.currentframe()
        self.frame = frame

        if baseDir is not None:
            assert BASE is None,'Cannot specify both'
            BASE = baseDir
            
        if BASE is None:
            BASE = pymisca.header.base__check('BASE', strict=1)
#         self.baseDir = pymisca.header.base__check(default=baseDir,strict=0)
        else:
            pass
        self.BASE = os.path.realpath(BASE)
        self.silent = silent
        self.check = check
        self.stack = stack 
        
        if not self.silent:
            print (self.msg__BASE())
            
#         if scope is None:
#             scope = self.scope
# #             scope = pymisca.header.runtime__dict()
#         self._scope = scope
    @property 
    def baseDir(self):
        '''legacy'''
        return self.BASE 
                    
    def msg__BASE(self):
        s = '[Worker]: Inputing and outputing at "{self.baseDir}"'.format(**locals())
        return s
    
    def msg__baseDir(self):
        return self.msg__BASE()
        

    def setEnv(self,**kw):
        for k,v in kw.items():
            os.environ[k] = unicode(v)
        
        
    def makeJob(self, 
            CMD,
#             error = 'raise',
            silent=None,             
            shell = True, 
            jobDir = None,
            check=None, 
#             resultSlice = None,
            
#             scriptPath, 
            ODIR=None, 
            opts='', 
            ENVDIR=None, 
            DATENOW='.', 
            baseFile=0, 
            baseOut=1, 
            inplace=False,
            prefix='results',
                **kw
            
           ):
        if silent is None:
            silent = self.silent
        if check is None:
            check = self.check
        if jobDir is None:
            jobDir = '.'
#         if resultSlice is None:
#             resultSlice = slice(None,None)
        CMD = self.fill(CMD)
#         CMD = CMD.format(**self.scope)
        def job():
                        
#             def _job():
            if 1:
                with pymisca.tree.getPathStack([jobDir,self.BASE],stack=self.stack) as stack:
                    self.setEnv( BASE= stack.d.abspath() )

#                 self.setEnv(BASE=self.baseDir)
#                 self.setEnv(BASE=stack.d)
                    p = pysh.shellpopen(CMD, silent=silent, shell=shell, **kw)
                    res = pysh.pipe__getSafeResult(p, CMD=CMD,check=check)
                    if not self.check:
                        suc,res = res
                    return res
            
#             return _job()
#             return pyext.func__inDIR(_job, DIR=jobDir)
        
        return job
    def runJob(self, 
            CMD,
#             error = 'raise',
            silent=None,             
            shell = True, 
            jobDir = None,  
               **kw
              ):
        job = self.makeJob(CMD,
#                            error=error,
                           silent=silent,shell=shell,jobDir=jobDir,**kw)
        res = job()
        return res
    def fill(self, fmt,**kw):
        kw.update(**self.scope)
        kw.update(**vars(self))
        kw['baseDir'] = kw['BASE']
#         res = fmt.format(**self.scope)
        res = fmt.format(**kw)
        return res
    
    @staticmethod
    def lastLine(res):
        res = res.splitlines()[-1]
        print ('[LastLine]%s'%res)
        return res

    
    
if __name__ =='__main__':
    w = Worker()
    w.runJob('echo "helloWorld"')
    w.runJob('echo something; exit 0')
    print ('[DONE]')

