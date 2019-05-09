import pymisca.shell as pysh
import path
import sys
# def abspath
def chdir(DIR):
    def getOS(self):
        os = sys.modules.get('os',None)
        if os is None:
            import os
        return os
    self._old_dir = ODIR = pysh.shellexec('pwd -L',silent=1).strip()

    ### abspath( ) with custom cwd with symlink deferred
    self._abspath = os.path.normpath(
        os.path.join(self._old_dir, self)
    )
    os._getcwd = os.getcwd
    os._getcwdu = os.getcwdu
    os.getcwd = lambda : str(self._abspath)
    os.getcwdu = lambda : unicode(self._abspath)
    
class FrozenPath(path.Path):
    '''
    Overwrite os.getcwd() so that symlink can be returned
    '''
    def __init__(self,*a,**kw):
        super(FrozenPath,self).__init__( *a,**kw)
        self.debug = 0
        self.printOnExit = False
        
    def getOS(self):
        os = sys.modules.get('os',None)
        if os is None:
            import os
        return os
        
    def __enter__(self):
        os = self.getOS()
        self._old_dir = ODIR = pysh.shellexec('pwd -L',silent=1).strip()
        
        ### abspath( ) with custom cwd with symlink deferred
        self._abspath = os.path.normpath(
            os.path.join(self._old_dir, self)
        )
        if self.debug:
            print ('[FrozenPath]registering _old_dir:%s'%self._old_dir)
            print ('[FrozenPath]registering _abspath:%s'%self._abspath)
        self._getcwd  = os._getcwd = os.getcwd
        self._getcwdu = os._getcwdu = os.getcwdu
        os.getcwd = lambda : str(self._abspath)
        os.getcwdu = lambda : unicode(self._abspath)
        os.chdir(self._abspath)
        return self

    def __exit__(self, *_):
        os = self.getOS()
        os = sys.modules['os']
        if self.printOnExit:
            print('[FrozenPath] exiting...')
            print(os.getcwdu())
        os.chdir(self._old_dir)
        if self.debug:
            print('[FrozenPath] returning to _old_dir:\n%s'%self._old_dir)
        os.getcwd = self._getcwd
        os.getcwdu = self._getcwdu