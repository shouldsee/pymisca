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
        os._getcwd = os.getcwd
        os._getcwdu = os.getcwdu
        os.getcwd = lambda : str(self._abspath)
        os.getcwdu = lambda : unicode(self._abspath)
        os.chdir(self._abspath)
        return self

    def __exit__(self, *_):
        os = self.getOS()
        os = sys.modules['os']
        os.chdir(self._old_dir)
        os.getcwd = os._getcwd
        os.getcwdu = os._getcwdu