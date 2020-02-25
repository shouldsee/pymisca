from pymisca.shell import  file__notEmpty, dir__makeIfNeed, file__safeInode
import os
class CopyEvent(object):
    def __init__(self, src,dest,force=0,delay=0):
        return 
    
class LinkEvent(object):
    def __repr__(self):
        return "%r"%((self.src,self.dest,self.force),)
    
    def __init__(self,src,dest,force=0,delay=0):
        self.src=src
        self.dest=dest
        self.force=force
        
        _f = file__safeInode
        assert os.path.isfile( src )
        if _f(src)==_f(dest):
            pass
        else:
            if file__notEmpty(src):
                if file__notEmpty(dest):
                    if not force:
                        return
                    else:
                        os.remove(dest)
                dir__makeIfNeed(fname=dest)
                print(dest,)
                os.link(src,dest)
            else:
                print('[EMPTY.src]',src,)