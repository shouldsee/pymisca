class CopyTo(object):
    def __init__(self, OUTDIR = None, INPUTDIR=None, basename = None, force=0,
                 DRY=0,
                 errorFunc=shutil.copy2):
        if OUTDIR is None:
            OUTDIR = os.getcwd()
        self.dest = OUTDIR
        self.src = INPUTDIR
        self.basename = basename
        self.force = force
        self.errorFunc = errorFunc
        self.DRY = DRY
#         self.allowCopy = 1
        
    def call_tuple(self, (FNAME, basename)):
#         _FNAME = os.path.realpath(FNAME)
#         FNAME = 
        _type = type(FNAME)
        if not os.path.isabs(FNAME):
            srcDir = self.src or os.getcwd()
            _src = os.path.join( srcDir, FNAME)
        else:
            _src = FNAME
        if not os.path.exists(_src):
            assert 0,("path does not exist", _src)
            
        _src;
        _basename = basename or self.basename or os.path.basename( _src)
        _dest = os.path.join( self.dest, _basename)
        _dest = _type(_dest)
        if not self.DRY:
            if os.path.isdir( _src):            
                pymisca.shell.dir__link( _src, _dest, force=self.force)

    #             assert 0,"can only download file, not diretory"
            else:
                if not pymisca.shell.file__notEmpty( _src):
                    assert 0, ("FILE is empty", _src)
                else:

                    pymisca.shell.real__dir(fname=_dest)
                    if os.path.abspath(_src) == os.path.abspath(_dest):
                        pass
                    else:
                        if os.path.isfile(_dest):
                            if self.force:
                                os.remove(_dest)
                            else:
                                assert 0,(self.force, "Specify force=1 to overwrite",_src,_dest)
                        try:
                            os.link( _src, _dest)
                        except OSError as e:
                            if e.errno == errno.EXDEV:
                                self.errorFunc(_src,_dest)
                            else:
                                raise e
        else:
            pass
        return _dest
    
    def __call__(self, FNAME, basename = None):
        return self.call_tuple((FNAME,basename))
