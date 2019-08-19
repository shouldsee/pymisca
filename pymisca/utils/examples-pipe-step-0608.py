
import pymisca.ext as pyext
DB_SCRIPT={
    'MODULE_NAME':__name__,
    
}
def THIS_FUNC(DB_WORKER):
    with pyext.getPathStack([DB_WORKER['INPUTDIR']],force=1):
        pyext.shellexec('echo hello world > LOG')