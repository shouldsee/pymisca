
import pymisca.ext as pyext
DB_SCRIPT={
    'MODULE_NAME':__name__,
    
}
def THIS_FUNC(DB_WORKER):
    with pyext.getPathStack([DB_WORKER['OUTDIR']],force=1):
        pyext.shellexec('echo hello world > LOG')
        pyext.printlines(['val'] + list(range(100)),'test.csv')
#         pyext.shellexec('echo > LOG')        