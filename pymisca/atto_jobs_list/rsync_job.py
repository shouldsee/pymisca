# -*- coding: utf-8 -*-
import pymisca.ext as pyext
# import pymisca.module_wrapper 

import pymisca.atto_jobs
from pymisca.atto_job import AttoJob
_getPathStack = pymisca.tree.getAttoDirectory



class rsync_job(AttoJob):
    PARAMS_TRACED  = [
    ('OPT_MAIN',('unicode','-avruPh')),
    ('OPTS_LIST',('list:AttoPath',[
        '--size-only','--ignore-times','--exclude=".*"','--stats',
    ]),),
    ('SOURCE',('AttoHostDirectory',None)),
    ('DEST',('AttoHostDirectory',None)),
    ('OUTDIR',('AttoPath','')),

    ('PASSWORD_FILE',('AttoPath','')),
    ('FILES_LIST',('list:AttoPath',[])),
    ('INCLUDE_LIST',('list:AttoPath',[])),
    ('EXCLUDE_LIST',('list:AttoPath',[])),
    ('FORCE',('int',0)),
    ]
    def _run(self):
        RUN_PARAMS =kw = self._data
#         RUN_PARAMS= DB_WORKER['RUN_PARAMS']
        OPT_MAIN = RUN_PARAMS['OPT_MAIN']
        OPTS_LIST = RUN_PARAMS['OPTS_LIST']
        SOURCE = kw['SOURCE']
        DEST = RUN_PARAMS['DEST']
        PASSWORD_FILE  = RUN_PARAMS['PASSWORD_FILE']
        FILES_LIST = RUN_PARAMS['FILES_LIST']
        INCLUDE_LIST = kw['INCLUDE_LIST']
        EXCLUDE_LIST = kw['EXCLUDE_LIST']
        kw['OUTDIR'] = OUTDIR = kw['OUTDIR'] or type(kw['OUTDIR'])(DEST)
#         FILES_EXCLUDE_LIST = RUN_PARAMS['FILES_EXCLUDE_LIST']
#         ERROR = DB_WORKER.get('ERROR','raise')
        FORCE = RUN_PARAMS['FORCE']
        ERROR = 'raise'
        
#         if not FORCE:
#             with pyext.getAttoDirectory([OUTDIR],force=1) as stack:
#                 if (stack.d / 'LOG').isfile():
#                     return 'SKIP'

#         if FILES_LIST:
#             pass
#     #         if SOURCE.is_local():    
#         else:
         

        if not SOURCE.is_remote():
            with pyext.getAttoDirectory([SOURCE],force=0) as stack:
                out = []
                for F in FILES_LIST:
                    res = stack.d.glob(F)
                    assert len(res),(F,stack.d )
                    out += res
                FILES_LIST = [x.relpath(SOURCE) for x in out]
#                     assert F.exists(),(F,)
            SOURCE = stack.d.realpath()
        else:
            print('[WARN]skipping remote file list check')
          
        if not SOURCE.endswith('/'):
            SOURCE = type(SOURCE)(SOURCE+'/')

        #                 F.isfile() or F.islink(), (F,)
        with pyext.getAttoDirectory([OUTDIR],force=1) as stack:
            if FILES_LIST:
                pyext.printlines(FILES_LIST,'FILES.txt')
                OPTS_LIST += ['--files-from=FILES.txt']
            if INCLUDE_LIST:
                pyext.printlines( INCLUDE_LIST,'INCLUDE.txt')
                OPTS_LIST += ['--include-from=INCLUDE.txt']
            if EXCLUDE_LIST:
                pyext.printlines( EXCLUDE_LIST,'EXCLUDE.txt')
                OPTS_LIST += ['--exclude-from=EXCLUDE.txt']
#             print pyext.shellexec('cat EXCLUDE.txt')

        if '--log-file' not in OPTS_LIST:
            OPTS_LIST += ['--log-file','LOG_RSYNC']
        CMD = pyext.f("rsync {OPT_MAIN} {' '.join(OPTS_LIST)} {SOURCE} {DEST}")

        if PASSWORD_FILE:
            assert pyext.file__notEmpty(PASSWORD_FILE),(PASSWORD_FILE,)

            CMD = pyext.f('sshpass -f {PASSWORD_FILE} {CMD} >LOG') 

#         DB_WORKER['RUNTIME']['TIME_DICT'] = TIME_DICT = pyext._DICT_CLASS()
#         _exec = pyext.func__timer( TIME_DICT, key=DB_SCRIPT['MODULE_NAME'])(pyext.shellexec,)
#         _exec = pyext.functools.partial(_exec,error=ERROR)
        with pyext.getAttoDirectory([OUTDIR],force=1):
#             res = _exec(CMD)
            res = pyext.shellexec(CMD)
        return res