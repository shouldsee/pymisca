import pymisca.module_wrapper
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  ### hotfix for runtime import

import pymisca.ext as pyext
pyext.shellexec('mkdir -p example-output/ ')
DB_WORKER={}
DB_JOBS = [
    {
        'RUN_MODULE':'examples.pipe-step1-0608',
        'INPUTDIR':'!{PWD}/example-output',
    },
    {
        'RUN_MODULE':'examples.pipe-step2-0608',
        'INPUTDIR':'!{LAST_DIR}',
    }
]
    

for DB_JOB in DB_JOBS:
    DB_WORKER.update(DB_JOB)
    DB_WORKER = pymisca.module_wrapper.worker__stepWithModule(DB_WORKER)
    try:
        pyext.ipd.display(DB_WORKER)
    except:
        print(repr(DB_WORKER))
    print()
print(DB_WORKER['LAST_DIR'],)
        