# work/users/fg368/mapDB/src/0722.py
# %%python2
# %%writefile /work/users/fg368/mapDB/src/0722.py
# pyext.self__install
import pymisca.ext as pyext
import pymisca.atto_jobs
import os
# pyext.self__install()
# pyext.shellexec("mkdir -m 777 -p /tmp/test")
with pyext.getPathStack(["/tmp","test",__file__],force=1):
    with open('TEST','w') as f:
        f.write('test\n')
        pass
    pymisca.atto_jobs.Cleaner({'OUTDIR': '/tmp/test/%s'%__file__})

    assert not os.path.isfile("TEST")

# import shutil
# ??shutil.rmtree
# !mkdir -p testd