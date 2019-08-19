# %%python2
# %%writefile /work/users/fg368/mapDB/src/0722.py
# pyext.self__install
import pymisca.ext as pyext
# import pymisca.atto_jobs
# pyext.self__install()
d = {}
@pyext.setItem(d,'hat')
def test():
    return

@pyext.setAttr(test,'hat2')
def test2():
    return

print(d)
print(d['hat'].hat2)


# pyext.self__install('/data/repos/wraptool/')
# pyext.self__install('/data/repos/lazydict/')
# pymisca.atto_jobs.Cleaner({'OUTDIR': 'testd'})
# # import shutil
# ??shutil.rmtree