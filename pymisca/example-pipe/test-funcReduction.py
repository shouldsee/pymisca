import pymisca.ext as pyext
np = pyext.np
def getData(d,**kw):
    d = dict(enumerate(range(10)))
#     np.save('d.npy',[d])
    pyext.printlines(d.items(),'out.txt')
    return d
def expData(d,**kw):
    d = {k:np.exp(v) for k,v in d.items()}
    pyext.printlines(d.items(),'out.txt')
    return d

pyext.funcs__reduceWithPath(['getData','expData'],outDir = pyext.getBname(pyext.runtime__file()))
print ('DONE')