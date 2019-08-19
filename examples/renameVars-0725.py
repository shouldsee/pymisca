# import pymisca.header
import pymisca.header as pyext
# reload(pyext)
import inspect

'''
```
code(argcount, nlocals, stacksize, flags, codestring, constants, names,
      varnames, filename, name, firstlineno, lnotab[, freevars[, cellvars]])
      
function(code, globals[, name[, argdefs[, closure]]])      
```
'''

@pyext.renameVars(['a',])
def f1(*xyz):
    pass
@pyext.renameVars({'xyz':'ab/c'})
def f2(*xyz):
    pass

for x in map(inspect.getargspec, [f1,f2]):
    print (x,)
# inspect.getargspec(abc)