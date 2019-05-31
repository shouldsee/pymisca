#!/usr/bin/env python2
import pymisca.shell as pysh
import itertools
reload(pysh)

p = pysh.ShellPipe()
# p.chain('convert2bed -iwig')
# p.chain('bs ')
p.chain('tee test.out')
p.chain("awk '$1 > 5' ")

it = list(range(10))
it = map(str,it)
# it = ['%s\n'%x]
p.readIter(it, lineSep='\n')
res = p.checkResult(cmd=None)
print (res)