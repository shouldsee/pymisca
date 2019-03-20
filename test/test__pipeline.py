#!/usr/bin/env python
import pymisca.header as pyheader
import pymisca.ext as pyext

### make sure $BASE is set within environment, or default to current direct$
pyheader.base__check() 

suc,res = pyext.job__baseScript('test/pipeline-src/init__seq.sh',silent=1)
assert suc,res

suc,res = pyext.job__baseScript('test/pipeline-src/histogram.py',silent=1)
assert suc,res

### we can also output to plots/ instead of results/ 
suc,res = pyext.job__baseScript('test/pipeline-src/histogram.py',silent=1,
                            prefix='plots')
assert suc,res

### job__baseScript calls job__script() internally, where all keywords are specified
### Documentation welcomed

print ('[DONE]')