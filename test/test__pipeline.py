#!/usr/bin/env python
import pymisca.header as pyheader
pyheader.base__check()

import pymisca.ext as pyext

suc,res = pyext.job__baseScript('pipeline-src/init__seq.sh',silent=1)
assert suc,res

suc,res = pyext.job__baseScript('pipeline-src/histogram.py',silent=1)
assert suc,res

print ('[DONE]')