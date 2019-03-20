#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pymisca.header as pyheader
pyheader.mpl__setBackend('agg')

import pymisca.ext as pyext
import matplotlib.pyplot as plt
import collections
import pandas as pd

fname = pyext.base__file('results/init__seq/seq.fa')
with open(fname,) as f:
    buf = '\n'.join([x.strip() for x in f if not x.startswith(">")])
ct = collections.Counter((buf))
ct = pd.DataFrame(ct.items()).set_index(0)
ct.hist()
fig = plt.gcf()
fig.savefig('histogram.png')