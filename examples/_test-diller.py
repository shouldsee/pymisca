#!/usr/bin/env python

import scipy.stats
import numpy as np
import cPickle as pickle
import dill
import json

# Must specify a `bw_method` for pickle to fail
KDE = scipy.stats.gaussian_kde(np.array([0,1,2]),bw_method=2)

try:
    pkl_str = pickle.dumps(KDE)
except:
    print('Pickle Failded')

try:
    dill_str = dill.dumps(KDE)
except:
    print('dill failed')
dill_str = dill_str.decode('unicode_escape')
# Now try to JSON encode the string
# try:
s = json.dumps([dill_str],encoding='unicode_escape')
rs= json.loads(s)[0]
print repr(rs[:50])
print repr(dill_str[:50])
print rs == dill_str

# except:
# print('JSON encode failed')