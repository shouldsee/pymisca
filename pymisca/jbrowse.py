import sys,os
import pandas as pd
import numpy as np

import json
def ppJson(d):
    '''
    Pretty print a dictionary
    '''
    s = json.dumps(d,indent=4, sort_keys=True)
    return s
def bwMeta2json(urlTemplate, label,
                ofname = None, 
                key = None,
                **kwargs):
    if ofname is not None:
        assert isinstance(ofname, basestring)
        ### calculate relative path if applicable
        urlTemplate = np.vectorize(os.path.relpath)(
            urlTemplate,
            os.path.dirname(ofname))
        
    res  = {
         "storeClass" : "JBrowse/Store/SeqFeature/BigWig",
         "urlTemplate" : urlTemplate,
         "autoscale" : "local",
         "label" : label,
        "key" : key,
         "type" : "JBrowse/View/Track/Wiggle/XYPlot"        
    }
    res  = dict(tracks = pd.DataFrame(res
                                     ).to_dict(orient='record'),)
    if ofname is not None:
        with open(ofname,'w') as f:
            ### [PERF]
            json.dump(res,f,indent=4,sort_keys=False)
#             f.write(pyutil.ppJson(res))
        res = ofname
    return ofname
