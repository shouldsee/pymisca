import pymisca.atto_string as pyatto
import glob
import json
import pymisca.ext as pyext

from pymisca.atto_string import AttoString
import sys
jsonRepoPath = sys.argv[1]
# AttoString = 
def _unidiff_output(expected, actual):
    """
    Helper function. Returns a string containing the unified diff of two multiline strings.
    """

    import difflib
    expected=expected.splitlines(1)
    actual=actual.splitlines(1)

    diff=difflib.unified_diff(expected, actual)

    return ''.join(diff)


import difflib


def tree__sanitise__char(tree,char=' /',fill='-'):
    _this_func = tree__sanitise__char
    if isinstance(tree,basestring):
        for c in char:
            tree = tree.replace(c,fill)
    elif isinstance(tree,dict):
        for k in tree:
            tree[k] = _this_func(tree[k],char,fill)
    elif isinstance(tree,list):
        for i in range(len(tree)):
            tree[i] = _this_func(tree[i],char,fill)
    else:
        pass
#    assert 0
    return tree
# for v in 
# AttoString.fromAttoString(pyext.AttoString.fromContainer(v)

# pyext.AttoString.fromAttoString(pyext.AttoString.fromContainer([123]).toAttoString())


for fname in glob.glob("{jsonRepoPath}/*.json".format(**locals())):
#%jsonRepoPath):
#    pyext.shel
    print(fname)
    buf = open(fname,'r').read().strip()
    v= v0 = pyext.readData(fname)

    try:
        s = pyatto.AttoString.fromContainer(v)
    except Exception as e:
        if 'invalid char' in str(e):
            v0 = v = tree__sanitise__char(v)
            s = pyatto.AttoString.fromContainer(v)
            continue
        else:
            raise e
    v1 = v = s.toContainer()
    msg = pyext.ppJson((s,v1,v0))
    buf1 = pyext.ppJson(v1)
    buf0 = pyext.ppJson(v0)

    assert v1==v0, _unidiff_output(buf0,buf1)
    s1 = s.toAttoString()
    s2 = AttoString.fromAttoString(s1)
    assert s2==s,pyext.ppJson((s,s1,s2))
#    diff = [x for x,y in zip(buf0,buf1) if x!=y]
#    diff = ''.join(diff)
#    assert not diff,diff

#if 1:
#    msg = pyext.ppJson((s,v1,v))
#    assert len(v0)  == len(v1),msg
#    if isinstance(v0,list):#
#        for _v0,_v1 in zip(v0,v1):
#            assert _v0 == _v1,pyext.ppJson((fname,s,_v1,_v0))
#   else:
#        assert isinstance(v1,dict)
#    assert list(v1) == list(v0)
#    for _v0,_v1 in zip(v0.items(),v1.items()):
#            assert _v0 == _v1,pyext.ppJson((fname,s,_v1,_v0))
       
#    for i in range(len(v0)):    
#    assert pyext.ppJson(v1)==pyext.ppJson(v0),msg
#    assert v1 == v0,pyext.ppJson((s,v1,v))
#    assert v1 == v0,pyext.ppJson((s,v1,v))
