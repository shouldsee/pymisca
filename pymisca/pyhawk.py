'''
Examples:
    echo "NC.100,2\n3,1" | python2 pyhawk.py --FS "," --column.mapper.0.3=3000
    echo "NC.100,2\n3,1" | python2 pyhawk.py --ARG_SEP : --FS "," --column:mapper:0:NC.100=XC.100
    echo "NC.100,2\n3,1" | python2 pyhawk.py --ARG_SEP : --FS "," --lambda:0:line="line.insert(0,line[0])"
    
'''
import sys
import argparse
# sys.argv
# parser=argparse.ArgumentParser()
parser= argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--FS',default=None,)
parser.add_argument('--OFS',default='\t', )
parser.add_argument('--ORS',default='\n', )
parser.add_argument('--ARG_SEP',default='.',)
# parser.add_argument('--ifname')
# parser.add_argument('--lines',default=False,action='store_true')
# parser.add_argument('--show',default=False,action='store_true')
# parser.add_argument('--basename',default=False,action='store_true')
# parser.add_argument('--absolute',default=False,action='store_true')

import json
import collections
def tree(): return collections.defaultdict(tree)
import re

class LambdaWithSource(object):
    def __init__(self,code):
        if isinstance(code, types.LambdaType):
            inspect.getsource()
        code = code.strip()
        assert code.startswith("lambda"),(code,)
        self.code = code
        self.func = eval(code)
    def __repr__(self):
        return '<WithSource>%s'%self.func.__repr__()
    def __call__(self,*a,**kw):
        return self.func(*a,**kw)

import types
import inspect
class _Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, LambdaWithSource):
            res = o.__getattribute__("code")
        else:
            assert 0,(type(o),o,)
        return res

# # class 
def main(argTree, stdinFunc = None, stdoutFunc=None, stderrFunc=None):
    if stdinFunc is None:
        stdinFunc = lambda :next(sys.stdin,None)
    if stdoutFunc is None:
        stdoutFunc = sys.stdout.write
    if stderrFunc is None:
        stderrFunc = sys.stderr.write
        
    #################    
    argTree.setdefault('ORS','\n')
    argTree.setdefault('OFS','\t')
    argTree.setdefault('FS',None)
    argTree.setdefault('ARG_SEP','.')
    
    assert argTree['ORS'] == '\n',('ORS not implemented',argTree['ORS'])
    ### set the default tree
    argTree['column']  = column = argTree.get('column',{})
    column.setdefault('mapper',{})
    argTree.setdefault('lambda',{})
    argTree.setdefault('lambdaFuncs',[])

    ### casting types
    mapper = argTree['column']['mapper']
    for k,v in mapper.items():
        if k.isdigit():
            mapper[int(k)] = mapper.pop(k)
        else:
            assert 0,"named columns is not supported"
            pass
    
    d = argTree['lambda']
    argTree['lambdaFuncs'] += [lambda x:x] * (len(d) - len(argTree['lambdaFuncs']) )
    out = argTree['lambdaFuncs']
    for k in d:
        if k.isdigit():
            v = d[k]
#             .pop(k)
            assert len(v)==1,(k,v,)
            arg, code = v.items()[0]
            v = 'lambda {arg}:{code}'.format(arg=arg,code=code)
            out[int(k)] = v
        else:
            assert 0,"named columns is not supported"

    #############
    d = argTree['lambdaFuncs']
    for k,v in enumerate(d):
        d[k] = LambdaWithSource(v)


#     sys.stderr.write( json.dumps(argTree,indent=4,default=repr)+'\n')
    stderrFunc( json.dumps(argTree,indent=4,cls=_Encoder)+'\n')
#     _p = re.compile(r' +')
    while True:
        ###### ORS is assumed to be \n
        line = stdinFunc()
        if line is None:
            break
#         else:            
#     for line in sys.stdin:
#         line = _p.sub('',line)
        line = line.strip()
        sp = line.split(argTree['FS'])
        for k,v in argTree['column']['mapper'].items():
            sp[k] = v.get(sp[k],sp[k])
            
        for f in argTree['lambdaFuncs']:
            f(sp)
        stdoutFunc(argTree['OFS'].join(sp)+argTree['ORS']) 
    return 

def parser__getArgTree(parser,args=None):
#     args = '--test=test --hi=hi --hibb=hi'.split()
    known, unknown_args = parser.parse_known_args(args=args)
    for i in unknown_args:
        if i.startswith('--'):
            parser.add_argument(i.split('=',1)[0],type=unicode)
    args = parser.parse_args(args=args)
    argD  = vars(args).copy()
    argD['__file__'] = __file__
    argTree = d = tree()
    for k,v in argD.items():
#         print argD['sep']
        sp =  k.split(argD['ARG_SEP'])
#         assert len(sp)==2
        _d = d
        I = len(sp)
        for i, _k in enumerate(sp):
            if i + 1 != len(sp):
                _d = _d[_k]
            else:
                _d[_k] = v    
    return argTree

if __name__ == "__main__":
    
    args = None
    argTree = parser__getArgTree(parser,None)
    main(argTree)

#         sys.stdout.write(argTree['OFS'].join(sp)+argTree['ORS'] )


    '''
    echo "1,2\n3,1" | python2 /data/repos/pymisca/pymisca/pyhawk.py --FS "," --column.mapper.1.1=2
    '''