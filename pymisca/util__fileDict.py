#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import argparse
# import pymisca.ext as pyext
import json,os,sys
import functools,collections

import pymisca.tree as pyTree


json.load_ordered= functools.partial(json.load,
                                    object_pairs_hook=collections.OrderedDict,)
def dictFilter(oldd,keys):
    d =collections.OrderedDict( 
        ((k,v) for k,v in oldd.iteritems() if k in keys) 
    )
    return d

def dictGetList(oldd,keys):
    d =collections.OrderedDict( ((k,oldd[k]) for k in keys) )
    return d

def jsonFile2dict(fname,force= 0):
    if os.path.exists(fname):
        if os.stat(fname).st_size != 0:
            with open(fname,'r') as f:
                res = json.load_ordered(f,)
                return res
    if force:
        return collections.OrderedDict()
    raise Exception('file does not exists or empty:%s' % fname)
#     return collections.OrderedDict([])

def fileDict__save(fname=None,d=None,keys=None,indent=4):
    assert d is not None,'set d=locals() if unsure'
    d = collections.OrderedDict(d)
    if keys is not None:
        d  = dictFilter(d,keys)
        
#     if keys is not None:
#         d  = pyext.dictFilter(d,keys)
    
    if isinstance(fname,basestring):
        res = jsonFile2dict(fname,force=1)
        res.update(d)
        d = res
                
    res = json.dumps(d,indent=indent,default=lambda x:x.toJSON())
    res = res + type(res)('\n')    
    
    if isinstance(fname,basestring):
        with open(fname,'w') as f:
            f.write(res)
        return fname
    elif fname is None:
        f = sys.stdout
        f.write(res)
    else:
        f = fname
        f.write(res)
        return f
def mapDict(func,d,mapkey=False):
    lst = []
    for k,v in d.items():
        if mapkey:
            k = func(k)
        res = func(v)
        lst += [(k,res)]
    return collections.OrderedDict(lst)


def dict__update(old,new,copy=False):
    if copy:
        old = old.copy()
    old.update(new)
    return old

parser=argparse.ArgumentParser()
parser.add_argument('--ofname',default='FILE.json',)
parser.add_argument('--ifname')
parser.add_argument('--lines',default=False,action='store_true')
parser.add_argument('--show',default=False,action='store_true')
parser.add_argument('--basename',default=False,action='store_true')
parser.add_argument('--absolute',default=False,action='store_true')


        
    



def fileDict__main(
        ofname = 'FILE.json', 
         ifname=None,
         lines=False,
         show=False,
         basename=False,
         absolute=False,
         showSave=False,
         force = 1,
         treeSep = '.',
         argD = {},
         **kwargs):
    
    #### We can pass through kwargs directly
    for key in ['ofname','ifname','lines','show','basename','absolute','treeSep']:
        if key in argD.keys():
            argD.pop(key)
            
    
    if ifname is not None:
        res = jsonFile2dict(ifname,force=force)
        argD = dict__update(res,argD)
#         if os.path.exists(args.ofname):
#             d = fileDict__load(fname=args.ofname)
#             argD.update(vars(d))
    
    if ofname is not None:
        d = jsonFile2dict(ofname,force=force)
        DIR = os.path.dirname(os.path.abspath(
            ofname))
    else:
        d = collections.OrderedDict()
        DIR=''
        
    if basename:
        d = mapDict(os.path.basename,d)
    elif absolute:
        d = mapDict(lambda line: os.path.join(DIR,line),d)
    else:
        pass
    
    
    
    if show or showSave:
            
        if lines:
            for line in d.values():
                print(line)
        else:
            print(json.dumps(d,indent=4))
        save = showSave
    else:
        save = True
    



#     output = collections.OrderedDict()
    

#     argD = dict__flat2tree(argD)
    if save:
        d.update(argD)
#        d = TreeDict.from_flatDict(d)                
        d = pyTree.TreeDict.from_flatPathDict(d,sep=treeSep)                
        fileDict__save(d=d, 
                       fname=ofname)
            
#             with open(ofname,'r') as f:
#                 print(f.read())
#             print(os.system('cat %s'%ofname))
#     print ('[DONE]')
    return d
main = fileDict__main
#             print(d)    
if __name__ =='__main__':
    args = None
#     args = '--test=test --hi=hi --hibb=hi'.split()
    
    known, unknown_args = parser.parse_known_args(args=args)
    for i in unknown_args:
        if i.startswith('--'):
            parser.add_argument(i.split('=',1)[0])
    args = parser.parse_args(args=args)
    
    ### for saving
    argD  = vars(args).copy()

    main( argD=argD,
         **vars(args))
