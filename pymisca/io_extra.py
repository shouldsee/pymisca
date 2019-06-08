import sys
import itertools,collections



import codecs
import StringIO

def unicodeIO(buf=None, handle=None, encoding='utf8',*args,**kwargs):
    if handle is None:
        handle = StringIO.StringIO()
    codecinfo = codecs.lookup(encoding)
    wrapper = codecs.StreamReaderWriter(
        handle, 
        codecinfo.streamreader, 
        codecinfo.streamwriter)
    if buf:
        wrapper.write(buf)
        wrapper.seek(0)
    return wrapper










def file__getWriter(f,newline=None,method=None):
    if method is None:
        method = 'writelines'
    if newline is None:
        newline = '\n'
        
    def _writer(s):
        f.write(s)
        if newline !=-1:
            f.write(newline) 
            
    def _func(ele):
        if method =='writelines':
            map(_writer, ele)
        elif method=='write':
            _writer(ele)
            
    return _func
    
def it__toFile(it,OFNAME,
               mode='w', buffering=-1,
               method = None, newline=None,
               **kw):
    '''
    Write an iterator to a file specified by path OFNAME
    '''
    if newline is None:
        newline = '\n'
    with open(OFNAME,mode=mode, buffering = buffering, **kw) as f:
        _writer = file__getWriter(f, newline=newline,method=method)
        map(_writer,it)
        
                
    return OFNAME

def itGrouped__toFiles(itGrouped, 
                       mode='w', buffering=-1, 
                       method = None, newline=None,
                       callbefore = None, callbefore__OFNAME=None,
                       **kw):
    '''
    Write an (OFNAME,it)-formatted iterator to multiple files specified by path OFNAME
    '''    
    dFiles = collections.OrderedDict()
    
    for OFNAME,it in itGrouped:
        if OFNAME not in dFiles:
            
            if callbefore__OFNAME is not None:
                ACTUAL_NAME = callbefore__OFNAME(OFNAME)
            else:
                ACTUAL_NAME = OFNAME
                
            f = open( ACTUAL_NAME,  mode=mode, buffering=buffering,**kw)
            writer = file__getWriter(f, newline=newline,method=method)
            
            dFiles[OFNAME] = {'handle':f,
                              'writer': writer,
                              'ACTUAL_NAME':ACTUAL_NAME}
        if callbefore is not None:
            it = (callbefore(x) for x in it )
        map(dFiles[OFNAME]['writer'],it)
        
    for OFNAME,d in dFiles.items():
        d['handle'].close()
    return {k:x['ACTUAL_NAME'] for k,x in dFiles.items()}
        
        
# import operator