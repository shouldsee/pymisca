import sys
import lazydict
# from lazydict import LazyDictionary
import pymisca.header
import pymisca.header as pyext
import pymisca.ext as pyext
Stepper = template =  stepper = lazydict.LazyDictionary(tb_limit=10)
stepper['input/list'] = lambda x:None
stepper['callback/main'] = pymisca.header.PlainFunction(lambda x: x)
stepper['callback/context'] = pyext.Suppress(1,1)
stepper['NCORE'] = 1
# stepper['suppress/stdout'] = 1
# stepper['suppress/stderr'] = 1

class NullContextManager(object):
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource
    def __enter__(self):
        return self.dummy_resource
    def __exit__(self, *args):
        pass
stepper['_callback/context'] = lambda self,k,callback_context: NullContextManager() if callback_context is None else callback_context

@pyext.setItem(stepper, 'input/length')
def _func(self,key, input_list):
    try:
        return len(input_list)
    except Exception as e:
        print(e)
        return -1

@pyext.setItem(stepper,'callback/message')
@pyext.PlainFunction
def _func(i,L,x,y):
    msg = '%s\t/\t%s\tFinished\n' % (i+1,L)
    sys.stderr.write(msg)
    return 

@pyext.setItem(stepper,'output/list')
def _func(self,key, input_list, input_length, callback_main, callback_message, 
          _callback_context,
#           suppres_stdout,suppress_stderr
         ):
    
    out = []
    for f in [callback_main,callback_message]:
        assert getattr(f,'_plain',0),(f,)

    for i,x in enumerate(input_list):
        with _callback_context:
            y = callback_main(x)
        _ = callback_message(i, input_length,x,y)
        out += [y]
        
    return out

@pyext.setItem(stepper,'_helper')
@pyext.PlainFunction
def _loop_helper( (i,x) , input_length, _callback_context, callback_main,callback_message):
    with _callback_context:
        y = callback_main(x)
    _ = callback_message(i, input_length,x,y)
    return y

@pyext.setItem(stepper,'callback/init')
def _func(self,k):
    '''Placeholder for things before strating the loop
    '''
    pass

@pyext.setItem(stepper,'output/list')   
@pyext.setItem(stepper,'output/list/mp')   
def _func(self,key, 
          callback_init,
          input_list, input_length, callback_main, callback_message, 
          _callback_context,
          
          NCORE,
#           suppres_stdout,suppress_stderr
         ):

    for f in [callback_main,callback_message]:
        assert getattr(f,'_plain',0),(f,)
#     _looper = pyext.functools.partial(_loop_helper,
#                               input_length = input_length,
#                               _callback_context=_callback_context,
#                               callback_message=callback_message,
#                               callback_main = callback_main,
                              
                             
#                                      )
    def _looper((i,x)):
        with _callback_context:
            y = callback_main(x)
        _ = callback_message(i, input_length,x,y)
        return y
    out  = pyext.mp_map(_looper,enumerate(input_list),NCORE=NCORE)
    return out
#     out = []
#     for i,x in enumerate(input_list):
#         with _callback_context:
#             y = callback_main(x)
#         _ = callback_message(i, input_length,x,y)
#         out += [y]
        
#     return out