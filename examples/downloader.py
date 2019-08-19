
import pymisca.ext as pyext
import pymisca.module_wrapper
_t = pymisca.module_wrapper.type__resolve
import os
_DIR = os.path.dirname(os.path.realpath(__file__))

# def MAIN()

class MAIN(_t('AttoJobResult')):
    def __init__(self,METHOD,URL,**kw):
        d = {
            'INPUTDIR': _DIR,
            'RUN_MODULE': 'wraptool.request-url-0628',
            'RUN_PARAMS':{'METHOD':METHOD,
                         'URL':URL},
        }
        super(MAIN,self).__init__(**d)
        
# class MAIN(object):
#     def __call__(self, METHOD, URL,**kw):
#         d = {
#             'INPUTDIR': _DIR,
#             'RUN_PARAMS':{'METHOD':METHOD,
#                          'URL':URL},
#         }
#         res = _t('AttoJobResult')(**d)
#     #         pymisca.module_wrapper.worker__stepWithModule()
#         return res
    
# pass
# def __call(self,METHOD,URL,**kw):
    
    
    
# def MAIN(self,METHOD,URL,**kw):
#     d = {
#         'INPUTDIR': _DIR,
#         'RUN_PARAMS':{'METHOD':METHOD,
#                      'URL':URL},
#     }
#     res = _t('AttoJobResult')(**d)
# #         pymisca.module_wrapper.worker__stepWithModule()
#     return res
# #     def call_tuples()