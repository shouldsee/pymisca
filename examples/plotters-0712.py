import pymisca.ext as pyext
pyext.self__install()
from pymisca.plotters import plotters
import pymisca.module_wrapper
# d = dict(DIR='/tmp')
try:
    import fisher
except Exception as e:
    '! pip2 install git+https://github.com/brentp/fishers_exact_test --user'
    raise e
with pyext.getPathStack(['/tmp/t1'],force=1):
    

    DB_JOB = {'RUN_MODULE':'wraptool.request-url-0628',
              'INPUTDIR':'!{PWD}',
             'RUN_PARAMS':{'METHOD':'get',
                          'URL':'http://google.co.uk'}}
    res = pymisca.module_wrapper.worker__stepWithModule(DB_JOB)

    d = {
      "FUNCTION": "!{plotters.venn_diagram}",
      "OFNAME": "venn-diagram-2.png",
        "index1":"!{range(3,12)}",
        "index2":"!{range(1,5)}",

    #   "index1": "!{pyext.readData('/home/feng/static/figures/1126__PIF7__Venn__pif7Resp-AND-pif7SimpleBound/Venn-index.csv',)['ind1']}",
    #   "index2": "!{pyext.readData('/home/feng/static/figures/1126__PIF7__Venn__pif7Resp-AND-pif7SimpleBound/Venn-index.csv',)['ind2']}",
      "axis": {
        "xlabel": "interval[3,12)",
        "ylabel": "interval[1,5)",
        "title": "Fisher exact test: p={pval}",
      }
    }
    range = range
    plotters.job__process(d);
# pymisca.plotters

