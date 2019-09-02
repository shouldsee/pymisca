from pymisca.atto_jobs_list.url_request import url_request
import zipfile
from pymisca.atto_job import AttoJob
from pymisca.ext import readData
import pandas as pd
import numpy as np
import pymisca.ext as pyext

def _null_numpy_recarray():
    a = np.rec.array((0,),dtype=[('_null',object)])
    return a


class httpJob_pathview(AttoJob):
    '''
    Ref: https://pathview.uncc.edu/example1
    '''
    PARAMS_TRACED = [
        ('OUTDIR',('AttoPath','')),
        ('REQUEST_DATA',('dict:object:object',{})),
        ('REQUEST_DATA_DEFAULT',('dict:object:object',
                                 {
                                    'limit_gene':'1',
                                    'species':None,
                                    'gene_id':None,
                                     'pathway_id':None,
                                    'low_gene':'#FF0000',
                                    'high_gene':'#00FF00',
                                    'node_sum':'mean',
                                    'suffix':'OUTPUT',
                                    'version':'1.0.2',
                                 }
                                )),
        ('GENE_DFRAME',('object',_null_numpy_recarray())),
        ('FORCE',('int',0)),
    ]


    def _run(self):
        kw = self._data
        kw['OUTDIR'] = OUTDIR = kw['OUTDIR'].realpath()
    #     kw['GENE_']
        GENE_DFRAME = pd.DataFrame.from_records(kw['GENE_DFRAME']).set_index('index')
        assert len(GENE_DFRAME)
        REQUEST_DATA = kw['REQUEST_DATA']
        DEFAULT_DATA = kw['REQUEST_DATA_DEFAULT']
        FORCE = kw['FORCE']

        OUTDIR = OUTDIR.realpath()
        pyext.real__dir(dirname = OUTDIR)

        GENE_FILE = OUTDIR/'IN.tsv'
        GENE_DFRAME.to_csv(GENE_FILE,sep='\t')

        DEFAULT_DATA.update(REQUEST_DATA,)
        REQUEST_DATA = DEFAULT_DATA


        node = url_request({
                    'URL':'https://pathview.uncc.edu/api/analysis',
                    'OFNAME':OUTDIR.rstrip('/') + '.json',
        #             'OFNAME':"./1.out",
                    'FORCE':FORCE,
                    'FILES':{'gene_data':[GENE_FILE, file(GENE_FILE,'rb')] },
                    'METHOD':'post',
                    'DATA':REQUEST_DATA,
                 }
        )

        URL = readData(node['OFNAME'],'json')['download link']
        node = url_request({
            'URL':URL,
            'OFNAME':node['OFNAME']+'.zip',
            'FORCE':FORCE,
        })
        with zipfile.ZipFile( node['OFNAME'], 'r') as zip_ref:
            zip_ref.extractall(OUTDIR)
if __name__ == '__main__':
    import path
    import pymisca.ext as pyext
    GENE_DATA = pyext.read__buffer('TO,Mock_No_0,Mock_No_6,VACV_No_6,VACV_No_12,Mock_AraC_6,VACV_AraC_6\nQ9NRD1,0.05339672749421531,0.02147663487399676,0.045514922082626086,-0.20058388044622788,0.07924081258853732,0.0009547834068541761\nQ9NRD1,0.05339672749421531,0.02147663487399676,0.045514922082626086,-0.20058388044622788,0.07924081258853732,0.0009547834068541761\n',ext='csv')
    httpJob_pathview(
        dict(OUTDIR = path.Path("./pathview-test"),
    REQUEST_DATA = {'species':'hsa',
           'gene_id':'UNIPROT',
           },
    GENE_DFRAME =GENE_DATA.to_records(),
    FORCE = 1,))