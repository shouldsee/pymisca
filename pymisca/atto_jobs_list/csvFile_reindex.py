import pymisca.ext as pyext
from pymisca.atto_job import AttoJob
import itertools
class csvFile_reindex(AttoJob):
    PARAMS_TRACED = [
        ('INPUT_FILE',('AttoPath','')),
        ('OUTPUT_FILE',('AttoPath','')),

        ('DB_FILE',('AttoPath','')),
        ('FROM',('unicode','')),
        ('TO',('list:unicode',[])),
        ('KWARGS_TO_CSV',('dict:object',{'index':0,})),
    ]
    def _run(self):
        kw = self._data
        assert kw['DB_FILE']
        assert kw['INPUT_FILE']
        assert kw['OUTPUT_FILE']
        kw['DB_FILE'] = DB_FILE = kw['DB_FILE'].realpath()
        kw['INPUT_FILE'] = INPUT_FILE = kw['INPUT_FILE'].realpath()
        kw['OUTPUT_FILE'] = OUTPUT_FILE = kw['OUTPUT_FILE'].realpath()
        TO = kw['TO']
        FROM = kw['FROM']
        assert TO
        assert FROM
        
        it = iter(open(INPUT_FILE,'r'))
        it = (x.strip() for x in it)
        it = (x for  x in it if x and not x.startswith("#"))
        _INPUT_ITER = it


        df = pyext.readData( DB_FILE, guess_index=0)
        _COLS = df.columns = df.columns.str.upper()
        _COLS = list(_COLS)

        assert len(set(_COLS))==len(_COLS),(_COLS,)
        assert FROM in _COLS,(FROM,_COLS)

        dup = df.loc[df[ FROM ].duplicated(0),FROM]
        assert len(dup)==0,(dup[:5],)
        
        _INPUT_ITER,it = itertools.tee(_INPUT_ITER)
        
        _FROMS = list( df[FROM])
        for ele in it:
            assert ele in _FROMS,(ele, DB_FILE, FROM, df[FROM][:5])
#         for ele in it:
#             assert ele in df[FROM],(ele, DB_FILE, FROM, df[FROM][:5])
        res = df.set_index(FROM)[TO].reindex(_INPUT_ITER)
        res.to_csv(OUTPUT_FILE,**kw['KWARGS_TO_CSV'])
        
if __name__ == '__main__':
#     import pymisca.ext as pyext
    with pyext.getPathStack(['csvFile_reindex.test-data']):
        node = csvFile_leftJoin(dict(
            DB_FILE = 'GENE_META.csv',
            FROM = 'GENE_GENE_NAME',
            TO = ['UNIPROT_ID'],
            INPUT_FILE = 'test.input.it',
            OUTPUT_FILE = 'OUTPUT.it',
            KWARGS_TO_CSV = {'index':0,"header":0,}
        ))
        assert filecmp.cmp('OUTPUT.it','OUTPUT.it.expect')        