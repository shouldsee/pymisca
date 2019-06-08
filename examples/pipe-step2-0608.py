
import pymisca.ext as pyext
DB_SCRIPT={
    'MODULE_NAME':__name__,
    
}
def THIS_FUNC(DB_WORKER):
    with pyext.getPathStack([DB_WORKER['INPUTDIR']]):
        df = pyext.readData('test.csv',header=0,guess_index=0)
        
    with pyext.getPathStack([DB_WORKER['OUTDIR']],force=1):
        df['gp'] = df.iloc[0]%3
        df.to_csv('output.csv')