CMD_TEMPLATE= '''
Rscript --slave --vanilla - <<EOF

df = read.table(
  "{{INPUT_FILE}}"
  ,stringsAsFactors = F,header = 1,row.names = 1)
df = as.matrix(df)

column__sanitise<-function(s){
    gsub('[^a-zA-Z_]','_',toupper(s))
}
suppressPackageStartupMessages({
    library(baySeq)
    library(DESeq2)

})


#LIB_SIZES = data.frame(row.names = colnames(df))

#### add vst output
res <- DESeq2::vst( 
  as.matrix(df)
 ,fitType = "parametric")
write.csv(res,"{{OFNAME}}.partial")
#head(res);
EOF
'''

from pymisca.atto_job import AttoJob
import shutil
import pymisca.ext as pyext
class csvFile_deseq2_vst(AttoJob):
    PARAMS_TRACED = [
        ("INPUT_FILE",("AttoPath","")),
        ("OFNAME",("AttoPath","")),
        ("FORCE",("int",0)),
        ("DEBUG",("int",0)),
                    ]
    CMD_TEMPLATE = CMD_TEMPLATE
    def _run(self):
        kw = self._data
        assert kw['INPUT_FILE']
        kw['INPUT_FILE'] = INPUT_FILE = kw['INPUT_FILE'].realpath()
        assert kw['OFNAME']
        kw['LAST_FILE'] = kw['OFNAME'] = OFNAME = kw['OFNAME'].realpath()
        
        FORCE = kw['FORCE']
        DEBUG = kw['DEBUG']
        CMD = pyext.jf2(CMD_TEMPLATE)
        
        if not FORCE and pyext.file__notEmpty(OFNAME):
            pass
        else:
            if DEBUG:
                pyext.printlines([CMD], OFNAME+'.debug.r')

            res = pyext.shellexec(CMD,silent=1)
            shutil.move(OFNAME + '.partial', OFNAME)
if __name__ == '__main__':
    test_csv_file = "/work/mapped-data/M20000000/0726-mRNA_readcounts_amalgamated.txt"
    csvFile_deseq2_vst({
        "INPUT_FILE": test_csv_file,
        "OFNAME":"/tmp/test2.csv",
        "DEBUG":1,"FORCE":1})
