from pymisca.atto_job import AttoJob
import pymisca.ext as pyext
TEMPLATE = '''
Rscript --slave --vanilla - <<EOF
suppressPackageStartupMessages({
    library(riboSeqR)
    })

source('{{R_PATCH}}',local=T)
env<- new.env()

read__findCDS <- function(FNAME){
    if (endsWith(FNAME,'.csv')){
        fastaCDS <- GRanges(read.csv(FNAME))
    }else if(endsWith(FNAME,'.rdata')){
        env <- new.env()
        load(file=FNAME,envir=env)
        env[['fastaCDS']]
    }else(
        stop(FNAME)
    )
    
}
result__findCDS <- read__findCDS("{{FILE_FINDCDS}}")

{
    ##### loading env[['fastaCDS']]

    gr = result__findCDS;
    result__findCDS <- gr <- gr[seqnames(gr)=="{{TRANSCRIPT_ID}}"]
    
}

riboDat <- riboSeqR::readRibodata(
            riboFiles = {{pyext.list__toR__vector(BAM_RIBO_LIST)}},
            rnaFiles =  {{pyext.list__toR__vector(BAM_RNA_LIST)}},
            columns = c(strand = 2, seqname = 3, start = 4, sequence = 5),
            replicates = {{pyext.list__toR__vector(REPLICATES)}}
)

filter.func <- {{R_FUNC_FILTER}}
env[['counts_filtered']] <- filter.func(riboDat, result__findCDS)

#par(mar=c(1,1,1,1))
{{d_axis['device']}}("{{OFNAME}}",
width={{d_axis['width']}},
height={{d_axis['height']}})

plotTranscript("{{TRANSCRIPT_ID}}",
               coordinates = env[['counts_filtered']]@CDS,
               riboData = riboDat, length = 28, 
               cap = 200)
dev.off();               
EOF
'''
def show__image(FNAME):
    if pyext.hasIPD:
        pyext.ipd.display(pyext.ipd.Image)
    pass
# from pymisca.atto_string import AttoPath
class riboseqr_plot_transcript(AttoJob):
            
    TEMPLATE = TEMPLATE 
    PARAMS_TRACED = [
#         ('BAM_FILES',('list:AttoPath',[])),
        ('FORCE',('int',0)),
        ('INPUT_FILE_DICT',('dict:AttoPath',
                            {'BAM_RIBO':'',
                             'BAM_RNA':'',
                             'FINDCDS':'',
                            
                            })),
        ('OFNAME',('AttoPath','')),
        ('TRANCSRIPT_ID',('unicode','')),
#         ('KEEP_TEMP',('int',0)),
        ('DEBUG',('int',0)),
        ('OPTS_AXIS',('dict:object',
                   {
                       'title':None,
                       'device':None,
                       'width':7,
                       'height':5,
                   })),
        ('R_PATCH',('AttoPath','/work/mapped-data/A20190807/src/riboseqr-patch.r')),
        ('R_FUNC_FILTER',('unicode',
                      '''
                   function(riboDat, fastaCDS){
                   ### input riboDat, fastaCDS and return ffCs
    fCs <- riboSeqR::frameCounting(riboDat, fastaCDS);
    fS <-  riboSeqR::readingFrame(rC = fCs); 
    ffCs <- filterHits_(fCs, 
                       lengths = c(26, 27, 28, 29, 30), 
                       frames = list(2, 1, 0, 2, 0), 
                       hitMean = 50, 
                       unqhitMean = 10 
                       ,fS = fS
                       )
    return(ffCs)
}
'''.strip()
                      )),

    ]
    def showImage(self):
        if pyext.hasIPD:
            res = pyext.ipd.Image(self['OFNAME'])
            pyext.ipd.display(res)
        else:
            res = None
        return res
        
    def __repr__(self):
        if pyext.hasIPD:
            pyext.ipd.display(pyext.ipd.Image(self['OFNAME']))
        return pyext.ppJson( self._data )
#         return "%r:%s"%(type(self),self['OFNAME'])
    
    def _run(self,pyext=pyext):
        
        kw = self._data
        assert kw['OFNAME'],("Must specify 'OFNAME'",)
        kw['LAST_FILE'] = kw['OFNAME'] = OFNAME = kw['OFNAME'].realpath()
        kw['OUTDIR'] = OFNAME.dirname()
#         KEEP_TEMP = kw['KEEP_TEMP']
        DEBUG = kw['DEBUG']
        INPUT_FILE_DICT = kw['INPUT_FILE_DICT']
        INPUT_FILE_DICT.setdefault('BAM_RNA','')
#         kw['INPUT_FILE_DICT'] = INPUT_FILE_DICT = {k:v.realpath() for k,v
#                                                   in kw['INPUT_FILE_DICT'].items()
#                                                   }
        TRANSCRIPT_ID = kw['TRANSCRIPT_ID']
        REPLICATES = ["UNKNOWN"]
        
        R_FUNC_FILTER = kw['R_FUNC_FILTER']
        R_PATCH = kw['R_PATCH']
        
        d_axis = kw['OPTS_AXIS']
        d_axis.setdefault('device',None)
        if d_axis['device'] is None:
            d_axis['device'] = OFNAME.rsplit('.',1)[1]
        if d_axis['device'] =='png':
            for key in ['width','height']:
                d_axis[key] *= 96. ### 1 inch is 96 pixel s
        FORCE = kw['FORCE']
        
        if not FORCE and pyext.file__notEmpty(OFNAME):
            return "SKIPPED"
#         if 0:
#             pass
        else:
            silent = 1
            with pyext.TempDirScope(keep=DEBUG) as stack:
                def _worker((i,BAM_FILE)):
                    OFNAME = stack.d / "%s-%s.bam"%(i,pyext.getBname(BAM_FILE))
                    CMD = ["samtools","view","-bh",BAM_FILE,TRANSCRIPT_ID,">",OFNAME]
                    CMD = pyext.stringList__flatten(CMD)
                    CMD = ' '.join(CMD)
                    pyext.shellexec(CMD,silent=silent)
                    return OFNAME
                
                for key in ["BAM_RIBO","BAM_RNA"]:
                    val = INPUT_FILE_DICT[key]
                    if val:
                        INPUT_FILE_DICT[key] = _worker((key,val.realpath()))
                    else:
#                         del INPUT_FILE_DICT[key]
                        pass
#                     pass

                BAM_RIBO_LIST = [ x for  x in [ INPUT_FILE_DICT['BAM_RIBO'] ] if x] 
                BAM_RNA_LIST = [ x for  x in [ INPUT_FILE_DICT['BAM_RNA'] ] if x] 
#                 BAM_FILES = [INPUT_FILE_DICT['BAM_RIBO']]
#         ,INPUT_FILE_DICT['BAM_RNA']]
#                 BAM_FILES = map(_worker,enumerate(BAM_FILES))
                FILE_FINDCDS = INPUT_FILE_DICT['FINDCDS']
                _FILE = FILE_FINDCDS + ".BY_CHROM/%s.csv"%(TRANSCRIPT_ID)
                if _FILE.isfile():
                    FILE_FINDCDS = _FILE


                CMD = pyext.jf2( self.TEMPLATE.replace('\t',''))
                pyext.printlines([CMD],"CMD.r")
                res = pyext.shellexec(CMD,silent=1)      
            kw['DIR_TEMP'] = stack.d 
            return "RUNNED"
#         * len(BAM_FILES)