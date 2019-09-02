from pymisca.atto_job import AttoJob
from pymisca.atto_jobs_list.url_request import url_request
from pymisca.shell import file__unzip
from pymisca import shell
class url_download_unzip(AttoJob):
    PARAMS_TRACED = [
        ('URL',('AttoPath','')),
        ('OUTDIR',('AttoPath','')),
        ('FORCE',('int',0)),
    ]
    def _run(self):
#         shell = self.shell
        kw = self._data
        URL = kw['URL']
        kw['OUTDIR'] = OUTDIR = kw['OUTDIR'].realpath()
        FORCE = kw['FORCE']
        SIGNAL_FILE = OUTDIR / self.__class__.__name__+'.done'
        if not FORCE and shell.file__notEmpty(SIGNAL_FILE):
            pass
        else:
            node = url_request({"URL":URL,
                                 "PARENT":self,
                                  "OFNAME":OUTDIR+'.zip',
                                   "FORCE":FORCE,
                               })
            res = file__unzip(node['OFNAME'], OUTDIR)
            with open(SIGNAL_FILE,"w") as f: f.write("DONE");