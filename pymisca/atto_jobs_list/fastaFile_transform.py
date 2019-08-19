from pymisca.atto_job import AttoJob
import pymisca.shell, shutil
import pymisca.bio

class fastaFile_transform(AttoJob):
    PARAMS_TRACED = [
        ('OFNAME',("AttoPath",'')),
        ("INPUT_FILE",("AttoPath",'')),
        ("FUNC_OBJ2LINES",("object",lambda ele:ele[-1])),
        ("FORCE",("int",0)),
    ]
    def _run(self):
        kw = self._data
        assert kw['OFNAME']
        kw['OFNAME'] = OFNAME = kw['OFNAME'].realpath()
        assert kw['INPUT_FILE']
        kw['INPUT_FILE'] = INPUT_FILE = kw['INPUT_FILE'].realpath()
#         FILTER_FUNC = kw['FILTER_FUNC']
        FUNC_OBJ2LINES = kw['FUNC_OBJ2LINES']
        FORCE = kw['FORCE']
        pymisca.shell.real__dir(fname=OFNAME)
        if not FORCE and pymisca.shell.file__notEmpty(OFNAME):
            pass
        else:
            with open(OFNAME+'.partial', "w",buffering= -1) as f:
                with pymisca.bio.fastaIter(INPUT_FILE) as it:
                    for ele in it:
                        lines = FUNC_OBJ2LINES(ele)
                        map(f.write, lines)
                pass
            shutil.move(OFNAME+'.partial',OFNAME)
            