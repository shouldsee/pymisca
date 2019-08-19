from pymisca.atto_job import AttoJob
import json
import pymisca.ext as pyext
import xml.dom.minidom as minidom
from pymisca.jupyter_extra import nbjson__iterNodes__OutputData
from pymisca.xml_extra import minidom__iterElementByAttributeValue, minidom__walk
class ipynb_replace_htmlTable(AttoJob):
    PARAMS_TRACED = [
        ('INPUT_FILE_DICT',('dict:AttoPath',{})),
        ('OFNAME',('AttoPath','')),
        ('FORCE',('int',0)), ###[TBC]
    ]
    def _run(self):
        kw = self._data
        INPUT_FILE_DICT = d = kw['INPUT_FILE_DICT']
        d['HTML'] = FILE_HTML = INPUT_FILE_DICT['HTML'].realpath()
        d['IPYNB'] = FILE_IPYNB = INPUT_FILE_DICT['IPYNB'].realpath()
        
        kw['OFNAME'] = OFNAME = kw['OFNAME'].realpath()
        
#         FILE_HTML_TABLE = 'test-meta-file-table.html'
#         FILE_IPYNB = "G00000002-mm10.ipynb"
#         OFNAME = "test.ipynb"

        idVal = 'meta-file-table'
        tabNode = next(minidom__walk(minidom.parse(FILE_HTML)))
        tabNode.setAttribute('id','%s-replaced'%idVal)
        
        d  = pyext.readData( FILE_IPYNB, ext='json')
        it = nbjson__iterNodes__OutputData(d)
#         idVal = 'meta-file-table'
        def _it(it=it,idVal=idVal):
            for ele in it:    
                if ele[0][0]=="text/html":
                    buf = ''.join(ele[0][1])
                    dom = minidom.parseString(buf)
                    it  = pyext.minidom__iterElementByAttributeValue(dom,'id',idVal)
                    e = next(it,None)
                    if e is not None:
                        yield e,ele
        it = _it()
        val = next(it,None)
        assert val is not None, ("Cannot find table with id,",idVal,FILE_IPYNB)

        e,ele = val
        lst = e.parentNode.childNodes
        for i,_e in enumerate(lst):
            lst[i] = tabNode
        ele[0][1][:]= tab = e.parentNode.toxml().splitlines()

        with open(OFNAME,"w") as f:
            json.dump(d,f)
#         json.dump
        tab = next(_it(nbjson__iterNodes__OutputData(d,),idVal='meta-file-table-replaced'))[1][0][1]

if __name__ == '__main__':
    ipynb_replace_htmlTable({
        "INPUT_FILE_DICT":{
            "HTML":'test-meta-file-table.html',
#             "IPYNB":"G00000002-mm10.ipynb",
            "IPYNB":"test-0.ipynb",
        },
        "OFNAME":"test-1.ipynb",
    })
    import filecmp
    assert filecmp.cmp('test-1.ipynb', 'test-1.ipynb.expect')
#     assert 