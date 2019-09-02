from pymisca.atto_jobs_list.url_request import url_request
import io
import pymisca.ext as pyext
import re
from pymisca.atto_job import AttoJob

def _getResp(pathway,node_colors):
    '''
    Ref: https://padua.readthedocs.io/en/latest/_modules/padua/visualize.html , Ctrl+F:kegg-bin
    '''
# from StringIO import StringIO
    with io.StringIO() as f:
    # with open('test.out') as f
        f.write(u'#hsa\tCOSMIC\n')
        for k, c in list(node_colors):
            f.write(u'%s\t%s\n' % (k, c ))

        f.seek(0)
        buf = f.read()
#         buf = unicode(buf).replace('\t','\x09')
#         pathway = "hsa05200"
        
    CMD = u'''curl -L 'https://www.kegg.jp/kegg-bin/mcolor_pathway' -H 'Connection: keep-alive' -H 'Cache-Control: max-age=0' -H 'Origin: https://www.kegg.jp' -H 'Upgrade-Insecure-Requests: 1' -H 'Content-Type: multipart/form-data; boundary=----WebKitFormBoundarye6OjqF0vxHXaUAHp' -H 'User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8' -H 'Referer: https://www.kegg.jp/kegg/tool/map_pathway3.html' -H 'Accept-Encoding: gzip, deflate, br' -H 'Accept-Language: zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7' --data-binary $'------WebKitFormBoundarye6OjqF0vxHXaUAHp\r\nContent-Disposition: form-data; name="map"\r\n\r\n{pathway}\r\n------WebKitFormBoundarye6OjqF0vxHXaUAHp\r\nContent-Disposition: form-data; name="unclassified"\r\n\r\n{buf}\r\n\r\n------WebKitFormBoundarye6OjqF0vxHXaUAHp\r\nContent-Disposition: form-data; name="s_sample"\r\n\r\ngradation\r\n------WebKitFormBoundarye6OjqF0vxHXaUAHp\r\nContent-Disposition: form-data; name="mapping_list"; filename=""\r\nContent-Type: application/octet-stream\r\n\r\n\r\n------WebKitFormBoundarye6OjqF0vxHXaUAHp\r\nContent-Disposition: form-data; name="mode"\r\n\r\nnumber\r\n------WebKitFormBoundarye6OjqF0vxHXaUAHp\r\nContent-Disposition: form-data; name="numericalType"\r\n\r\nmm\r\n------WebKitFormBoundarye6OjqF0vxHXaUAHp\r\nContent-Disposition: form-data; name="minColor"\r\n\r\n#ffff00\r\n------WebKitFormBoundarye6OjqF0vxHXaUAHp\r\nContent-Disposition: form-data; name="maxColor"\r\n\r\n#ff0000\r\n------WebKitFormBoundarye6OjqF0vxHXaUAHp\r\nContent-Disposition: form-data; name="negativeColor"\r\n\r\n#00ff00\r\n------WebKitFormBoundarye6OjqF0vxHXaUAHp\r\nContent-Disposition: form-data; name="zeroColor"\r\n\r\n#ffff00\r\n------WebKitFormBoundarye6OjqF0vxHXaUAHp\r\nContent-Disposition: form-data; name="positiveColor"\r\n\r\n#ff0000\r\n------WebKitFormBoundarye6OjqF0vxHXaUAHp\r\nContent-Disposition: form-data; name="reference"\r\n\r\nwhite\r\n------WebKitFormBoundarye6OjqF0vxHXaUAHp--\r\n'
'''
    CMD = CMD.format(**locals())
#     CMD = pyext.jf2(CMD.strip())
#     CMD = [
#     "curl -L 'https://www.kegg.jp/kegg-bin/mcolor_pathway'",
#     "-H 'Connection: keep-alive'",
#     "-H 'Cache-Control: max-age=0'"
#     " -H 'Origin: https://www.kegg.jp'",
#     "-H 'Upgrade-Insecure-Requests: 1'"
#     " -H 'Content-Type: multipart/form-data; boundary=----WebKitFormBoundarysr4XwxFZws97zqeF'",
#     " -H 'User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36'",
#     " -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8'",
#     " -H 'Referer: https://www.kegg.jp/kegg/tool/map_pathway3.html' -H 'Accept-Encoding: gzip, deflate, br'",
#     " -H 'Accept-Language: zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7'",
#     ('''
#     --data-binary $'------WebKitFormBoundarysr4XwxFZws97zqeF\r\nContent-Disposition: \
# form-data; name="map"\r\n\r\n'''+pathway+'''\r\n------WebKitFormBoundarysr4XwxFZws97zqeF\r\n\
# Content-Disposition: form-data; name="unclassified"\r\n\r\n'''+buf+'''\r\n\r\n------WebKitFormBoundarysr4XwxFZws97zqeF\r\n\
# Content-Disposition: form-data; name="s_sample"\r\n\r\ngradation\r\n------WebKitFormBoundarysr4XwxFZws97zqeF\r\n\
# Content-Disposition: form-data; name="mapping_list"; filename=""\r\nContent-Type: application/octet-stream\
# \r\n\r\n\r\n------WebKitFormBoundarysr4XwxFZws97zqeF\r\n\
# Content-Disposition: form-data; name="mode"\r\n\r\nnumber\r\n------WebKitFormBoundarysr4XwxFZws97zqeF\r\n\
# Content-Disposition: form-data; name="numericalType"\r\n\r\nmm\r\n------WebKitFormBoundarysr4XwxFZws97zqeF\r\n\
# Content-Disposition: form-data; name="minColor"\r\n\r\n#ffff00\r\n------WebKitFormBoundarysr4XwxFZws97zqeF\r\n\
# Content-Disposition: form-data; name="maxColor"\r\n\r\n#ff0000\r\n------WebKitFormBoundarysr4XwxFZws97zqeF\r\n\
# Content-Disposition: form-data; name="negativeColor"\r\n\r\n#00ff00\r\n------WebKitFormBoundarysr4XwxFZws97zqeF\r\n\
# Content-Disposition: form-data; name="zeroColor"\r\n\r\n#ffff00\r\n------WebKitFormBoundarysr4XwxFZws97zqeF\r\n\
# Content-Disposition: form-data; name="positiveColor"\r\n\r\n#ff0000\r\n------WebKitFormBoundarysr4XwxFZws97zqeF--\r\n\
# Content-Disposition: form-data; name="reference"\r\n\r\nwhite\r\n------WebKitFormBoundarysr4XwxFZws97zqeF--\r\n\
# '
# ''').strip()
#     ]
#     CMD = pyext.stringList__flatten(CMD)
#     CMD = ' '.join(CMD)

    res = pyext.shellexec(CMD,silent=1)
#     assert 0,(len(res),res[:1000])
    return res

class httpJob_keggPathwayColored(AttoJob):
    PARAMS_TRACED = [
        ('KEGG_PATHWAY_ID',('unicode','')),
        ('NODE_COLORS',('list:object',[])),
        ('OUTPUT_FILE',('AttoPath','')),
        ('FORCE',('int',0)),
    ]
    def _run(self):
        kw = self._data
        assert kw['OUTPUT_FILE']
        KEGG_PATHWAY_ID = kw['KEGG_PATHWAY_ID']
        NODE_COLORS = kw['NODE_COLORS']
        kw['OUTPUT_FILE'] = OUTPUT_FILE = kw['OUTPUT_FILE'].realpath()
        FORCE = kw['FORCE']
        pyext.real__dir(fname= OUTPUT_FILE)
        res = _getResp( KEGG_PATHWAY_ID, NODE_COLORS)
        with open(OUTPUT_FILE+'.html','w') as f:
            f.write(res)
        # node_colors)
#         tmp/mark_pathway1566923389142087/hsa04141.png
        ms = re.finditer("(/tmp/mark_pathway[^']*?.png)", res)
        m = list(ms)[0]
        node= url_request({
            "URL":'http://www.kegg.jp%s'%m.group(1),
            'FORCE':FORCE,
            "OFNAME":OUTPUT_FILE})

if __name__ == '__main__':
    httpJob_keggPathwayColored(dict(
            KEGG_PATHWAY_ID = 'hsa05200',
            NODE_COLORS = [('hsa:25',678)],
            OUTPUT_FILE = 'test.png',
            FORCE = 1,
    ))
