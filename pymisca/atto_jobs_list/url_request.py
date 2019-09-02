# import pymisca.ext as pyext
import requests
# import pymisca.atto_string as pyatt
import shutil
from pymisca.atto_job import AttoJob
from pymisca.shell import file__notEmpty,real__dir
import urllib 

# urllib.urlretrieve('ftp://server/path/to/file', 'file')

class url_request(AttoJob):
    PARAMS_TRACED= [
        ("OFNAME",("AttoPath",'')),
        ('METHOD',('unicode',"GET")),
        ('URL',('AttoPath',None)),
        ('PARAMS',('dict:unicode',{})),
#         ('FILES',('dict:list:object',{})),        
        ('FILES',('dict:object:object',{})),        
        ('DATA',('dict:unicode',{})),
        ('JSON',('dict',{})),
        ('KW',('dict',{})),
        
        ('RUNTIME_BUFFER',('bool',False)),
        ('STREAM',('bool',True)),
        ('CHUNK_SIZE',('int',8192)),
        
        ("FORCE",('int',0)),
    ]
    
#     def THIS_FUNC(DB_WORKER):
    def _run(self):
        
        kw = RUN_PARAMS = self._data
#         DB_WORKER['RUN_PARAMS']
        kw['METHOD'] = METHOD = RUN_PARAMS['METHOD'].upper()
        URL = RUN_PARAMS['URL']
        params = PARAMS = RUN_PARAMS['PARAMS']
        files = FILES = RUN_PARAMS['FILES']
        data= DATA = RUN_PARAMS['DATA']
        json = JSON = RUN_PARAMS['JSON']
        KW = RUN_PARAMS['KW']
        FORCE = kw['FORCE']
        assert kw['OFNAME']
        kw['OFNAME'] = OFNAME = kw['OFNAME'].realpath()
        
#         FORCE = DB_WORKER.get('FORCE',0)

        stream = STREAM = RUN_PARAMS['STREAM']
        assert METHOD in ['GET','POST']


        chunk_size = CHUNK_SIZE = RUN_PARAMS['CHUNK_SIZE']
        if not FORCE and file__notEmpty(OFNAME):
            pass
        else:
            real__dir(fname=OFNAME)
            if URL.startswith("ftp://"):
                urllib.urlretrieve(URL, OFNAME+'.partial')
            elif URL.startswith("http"):
                if METHOD == 'GET':
                    res = requests.get(URL,params=params, stream = stream, **KW)
                elif METHOD =='POST':
                    res = requests.post(URL,data=data,json=json,files=files,stream=stream,**KW)
                res.raise_for_status()
            #     local_filename = 'OUT'
                real__dir(fname=OFNAME)
                with open( "%s.partial"%OFNAME,  'wb') as f:
                    for chunk in res.iter_content(chunk_size=chunk_size): 
                        if chunk: # filter out keep-alive new chunks
                            f.write(chunk)
                            # f.flush()   
                res.close()
            shutil.move("%s.partial"%OFNAME,OFNAME)

#         if RUN_PARAMS['RUNTIME_BUFFER']:
#             DB_WORKER['RUNTIME']['BUFFER'] = ''
#             with open('OUT','rb') as f:
#                 chunk = f.read(chunk_size)
#                 DB_WORKER['RUNTIME']['BUFFER']+= chunk
#         pass