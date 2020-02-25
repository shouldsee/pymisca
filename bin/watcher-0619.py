# %load /data/repos/pymisca/bin/directory_hashmirror_0520.py
#!/usr/bin/env python2
'''
Usage:
    
Purpose:
    execute a single job- align dataset H0000000 to 'Chlamydomonas_reinhardtii'
'''


### importing appropriate modules
import pymisca.ext as pyext
import pymisca.ptn
import pymisca.jobs
import argparse
pd = pyext.pd

import watchdog.events
import watchdog.observers
import watchdog.observers.polling
import time,sys,os,logging,glob,datetime
# ,logging,glob
import path

import imp



DEFAULT_UID = '65529'


parser= argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--INPUTDIR',type=unicode, 
                    help="A directory to be hashedT")
parser.add_argument('--minIntervalSeconds',type=int, default=30,
                    help="")
parser.add_argument('--callbackModule',type=unicode, 
                    help="A python2 file containing 'def main():' as a callback")
# /data/repos/fastqDB/fastqDB/callback-0621.py

REGEX_EXCLUDE = ['.*\.watchdog/.*',
                 '(^|.*/)[_\.][^/]*',
                 '.*\.git/.*',
                ]

def get__metaCols(df,key):
    assert df.index.name=='ROWNAME',df.index
    res = df.T
    res = res.get(key,[])
    if not len(res):
        print ('[WARN] TEMPLATE.tsv lacks ROW named "{key}"'.format(**locals()))
        return res
    else:
        res = res.fillna(0).astype(int) == 1
        res = pyext.series2index(res).tolist()
    return res

# INDEX_COLS = ['BOX_NUMBER','TUBE_NUMBER']
# FILE_COLS = ['PLASMID_MAP_FILE']

def get__templateFile(INPUTDIR):
    with pyext.getPathStack([INPUTDIR],force=1):
        ptn = 'TEMPLATE.*sv'
        res = glob.glob(ptn)
        assert len(res)==1,pyext.f('Cannot find a unique template file, with ptn={ptn}')
        res = res[0]
        return res


class MyEventHandler(watchdog.events.RegexMatchingEventHandler):
    """Logs all the events captured."""
    def __init__(self,INPUTDIR,
                 logger= None, 
                 callback = None,
                 *a,**kw
#                  minIntervalSeconds = 30
                ):
        self.callback  = callback
        self.INPUTDIR = INPUTDIR
        if logger is None:
            logger = logging.getLogger()
        self.logger = logger
        self.lastTime = datetime.datetime.now()
        
        self.TEMPLATE_FILE = get__templateFile(INPUTDIR)
        
        super(MyEventHandler,self).__init__(
            ignore_regexes = REGEX_EXCLUDE,
            ignore_directories = False,
            *a,**kw)
        
#         self.minIntervalSeconds = -1

    def on_any_event(self, event):
#     def dispatch(self, event):
        """Catch-all event handler.

        :param event:
            The event object representing the file system event.
        :type event:
            :class:`FileSystemEvent`
        """    
        if isinstance(event,watchdog.events.DirModifiedEvent):
            return 
        if str(event) == 'INIT':
            return 
#         while True:
#             tNow = datetime.datetime.now()
#             dt = (tNow - self.lastTime)
#             ddt = (dt.total_seconds() - self.minIntervalSeconds)
#             if ddt>0:
#                 self.lastTime = tNow
#                 break
#             else:
#                 toSleep = -ddt 
#                 self.logger.warn('Recevied signal, but gonna sleep for {toSleep} seconds'.format(**locals()))
#                 time.sleep(-ddt)
                
#         import watchdog.observers.api
# watchdog.observers.api.DEFAULT_EMITTER_TIMEOUT    

#         def db__init():
#             self.logger.warn('[db__init]')
#             pyext.pathDF__combineMetaTsv(_indDF,COLS)
#             pass

        def db__step(indDF):
            _indDF = indDF.query('FILEACC.str.contains("\.(tsv|csv)$")')
            _indDF = _indDF.query('~FILEACC.str.contains("TEMPLATE",case=0)')
            _indDF = _indDF.query('SIZE>0')
            
                
            with pyext.getPathStack([self.INPUTDIR]):
                FILES = ['_BACKUP/_STATUS.json','_MASTER.csv',]
                
#                 with pyext.getPathStack(['data'],force=1):
#                     path.Path('_README').touch()
#                     pass


                _templateDF = pyext.readData(self.TEMPLATE_FILE,guess_index=0).set_index('ROWNAME')
                COLS = _templateDF.columns
                INDEX_COLS = get__metaCols(_templateDF, 'IS_INDEX')
                FILE_COLS = get__metaCols(_templateDF, 'IS_FILE')
                
                
                ###### Compute _MASTER.csv
                MASTER_DF = pyext.pathDF__combineMetaTsv(_indDF, COLS=COLS,FILE_KEY='FILEACC',
                                                        guess_index=0)
                MASTER_DF = MASTER_DF.dropna(subset=INDEX_COLS)
                MASTER_DF.insert(0,'STATUS','[ACTIVE]') 


                MASTER_DF.loc[MASTER_DF.duplicated(subset=INDEX_COLS,keep=False),
                              'STATUS'] = '[index:DUPLICATED]'

                FLAG = (MASTER_DF[INDEX_COLS].isnull()).any(axis=1) \
                        | MASTER_DF[INDEX_COLS[0]].astype(str).str.strip().str.len()==0
                MASTER_DF.loc[FLAG,
                              'STATUS'] = '[index:INVALID]'

                for FILE_COL in FILE_COLS:
                    print (FILE_COL,)
                    _MASTER_DF = MASTER_DF.dropna(subset=[FILE_COL])
                    _MASTER_DF = _MASTER_DF.query('{FILE_COL}.str.strip().str.len() !=0 '.format(**locals()))
#                     FLAG = MASTER_DF.query('~{FILE_COL}.map(@pyext.file__notEmpty)'.format(**locals())).index
#                     FLAG = _MASTER_DF.query('~@_MASTER_DF[@FILE_COL].map(@pyext.file__notEmpty)').index

#                     _query = pyext.f('~ {FILE_COL}.map(@pyext.file__notEmpty)') 
#                     try:
#                         FLAG = _MASTER_DF.query( _query ,engine='python') .index
#                     except Exception as e:
#                         _MASTER_DF.to_pickle('/tmp/temp.pk')
#                         print (_query,)
#                         pyext.sys.exit(0)

                    FLAG = _MASTER_DF[ FILE_COL ].map(pyext.file__notEmpty)
                    FLAG = pyext.series2index(FLAG)
#                     assert 0, pyext.ppJson(dict(_query=_query,head=_MASTER_DF.head().to_string().splitlines()))

#                     FLAG  = ~MASTER_DF[FILE_COL].map(lambda x:pyext.file__notEmpty(x) if x else True)
#                     FLAG  = ~MASTER_DF[FILE_COL].map(pyext.file__notEmpty)
#                     FLAG &=  MASTER_DF[FILE_COL].str.strip().str.len() !=0 
                    MASTER_DF.loc[FLAG,'STATUS'] = pyext.df__format( MASTER_DF.loc[FLAG,],
                                                                   '{STATUS}[MISSING_FILE:{FILE_COL}]',
                                                                    FILE_COL=FILE_COL)
#                     MASTER_DF.loc[FLAG,'STATUS'] = 
#                 callback = None

                if self.callback is not None:
                    MASTER_DF = self.callback(MASTER_DF)
                ### sort before dump ####
                MASTER_DF =MASTER_DF.sort_values(['STATUS','SUBCSV_FNAME'] + INDEX_COLS)                

                MASTER_DF.to_csv('_MASTER.csv',index=0)
                ###########################
        
                ###############################
                ##### add _STATUS.json
                with pyext.getPathStack(['_BACKUP'],force=1):
                    path.Path('_README').touch()
                    pass
                    if pyext.file__notEmpty('_STATUS.json'):
                        DB = pyext.read_json('_STATUS.json')
                        DB['VERSION'] += 1                
                    else:
                        pyext.util__fileDict.main('_STATUS.json',argD={'VERSION':0})
                        DB = pyext.read_json('_STATUS.json')

                    DB['SUMMARY'] = MASTER_DF.groupby('STATUS').apply(len).to_dict()                
                    pyext.util__fileDict.main('_STATUS.json',argD=DB)
            
                    ####### exit context _BACKUP/
            
                FILES_FLAT = ' '.join(FILES)                
                _now = pyext.datenow()
                _USER = pyext.os.environ.get('UID', DEFAULT_UID)
                print(pyext.shellexec('pwd'),)
                pyext.shellexec(
                    pyext.template__format('''
git config user.name "{_USER}" && \
git config user.email "{_USER}@example.com" && \
git add {FILES_FLAT} \
    && git commit -F {FILES[0]} \
    && cp -p _MASTER.csv _BACKUP/VERSION-{DB["VERSION"]}.csv \
# &&  chmod -f -R 777 ./.git 
                    '''
                                           ,locals())
                            )
#                 pyext.shellexec('')
                self.logger.warn('[db__step]')
            pass
    
        db__init = db__step

#         res = indexDF = pyext.dir__indexify(self.INPUTDIR)
        def getIndex():
            res = indexDF = pyext.dir__indexify(self.INPUTDIR, OPTS_exec = 'du -al -B1 --apparent-size')
            res = res.reset_index(drop=1)[['SIZE','FILEACC','BASENAME']]
            for reg in REGEX_EXCLUDE:
                res = res.query('~FILEACC.str.match(@reg)')
            return res
        
        ##### check index
        _indDF = res = getIndex().sort_values('SIZE',ascending=False)
        if pyext.file__notEmpty('index.csv'):
            indDF = pyext.readData('index.csv',guess_index=0,index_col=[0])
            if indDF.equals(_indDF):
                'do nothing'
            else:
                db__step(_indDF)
        else:
            db__init(_indDF)
             
        res.to_csv('index.csv')
        msg = pyext.template__format('indDF.shape: {_indDF.shape}; event:{event}',locals())
        self.logger.warn(msg)
        return
    
def main(INPUTDIR,minIntervalSeconds,callbackModule,**kw):    
    del kw
    assert INPUTDIR is not None
    callback = callbackModule
    if callback is not None:
        if not callable(callback):
            callback = imp.load_source('_temp', callback).main
        
    INPUTDIR =os.path.realpath(INPUTDIR)
    with pyext.getPathStack([INPUTDIR,'_BACKUP'],force=1):
#     with pyext.getPathStack([INPUTDIR,'.watchdog'],force=1):
        import logging
        logger = logging.getLogger()
        del logger.handlers[:]

        loggingConfig = dict(level=logging.WARN,
                            format='%(asctime)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            )
        pyext.logger__addBasicHandler(
                            filename='hashmirror.log',
                            **loggingConfig)
        pyext.logger__addBasicHandler(
                            stream=sys.stdout,
                            **loggingConfig)




        event_handler = MyEventHandler(INPUTDIR = INPUTDIR,logger=logger,callback=callback,
    #                                    ignore_patterns
    #                                    minIntervalSeconds=minIntervalSeconds
                                      )

        observer = watchdog.observers.polling.PollingObserverVFS(
            pyext.os.stat,
            pyext.os.listdir,
            polling_interval=minIntervalSeconds,
        )
        observer.schedule(event_handler, INPUTDIR, recursive=True)
        observer.start()

        event_handler.on_any_event('INIT')    
        try:
            while True:
                time.sleep(minIntervalSeconds)
#                 pyext.shellexec('touch _ALIVE',silent=1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()    
    
if __name__ == '__main__':
    args = parser.parse_args()
    dargs = vars(args)
    dargs['UID'] = os.environ.get('UID',None)
    print (pyext.ppJson(dargs))
    main(**vars(args))