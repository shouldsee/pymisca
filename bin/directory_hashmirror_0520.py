#!/usr/bin/env python2
'''
Usage:
    
Purpose:
    execute a single job- align dataset H0000000 to 'Chlamydomonas_reinhardtii'
'''


### importing appropriate modules
import pymisca.ext as pyext
import pymisca.jobs
import argparse

import watchdog.events
import watchdog.observers
import watchdog.observers.polling
import time,sys,os,logging,glob,datetime
# ,logging,glob




parser= argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--INPUTDIR',type=unicode, 
                    help="A directory to be hashedT")
parser.add_argument('--minIntervalSeconds',type=int, default=30,
                    help="")
class MyEventHandler(watchdog.events.LoggingEventHandler):
    """Logs all the events captured."""
    def __init__(self,INPUTDIR,logger= None, minIntervalSeconds = 30):
        self.INPUTDIR = INPUTDIR
        if logger is None:
            logger = logging.getLogger()
        self.logger = logger
        self.lastTime = datetime.datetime.now()
#         self.minIntervalSeconds = minIntervalSeconds
        self.minIntervalSeconds = -1

    def on_any_event(self, event):
        """Catch-all event handler.

        :param event:
            The event object representing the file system event.
        :type event:
            :class:`FileSystemEvent`
        """    
        while True:
            tNow = datetime.datetime.now()
            dt = (tNow - self.lastTime)
            ddt = (dt.total_seconds() - self.minIntervalSeconds)
            if ddt>0:
                self.lastTime = tNow
                break
            else:
                toSleep = -ddt 
                self.logger.warn('Recevied signal, but gonna sleep for {toSleep} seconds'.format(**locals()))
                time.sleep(-ddt)
                
#         import watchdog.observers.api
# watchdog.observers.api.DEFAULT_EMITTER_TIMEOUT    
        res, msg = pymisca.jobs.dir__toHashDir(DIR=self.INPUTDIR)
        res.to_csv(self.INPUTDIR.rstrip('/')+'.index.csv')
        self.logger.warn(msg)
        return
    
def main(INPUTDIR,minIntervalSeconds):
    
    assert INPUTDIR is not None
    
    import logging
    logger = logging.getLogger()
    del logger.handlers[:]

    loggingConfig = dict(level=logging.WARN,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        )
    pyext.logger__addBasicHandler(
                        filename=INPUTDIR+'.hashmirror.log',
                        **loggingConfig)
    pyext.logger__addBasicHandler(
                        stream=sys.stdout,
                        **loggingConfig)
  
    

        
    event_handler = MyEventHandler(INPUTDIR = INPUTDIR,logger=logger,
                                   minIntervalSeconds=minIntervalSeconds)
    
#     observer = watchdog.observers.Observer()
#     watchdog.observers.polling.DEFAULT_EMITTER_TIMEOUT = minIntervalSeconds
#     observer = watchdog.observers.polling.PollingObserver(
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
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()    
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
