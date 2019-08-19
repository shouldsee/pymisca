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
import time


parser= argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--INPUTDIR',type=unicode, 
                    help="A directory to be hashedT")
class MyEventHandler(watchdog.events.FileSystemEventHandler):
    """Logs all the events captured."""
    def __init__(self,INPUTDIR):
        self.INPUTDIR = INPUTDIR

    def on_any_event(self, event):
        """Catch-all event handler.

        :param event:
            The event object representing the file system event.
        :type event:
            :class:`FileSystemEvent`
        """    
        res, msg = pymisca.jobs.dir__toHashDir(DIR=self.INPUTDIR)
        res.to_csv(self.INPUTDIR.rstrip('/')+'.index.csv')
        logging.info(msg)
        return res
    
def main(INPUTDIR):
    INPUTDIR = '/data/users/BASE/production/mapped-data/'
        
    event_handler = MyEventHandler(INPUTDIR = INPUTDIR)
    observer = watchdog.observers.Observer()
    observer.schedule(event_handler, INPUTDIR, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()    
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))

