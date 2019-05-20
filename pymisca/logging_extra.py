import logging
from logging import *
def logger__addBasicHandler(logger=None,**kwargs):
    if logger is None:
        logger = logging.root

    logging._acquireLock()
    try:
        if 1:
            filename = kwargs.get("filename")
            if filename:
                mode = kwargs.get("filemode", 'a')
                hdlr = FileHandler(filename, mode)
            else:
                stream = kwargs.get("stream")
                hdlr = StreamHandler(stream)
            fs = kwargs.get("format", BASIC_FORMAT)
            dfs = kwargs.get("datefmt", None)
            fmt = Formatter(fs, dfs)
            hdlr.setFormatter(fmt)
            logger.addHandler(hdlr)
            level = kwargs.get("level")
            if level is not None:
                logger.setLevel(level)
    finally:
        logging._releaseLock()