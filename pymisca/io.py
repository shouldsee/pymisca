import codecs
import StringIO


def unicodeIO(handle=None, buf=None,encoding='utf8',*args,**kwargs):
    if handle is None:
        handle = StringIO.StringIO()
    codecinfo = codecs.lookup(encoding)
    wrapper = codecs.StreamReaderWriter(
        handle, 
        codecinfo.streamreader, 
        codecinfo.streamwriter)
    if buf:
        wrapper.write(buf)
        wrapper.seek(0)
    return wrapper