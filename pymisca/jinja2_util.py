import jinja2 
from jinja2 import Template,Environment, BaseLoader
import pymisca.util as pyutil

def quickRender(templateFile,context, ofname =None, env=None,loader=None,searchpath=None):
    if loader is None:
        if searchpath is None:
            searchpath = [pyutil.os.environ.get('HOME',None),'/','.' ]
        loader = jinja2.FileSystemLoader(searchpath=searchpath)
    if env is None:
        env = Environment(loader=loader)
    t = env.get_template(templateFile)
    res = t.render(**context)
    
    if ofname is not None:
        pyutil.printlines([res],ofname)
        return ofname
    else:
        return res
