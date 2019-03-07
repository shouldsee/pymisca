import re
# import pymisca.shell as pysh
import pandas as pd
def getCounts__bowtie2log(buf):
    ptn = re.compile(r'([\d\.\>]+)')
    res = pd.DataFrame(
        map(ptn.findall,buf.splitlines()),
        columns=['count','per','how']).set_index('how')
    return res.to_dict(orient='index')