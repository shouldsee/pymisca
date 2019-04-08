#!/usr/bin/env python2
import pymisca.header as pyheader
pyheader.base__check()
pyheader.execBaseFile('headers/header__import.py')
figs = pyext.collections.OrderedDict()


def worker(seed,stepSize=None):
    print seed
#     re = 1.00001
#     re = 0.00001
    re = 0.
    query = "per_SD > 0.95"
#     baseDist = 'alignedGaussianDistribution'
    baseDist = 'vmfDistribution'
    data = 'headers/mnist.csv'
    
    ALIAS = data.replace('/','-')[-20:]
    script = 'src/0408-freezerCluster.py'
    suc,res = pyext.job__baseScript(script,
                      opts = '\
                      --data {data} \
                      --quick 100 \
                      --cluMax 30 \
                    --start 0.1 --end 35.5 \
                    --XCUT 85 \
                    --baseDist {baseDist} \
                    --query "{query}" \
                    --seed {seed} \
                    --debug 1'.format(**locals()),
#                     DATENOW='{ALIAS}-{baseDist}-{seed}'.format(**locals()),
                    prefix='results/%s'%pyext.getBname(__file__),
        )
    
    assert suc,res
    
pyext.mp_map( funcy.partial(worker,), range(150,156),NCORE=6);