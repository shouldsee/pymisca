#!/usr/bin/env python2
import pymisca.header as pyheader
pyheader.base__check()
pyheader.execBaseFile('headers/header__import.py')
figs = pyext.collections.OrderedDict()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data',type=unicode)
parser.add_argument('--NJOB',default =3, type=int)


# %%time
# __file__
def worker(seed,data, stepSize=None,**kwargs):
    print seed
#     re = 1.00001
#     re = 0.00001
    re = 0.
    query = "per_SD > 0.95"
#     baseDist = 'alignedGaussianDistribution'
    baseDist = 'vmfDistribution'
#         data = 'results/0407-prefiltering-CAMICE/topped-20-nmsd-meannorm.pk'
#     data = 'headers/mnist.csv'

#     script = 'src/0408-freezerCluster.py'

    suc, res = pyext.job__scriptCMD(
        'repos/pymisca/bin/0408-freezerCluster.py \
--data {data} \
--quick 40 \
--cluMax 30 \
--start 0.01 \
--end 50. \
--XCUT 75 \
--debug 1 \
--query "{query}" \
--seed {seed} \
--baseDist {baseDist} \
'.format(**locals()),
     prefix='results/%s'%pyext.getBname(__file__),
    baseFile=1,
    )

    assert suc,res
def main(
#     data=None,
    NJOB=None,
    **kwargs):
    
    res = pyext.mp_map( pyext.functools.partial(worker,**kwargs),range(NJOB),NCORE=NJOB)
    return res

if __name__=='__main__':
    args = parser.parse_args()
    main(**vars(args))
# ! ln -s ~/repos .