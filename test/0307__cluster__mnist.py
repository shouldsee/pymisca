#!/usr/bin/env python
# -*- coding: utf-8 -*-

NCORE = 3
# NJOB = 10 ###  number of  parallel  jobs
import pymisca.header as pyhead
pyhead.mpl__setBackend('agg')
# execfile(pyhead.base__file('headers/header__import.py'))
import pymisca.jobs as pyjob
import pymisca.callbacks as pycbk
import pymisca.util as pyutil
pd = pyutil.pd
import pymisca.vis_util as pyvis
import matplotlib.pyplot as plt

figs = pyutil.collections.OrderedDict()

def loadData():
    import sklearn.datasets as skdat
    data_digit = data = skdat.load_digits()
    din = data['data']
    data_digit.keys()
    y_true = data_digit['target']
    return din

tdf = loadData()
tdf = pd.DataFrame(tdf)



# ##### debugging!!!!!!!!!!!!
# tdf = pyutil.readData('http://172.26.114.34:81/static/results/0129__cluster__Brachy-RNA-all/mdl.npy').tolist().data
# ##### debugging!!!!!!!!!!!!


# lst = [75434668]

# betas = np.linspace(0.001,1.52, 25).tolist() + [1.52] * 25
# for i,r in enumerate(lst):

# def getBeta(i):
#     betas = [_betas[i]] * 100
#     return betas


def worker((i,r)):
#     betas = [3.0] * 25
#     betas  = getBeta(i)
    nIter = 100
    alias = 'i-%d_r-%d'%(i,r)
    
    
    
    mdl0 = pyjob.job__cluster__mixtureVMF__incr(
        normalizeSample=0, #### set to 1 to normalize the vector lenght
        tdf=tdf,
        meanNorm=1, ##### perform X = X-E(X)_
        weighted=True,
        init_method='random',
        nIter=nIter,
#         start=0.001, #### specify temperature range
#         end=2.0,
#         end=0.7,

        start=0.2, #### specify temperature range
#         end=2.0,
        end=0.7,
        
        #         betas = betas, #### alternatively, pass a callable for temperature
        randomState = r,
        alias = 'mdl_'+alias,  #### filename of cache produced
        verbose=2,
        K=60,
    )

    ##### produce diagnostic plot
    YCUT = entropy_cutoff=    2.5
    XCUT = step = 30    
    
    axs = pycbk.qc__vmf__speed(mdl0,
#                                XCUT=step,YCUT=entropy_cutoff  ### not working yet
                              )
    fig = plt.gcf()
    ax=  fig.axes[0]
#     pyvis.abline(y0=3.7,k=0,ax=ax)
    pyvis.abline(y0=YCUT,k=0,ax=ax)
    pyvis.abline(x0=XCUT,k=0,ax=ax)
    figs['diagnostic-plot'] = plt.gcf()
    

    
    #### using the last model to predict cluster
    mdls = mdl0.callback.mdls  #### models is recorded for each point
    mdl = mdls[step][-1]   #### getting the model at step
    clu = mdl.predictClu(tdf,
                        entropy_cutoff=entropy_cutoff)     
    clu.to_csv('cluster.csv') ### getting cluster assignment
    
    pyvis.heatmap( tdf.reindex(clu.sort_values('clu').index),
                 figsize=[14,7])
    figs['clustered-heatmap'] = plt.gcf()
    return (alias,fig)

res = [ worker((0,1)) ]

# figs.update(res)

# N = 5
# _betas = np.linspace(0, 2.0, N)

# np.random.seed(0)
# lst = np.random.randint(100000000,size=(N))
# it = enumerate(lst)
# res = pyutil.mp_map(worker,it,n_cpu=NJOB)
# res = res[::60//5]
# figs.update(res)

pyutil.render__images(figs,)    
# pyutil.render__images(figs,)