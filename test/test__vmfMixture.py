#!/usr/bin/env python
NCORE= 1
import sklearn.datasets as skdat
data_digit = data = skdat.load_digits()
din = data['data']
data_digit.keys()
y_true = data_digit['target']

# import mixem.distribution

import pymisca.model_collection.mixture_vmf as mod
reload(mod)
data = din


def worker(sample_weights):
    mdl = mod.MixtureVMF(init_method = 'kmeans',
                        NCORE=NCORE)
    res = mdl.fit(data,verbose=1,nStart=5,fix_weights=1,sample_weights=sample_weights,
                 )
    return mdl
# res = map(worker,lst)
# lst = ['sd','var','expSD','expVAR','constant']
lst = ['expVAR']
res = map(worker, lst)
res = dict(zip(lst,res))

# %matplotlib inline
def get__logProba(dists,weights,data):
    n_data = len(data)
    n_distr = len(dists)
    log_density = np.empty((n_data, n_distr))
    for d in range(n_distr):
        log_density[:, d] = distributions[d].log_density(data)
    logProba = np.log(weights[None,: ]) + log_density    
    return logProba


def getConfusionMat(y_pred,y_true,):
#     pred_targ = mdl.predict(test_data)
    dfc = pd.DataFrame(dict(pred=y_pred, ground=y_true))
    dfc['num'] = 1
    confusion = dfc.pivot_table(index='ground',columns='pred',values='num',aggfunc='sum').fillna(0.)
    return confusion


import sklearn.metrics as skmet
import matplotlib; matplotlib.use("Agg")
# logProba = get__logProba(dists,weights,data)
# y_pred = logProba.argmax(axis=1)
# mdl = MixtureModel(**dict(weights = weights, dists = dists))
import pymisca.util as pyutil;reload(pyutil)
import pymisca.vis_util as pyvis

pd = pyutil.pd
plt = pyutil.plt

for key,mdl in res.items():

    y_pred = clu = mdl.predict(data)
    clu = pd.DataFrame(y_pred,columns=['clu'])
    cluCount = pyutil.get_cluCount(clu)
    idx = cluCount.sort_values('count').clu
    confMat = getConfusionMat(y_pred,y_true)
    pyvis.heatmap(confMat.loc[:,idx],xtick=idx)
    ll = mdl.lastLL
    plt.title('sample_weight:{key}\nll={ll:.1f}'.format(**locals()))
    plt.grid(1)
    plt.savefig('confusionMat_key-{key}.png'.format(**locals()))
    
print ('[PASSED] no metric checked')
pyutil.sys.exit(0)