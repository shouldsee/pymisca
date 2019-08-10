import numpy as np
# import pymisca.vis_util as pyvis
# plt = pyvis.plt

import sklearn.decomposition


from pymisca.numpy_extra import cdist


def ax__drawVector(ax, v0, v1,**kw):
#     ax.plot()
    kw.setdefault('linewidth',2)
    arrowKw = arrowprops=dict(
#                     linewidth=2,
                    **kw)

    lines=  ax.plot(*zip(v0,v1),**arrowKw)
    return map(pyvis.add_arrow,lines)
    
def pca__fitPlot(X=None,pca=None,ax=None, silent = 1, **kw):
    silent =1
    if pca is None:
        assert X is not None
        pca = sklearn.decomposition.PCA(n_components=2)
        pca.fit(X)

    if not silent:
        ax = ax or plt.gca()
#         print ('[1]',pca.explained_variance_, pca.components_)
        for length, vector in zip(pca.explained_variance_, pca.components_):
            v = vector * 3 * np.sqrt(length)
            ax__drawVector(ax, pca.mean_, pca.mean_ + v,**kw)
        ax.axis('equal');
    return pca
# pyext.pca__fitPlot = pca__fitPlot


import copy
def pca__alignToPrior(mdl, prior,inplace=0,silent=1):
    silent=1
    if not inplace:
        mdl = copy.deepcopy(mdl)
    out0 = out = cdist(prior, mdl.components_, distF= np.inner)
    mapper = np.argmax(abs(out),axis=1,)
    mapper = list(mapper)
    assert len(mapper) == len(set(mapper)),'Cannot uniquely map prior vectors to PCs'
    

    #### invert eigenvector where applicable
    for i,_ in enumerate(prior):
        proj = out[i,mapper[i]]
#         pcv = mdl.components_[mapper[i]]
        if proj < 0:
            mdl.components_[mapper[i]] = - mdl.components_[mapper[i]]
#             mdl.components_[i] = -pcv
            
    #### reorder components
    _static_attrs = ['components_','explained_variance_','explained_variance_ratio_','singular_values_']
#     print ('mapper',mapper,)
    _func = lambda x:x[mapper]
    _obj = mdl
    for key in _static_attrs:
        v = getattr(_obj,key)
        _v = _func(v)
#         print ( key, _v.shape ,v.shape)
        setattr(_obj, key, _v)    

    out = cdist(prior, mdl.components_, distF= np.inner)
    if not silent:
        pyvis.heatmap(out0,vlim=[-1,1],cname='out0')
        pyvis.heatmap(out,vlim=[-1,1],cname='out')

    return mdl,out
# pyext.pca__alignToPrior = pca__alignToPrior



if __name__ =='__main__':
    import pymisca.vis_util as pyvis
    plt = pyvis.plt    
    
    fig,axs = plt.subplots(1,2,figsize=[12,4])
    rng = np.random.RandomState(1)
    X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
    plt.sca(axs[0])
    plt.scatter(*X.T,alpha=0.3)
    pca = pca__fitPlot(X)

    prior = [[1,1],[1,-1]]
    mdl = pca


#     mdl = copy.copy(mdl)
    mdl = pca__alignToPrior(mdl,prior)[0]

    
    plt.sca(axs[1])
    plt.scatter(*X.T,alpha=0.3)
    pca = pca__fitPlot(pca=mdl)
    
    ax =axs[1]
    mdl = pca__alignToPrior(mdl=pca, prior=[[1,1]])[0]
    _ = pca__fitPlot(pca=mdl,silent=0,ax=ax)
    mdl = pca__alignToPrior(mdl=pca, prior=[[-1,-1]])[0]
    _ = pca__fitPlot(pca=mdl,silent=0,ax=ax)
    # ax.axis('auto')

    ax =axs[1]
    mdl = pca__alignToPrior(mdl=pca, prior=[[-1, 1]])[0]
    _ = pca__fitPlot(pca=mdl,silent=0,ax=ax)

    ax =axs[1]
    mdl = pca__alignToPrior(mdl=pca, prior=[[1, -1]])[0]
    _ = pca__fitPlot(pca=mdl,silent=0,ax=ax)
    
    axs[0].set_title('before align')
    axs[1].set_title('after align')
    
    