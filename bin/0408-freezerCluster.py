#!/usr/bin/env python2
import pymisca.header as pyheader
pyheader.base__check()


# pyheader.execBaseFile('headers/header__import.py')

import pymisca.ext as pyext
pd = pyext.pd; np =pyext.np; 
pyheader.mpl__setBackend('agg')
import matplotlib.pyplot as plt

figs = pyext.collections.OrderedDict()



import pymisca.jobs as pyjob
import pymisca.callbacks as pycbk
import pymisca.numpy_extra as pynp
import slugify
import pymisca.blocks


import pymisca.util as pyutil ### render__images and get__cluCount

import synotil.CountMatrix as scount
import synotil.PanelPlot as spanel

# import pymisca.models as pymod
# import pymisca.iterative.em

# import pymisca.iterative.resp__entropise
# import pymisca.iterative.weight__entropise
# np.random.seed(0)

import argparse
class RawTextArgumentDefaultsHelpFormatter(
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawTextHelpFormatter
    ):
        pass
parser= argparse.ArgumentParser(description=__doc__,
                                formatter_class=RawTextArgumentDefaultsHelpFormatter,
#                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter
#                                formatter_class=argparse.RawDescriptionHelpFormatter
                               )
# parser.add_argument('--version', default = '0421', type=unicode)
parser.add_argument('--version', default = '0423', type=unicode)

parser.add_argument('--silent', default = 0, type=int)
parser.add_argument('--debug', default = 0, type=int)
parser.add_argument('--data', type=unicode)
parser.add_argument('--pathLevel', default=3, type=int)
parser.add_argument('--baseDist', 
                    default = 'vmfDistribution', type=unicode)
parser.add_argument('--quick', default = 1, type=int)
parser.add_argument('--XCUT', default = 40, type=int)

parser.add_argument('--YCUT', default = None, type=float)

parser.add_argument('--start', default = 0.25, type=float)
parser.add_argument('--end', default = 4.0, type=float)
parser.add_argument('--baseFile', type=int, default=1)
# parser.add_argument('--CUTOFF_SCORE', default = None, type=float)
parser.add_argument('--seed', default = 0, type=int)
parser.add_argument('--cluMax', default = None, type=int)
# parser.add_argument('--stepSize', default = 0.1, type=float)
parser.add_argument('--stepSize', default = 0.1, type=float,
                   help='!CAUTIOUS! in changing this parameter')
parser.add_argument('--reduce_entropy', default = None, type=float)
parser.add_argument('--lossTol', default = 0.01, type=float)

parser.add_argument('--query', default = 'index==index', type=unicode)

parser.add_argument('--figsize', default = [12,8],nargs=2,type=float )
parser.add_argument('--width_ratios', default = [3,10,0],nargs=3, type=float)
parser.add_argument('--update_proba', default = 1.0, type=float)
#         figsize=[ 12, 8 ],
#         width_ratios = [3,10,0],        

# data = '/home/shouldsee/work/ana__900'
    

    
def main(**kwargs):        
#     print (type(quick))

    class space:
        dataName = 'test'
        baseDist = kwargs['baseDist']
        seed = kwargs['seed']
        stepSize = kwargs['stepSize']
        end = kwargs['end']
        
        
    def check_data(data,pathLevel,baseFile,**kwargs):
#         data= kwargs['data']
        
        assert data is not None
        if isinstance(data,basestring):
            res = pyext.splitPath(unicode(data),pathLevel)[1]
            space.dataName = pyext.path__norm(res)
#             space.dataName = slugify.slugify(unicode(res))
    #         alias += pyext.getBname(clu)
            data = pyext.readBaseFile(data,baseFile=baseFile)
        
        if not isinstance(data,pd.DataFrame):
            data = data.tolist()
            
#         assert 1
        if not isinstance(data,scount.countMatrix):
            data = scount.countMatrix(data,)
        data.height=10        
        return data
    
    if kwargs['debug']:
        print(pyext.ppJson(kwargs))

    kwargs['data'] = check_data(**kwargs)
#     pyext.printlines(kwargs.items())
    
    ODIR = '{space.dataName}/baseDist-{space.baseDist}_seed-{space.seed}_end-{space.end}_stepSize-{space.stepSize}/'.format(**locals())
    

    def jobInDIR(
        data= None,
        baseDist = None,
        XCUT = None,
        # XCUT = 50
        YCUT = None,
        figsize= None,
        width_ratios = None,
#         figsize=[ 12, 8 ],
#         width_ratios = [3,10,0],        
        quick = None,
        start = None,
        end = None, 
        baseFile=None,
    #     CUTOFF_SCORE=None,
        stepSize = None,
        cluMax = None,
        seed = None,
        verbose = None,
        reduce_entropy = None,
        query = None,
        silent=None,    
        lossTol = None,
        debug = None,
        pathLevel = None,
        version = None,
        update_proba = None,
    ):
        dfc = data
#         dfc = _data
    #     dfc = sutil.meanNorm(dfc)


        #### start clustering
        config = {}
        config['baseDist'] = baseDist
    #     callbacks = pymisca.callbacks.

        #### set cluster number and iteration clip
        if quick == 1:
            config['K'] = 10
            config['nIter'] = 50
    #         job = pyext.functools( pyjob.EMMixture__anneal,
    #         mdl0 = pyjob.EMMixture__anneal(dfc, 
    #                                        start=start, end=end,seed=seed,
    # #                                         K=10,
    #                                         nIter=50,**config)
            XCUT = min(49,XCUT)

        elif quick ==0 :
            config['K'] = 30
            config['nIter'] = 200
    #         job = pyext.functools( pyjob.EMMixture__anneal,
    #         mdl0 = pyjob.EMMixture__anneal(dfc, 
    #                                        start=start, end=end,seed=seed,
    # #                                         K=30,
    #                                         nIter=200,**config)
        elif quick > 1 :
            config['K'] = 30
            config['nIter'] = quick 

        if cluMax is not None:
            config['K'] = cluMax




        ##### set callbacks
        callbacks =  []
        if verbose:
            callbacks += [ pycbk.verbose__callback ]
#         if reduce_entropy is None:
#             reduce_entropy = cluMax
    #     if reduce_entropy:
    #         callbacks += [ pycbk.weight__entropise(beta = reduce_entropy) ]
    #     if reduce_entropy:
#         if 1:
    #         callbacks += [ pycbk.resp__entropise(beta = reduce_entropy) ]
    #         callbacks += [ pycbk.MCE(stepSize = reduce_entropy) ]
    #         callbacks += [ pycbk.MCE(beta = reduce_entropy,
    #                                  stepSize=stepSize*len(dfc),

    #                                  lossTol = lossTol,
    #                                  speedTol = 1000.,

    #                                  maxIter=50, debug = debug,
    #                                 aggFunc='mean') ]

    #         callbacks += [pycbk.resp__sample(n_draw=reduce_entropy)]
        if version == '0421':
            
            callbacks += [pycbk.resp__MCE__surfer(
                stepSize=stepSize,
    #             beta = reduce_entropy,
                maxIter=5,
                debug=debug,
    #                                               maxIter=reduce_entropy
                                                 )]
        elif version =='0422':
            callbacks += [
                pycbk.resp__random__dirichlet(),]
        elif version =='0423':
            callbacks += [
                pycbk.resp__momentum(alpha=stepSize)
                ,]
        else:
            assert 0, version

        config['callbacks'] = callbacks



        job = pyext.functools.partial( pyjob.EMMixture__anneal,
                                       start=start, end=end,seed=seed,
                                      update_proba = update_proba,
                              **config)
    #     mdl0 = job(dfc)
    #         mdl0 = pyjob.EMMixture__anneal(dfc, 
    #                                        start=start, end=end,seed=seed,
    #                                         K=30,
    #                                         nIter=quick,

    #     if cluMax
        if quick != -1:
            mdl0 = job(dfc)
            np.save('mdl0.npy',mdl0)
        else:
            mdl0 = np.load('mdl0.npy').tolist()

        XCUT = min(len(mdl0.callback.mdls)-2, XCUT)
        pycbk.qc__vmf__speed(mdl0,XCUT=XCUT,YCUT=YCUT)
        figs['qcVMF'] = plt.gcf()


        ##### Post visualisation

        # mdl0 = np.load('mdl0.npy').tolist()


        mdl = mdl0.callback.mdls[XCUT][-1]
        clu  = mdl.predictClu(mdl0.data,
                              entropy_cutoff = YCUT,
                              index=mdl0.data.index)
        # score = mdl.score(mdl0.data)
        score = pynp.logsumexp( 
            mdl._log_pdf(mdl0.data),axis=1)
        score = score / mdl.dists[0].beta
        clu['score'] = score 
        print pyutil.get_cluCount(clu).T
        clu.to_csv('clu.csv')

        vdf = dfc.copy()
    #     vdf = vdf.reindex(columns = rnaCurr.index & vdf.columns)
    #     vdf=  scount.countMatrix(vdf,colMeta = rnaCurr)

        # vdf =sutil.sumNorm(scount.countMatrix(rnaTable.astype(float),
        #                                      colMeta =rnaCurr))
        # .apply(np.sqrt)
    #     vdf.relabel(colLabel=keys)

        # clu.clu.replace({0:3,1:3,2:3},inplace=True)


        ###### post-filtering
        dfc = dfc.qc_Avg();
        dfc.summary = pd.concat([dfc.summary,clu],axis=1)
    #         stats = dfc.summary
#         cluc = dfc.summary.query('clu > -1 & %s' % query)
        cluc = dfc.summary.query(' %s' % query)
    #         cluc = clu.query('clu>-1 & score > @CUTOFF_SCORE')
        cluc.to_csv('cluc.csv')

        if not silent:
            if 1:
                fig = plt.figure()
    #             cbk = [x for x in mdl0.callbacks if isinstance(x,pycbk.MCE)]
                cbk = [x for x in mdl0.callbacks if hasattr(x, '_hist')]
                if cbk:
                    cbk = cbk[0]
        #             x = mdl0.callbacks[0]._hist
                    for i,y in enumerate(cbk._hist):
                        plt.plot(y['loss'])
#                         if debug:
#                             print np.diff(y['loss'])[-3:]
                        plt.plot(i,y['loss'][-1],'x')
                    figs['MCE-loss-hist']= fig

            if 1:
                fig = plt.figure()
                clu.hist('score',bins=30)
                figs['hist-score'] = plt.gcf()
            
            
            if 1:
                pp = spanel.panelPlot([spanel.fixCluster(clu['clu']), 
                                       vdf,
                                      ],figsize=figsize,
                                      width_ratios=width_ratios,

                                       show_axa=1
                                     )
                # pp.render(order = clu )
                fig = pp.render(order = clu,
        #                   index = clu.query('clu>=-1').index
                         );
                figs['heatmap-all'] = fig

            if 1:
                pp = spanel.panelPlot([spanel.fixCluster(clu['clu']), 
                                       vdf,
                                      ],figsize=figsize,
                                      width_ratios=width_ratios,

                                       show_axa=1
                                     )

    #             dfc = dfc.qc_Avg();
    #             dfc.summary = pd.concat([dfc.summary,clu],axis=1)
    #     #         stats = dfc.summary
    #             cluc = dfc.summary.query('clu > -1 & %s' % query)
    #     #         cluc = clu.query('clu>-1 & score > @CUTOFF_SCORE')
    #             cluc.to_csv('cluc.csv')


                if len(cluc)==0:
                    index = None
                else:
                    index = cluc.index
                pp.index = index
                fig = pp.render(order = clu,
        #                   index = index
                         );

                figs['heatmap-filter'] = fig

            if 1:
                pyutil.render__images(figs)

        pymisca.blocks.printCWD()()
        return mdl0
    
    

    res = pyext.func__inDIR(
        pyext.functools.partial(jobInDIR,**kwargs), 
        DIR=ODIR)
    return res


if __name__=='__main__':
    args = parser.parse_args()
    main(**vars(args))