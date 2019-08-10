import pymisca.header
import pymisca.ext as pyext
np = pyext.np; pd = pyext.pd
from pymisca.module_wrapper import tree__worker__interpret
import pymisca.vis_util as pyvis
plt = pyvis.plt
class plotters(object):
    @classmethod
    def fig__save(cls, fig,   ofname, transparent = False, bbox_inches = 'tight',facecolor = None,
                      **kwargs):
        if facecolor is None:
            facecolor = fig.get_facecolor()
        facecolor = 'white'
        _renderer=fig.canvas.get_renderer()
        bbox_inches = fig.get_tightbbox(_renderer)
        fig.patch.set_facecolor(facecolor)
        res = fig.savefig(ofname,
                    bbox_inches=bbox_inches,
                    transparent= transparent,
                    facecolor=facecolor,
                    **kwargs
                   )
        return res     
    
    @classmethod
    def job__process(cls, d,context=None):
        if context is None:
            context = pymisca.header.get__frameDict(level=1)
        _ = pyext.printlines([pyext.ppJson(d)],d['OFNAME']+'.json')
        d['FUNCTION'] = tree__worker__interpret(d['FUNCTION'],context)
        res = d['FUNCTION'](d,context) 
        return res    
    
    @classmethod
    def html__tableLine(cls,OFNAME):
        res = '''<table style="width:auto; height="75%" border="1">
        <tr>
        <th>
        <a href="{OFNAME}">{OFNAME}</a>
        <br>
        <a href="{OFNAME}.json">{OFNAME}.json</a>
        </th>
        </tr>
        <tr>
            <th>
            <img src="{OFNAME}"></img>
            </th>
        </tr>
        </table>
        '''.format(OFNAME=OFNAME)    
        return res
    @classmethod
    def boxplot(cls,d,context):
        
        assert "get__fcValues" in context
        d = tree__worker__interpret(d,context)
        OFNAME = d.get('OFNAME',None) 
        assert OFNAME,(pyext.ppJson(d),)

        d_ax = d.get('axis',{})
        d_ax = cls.dict__castAxis(d_ax)
#         ylim = d_ax.get('ylim',[])
#         ylabel = d_ax.get('ylabel',None)
#         figsize = d_ax.get('figsize',None)

        fig, ax = plt.subplots(1,1,figsize=d_ax['figsize'])


        if d_ax['ylim']:
            ax.set_ylim(d_ax['ylim'])

        if d_ax['ylabel']:
            ax.set_ylabel(d_ax['ylabel'])

        # d['datasets'] = 
        res = [pd.Series(_d['value'],name=_d['label']) for _d in d['datasets']]
        res = pd.DataFrame(res).T
        d['_df'] = res
        import scipy.stats
        # .ttest_rel

        # INDEX_FILE = '/home/feng/static/figures/1126__PIF7__tempResp-AND-pif7Resp/Venn-index.csv'
        # pyext.MDFile('/home/feng/static/figures/1126__PIF7__tempResp-AND-pif7Resp/Venn-index.csv')
#         index : "!{pyext.readData('/home/feng/static/figures/1126__PIF7__tempResp-AND-pif7Resp/Venn-index.csv',)['ind2'].dropna()}"
        # index = pyext.readData('/home/feng/static/results/0206__heatmap__PIF7/clu.csv').query('clu==7').index
        # print len(index)
        df = d['_df']
        index = d.get('index',[])
        if len(index):
            df = df.reindex(index)
        # testResult = scipy.stats.ttest_rel(*df.values.T[:2])
        testResult = scipy.stats.ttest_ind(*df.values.T[:2])

        ax.set_title('''
        independent-t-test-between-two-leftmost-samples
        p={testResult.pvalue:.3E}
        N={df.shape[0]}
        '''.format(**locals()))
        df.boxplot(rot='vertical',ax=ax)
        
        cls.fig__save(fig,OFNAME)
#         fig.savefig(OFNAME)
        res = cls.html__tableLine(OFNAME)

        return res
    
    @classmethod
    def dict__castAxis(cls, d_ax, context=None):
        res = dict(ylim = d_ax.get('ylim',None),
                   xlim=d_ax.get('xlim',None),
            ylabel = d_ax.get('ylabel',None),
            xlabel = d_ax.get('xlabel',None),
            figsize = d_ax.get('figsize',None),
            title = d_ax.get('title',''),
                  )
        res['legend.markerscale'] =  d_ax.get('legend.markerscale',None)
        return res
    
    @classmethod
    def venn_diagram(cls,d,context):
        d = tree__worker__interpret(d,context)
        import pymisca.proba
        d_ax = cls.dict__castAxis(d.get('axis',{}))
        
        OFNAME = d.get('OFNAME',None) 
        assert OFNAME,(pyext.ppJson(d),)
        
        
        d['index1']= pd.Index(d['index1']).dropna()
        d['index2']= pd.Index(d['index2']).dropna()
        if d.get('index_bkgd',None) is not None:
            pass
        else:
            d['index_bkgd'] = d['index1'] | d['index2']
        d['index_bkgd'] = pd.Index(d['index_bkgd']).dropna()
        
#         d['title'] = d.get('title', "Fisher exact test: p={pval}")
        fig, ax = plt.subplots(1,1,figsize=d_ax['figsize'])

        testResult = pymisca.proba.index__getFisher(cluIndex=d['index1'], 
                                                    featIndex=d['index2'])
        pval = '%.3E'%testResult['p']
        ax= plt.gca()
        res = pyvis.qc_index(d['index1'],d['index2'],
                             xlab=d_ax['xlabel'],ylab=d_ax['ylabel'],silent=0,ax=ax);
        ax.set_title(d_ax['title'].format(**locals()))    
        
        cls.fig__save(fig,OFNAME)
        res = cls.html__tableLine(OFNAME)
        
        return res
    
    COLORS = ['blue', 'green', 'red',  'magenta', 'orange', 'black']
    @classmethod
    def qc__2var(cls,d,context):        
        d = tree__worker__interpret(d,context)
        d_ax = cls.dict__castAxis(d.get('axis',{}))        
        OFNAME = d.get('OFNAME',None) 
        assert OFNAME,(pyext.ppJson(d),)
        
        fig,axs = plt.subplots(1,2,figsize=d_ax['figsize'],
                               gridspec_kw={'wspace':0.2,
                                            'width_ratios':[0.7,0.3]})    

        
        refline = d.get("refline",None)
        addPCA = d.get('addPCA',1)
        cmap = d.get('cmap', pyvis.mpl.colors.ListedColormap(cls.COLORS) )
        colorDict = d.get('colorDict',None)
        
    #     clu = None
#         xs = xdat['value'][col]
#         ys = ydat['value'][col]
        xs = d['xs']
        ys = d['ys']
        clu = d.get('clu',None)
        index = d.get('index',None) or getattr(xs,'index',None)
        assert index is not None


        if not isinstance(xs, np.ndarray):
            xs = getattr(xs,"values",np.array(xs))
        if not isinstance(ys, np.ndarray):
            ys = getattr(ys,"values",np.array(ys))


        coords = pyext.arr2d__pctransform(xs,ys,index=index)[0]
        ppf = coords.apply(pyext.dist2ppf)
        ax = axs[0]
        
        pyvis.qc_2var(xs,ys,nMax=-1,axs=[None,ax,None,None],refline=refline,clu=clu,cmap=cmap,
                      markersize=10,colorDict=colorDict,
    #                   alpha=0.5,
                      )
        ax.set_xlim(d_ax['xlim'])
        ax.set_ylim(d_ax['ylim'])
#         ax.set_title(title)
        ax.set_title(d_ax['title'].format(**locals()))
        ax.set_xlabel(d_ax['xlabel'] )
        ax.set_ylabel(d_ax['ylabel'] )

        if addPCA:
            X = np.vstack([xs,ys]).T
            pca = pymisca.sklearn_extra.pca__fitPlot( X,silent=1,ax=None)
            mdl = pymisca.sklearn_extra.pca__alignToPrior(mdl=pca, prior=[[1,1],[-1,1]])[0]
    #         print mdl.mean_

            ax.get_xscale()
            l = np.sqrt(np.diff(ax.get_xlim())[0]**2 + np.diff(ax.get_ylim())[0]**2) * 0.5
            line = ax.plot( *zip(mdl.mean_, mdl.mean_ + l * mdl.components_[0]),color='red',label='PC0')
            pyvis.add_arrow(line[0])
            line = ax.plot( *zip(mdl.mean_, mdl.mean_ + l * mdl.components_[1]),color='cyan',label='PC1')
            pyvis.add_arrow(line[0])
            
#         _markerscale=d_ax.get('legend.markerscale',None)
#         print(_markerscale,)
        axs[-1].legend(*ax.get_legend_handles_labels(),markerscale = d_ax['legend.markerscale'])
        pyvis.hide_Axes(axs[-1])
        #####
        cls.fig__save(fig,OFNAME)
        res = cls.html__tableLine(OFNAME)        
        return res        
#     @classmethod
#     def fig__save(cls,fig,OFNAME):
   
#         return pyext.fig__save(fig,OFNAME)
plt.rcParams['font.size'] = 14.
plt.rcParams['xtick.labelsize'] = 16.
plt.rcParams['axes.titlepad'] = 24

