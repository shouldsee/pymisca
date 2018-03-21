from util import *
import numpy as np
import  scipy.cluster.hierarchy as sphclu

try:
    import network
    namedict = {network.callback_exp_hamm:'EDNA',
                network.callback_nCorrect:'CRR',
                'sizelst':'Network size\n(I)',
                'nTarlst':'Number of stored patterns',
               'pcor':'$p_c$'}
except:
    namedict = {}
    print "[WARN] %s cannot find network" %__name__



def snap_detail(snap,YTICK = 0, truncate = 50, uniq = 1):
    if isinstance(snap,network.hopfield_discrete):
        h = snap
        attr = snap.out[0]
    else:
        h,attr,_,_ = snap
    H = len(attr)
    attr = canonlise(attr)
    if uniq:
        attr,count = np.unique(attr,axis = 0,return_counts=1)
    im = canonlise(np.vstack([attr*0.5,h.target]))
    _,lab = make_label(attr,h.target,)
    imclu = canonlise(np.vstack([attr,h.target]))
    Z = sphclu.linkage(imclu, 
    #                    'complete',
    #                    'single',
                       'average',
                       metric=symmetric_hamm)
    ##### Calculating optimal ordering
    dendo = sphclu.dendrogram(Z,no_plot=1)
    ridx = dendo['leaves']
#     print len(uniq), len(h.target)
    # plt.imshow(im[ridx][-truncate:],)
    fig,axs = plt.subplots(2,1, gridspec_kw = {'height_ratios':[1, 3]},
                          figsize = plt.gcf().get_size_inches())
    plt.sca(axs[1])
    if not YTICK:
        lab = ['*' if l.startswith('*') else '' for l in lab]
    if YTICK:
        plt.pcolormesh(im[ridx][-truncate:].T,)
    else:
        plt.pcolormesh(im[ridx].T,)
    if YTICK:
        y,ytick =make_label( attr, h.target,)
#         print ridx
        y =  np.array(y);ytick = np.array(ytick)[ridx]
#         print ytick[::-1]
        plt.xticks(y +.5,ytick[::-1], rotation='vertical')
    
    plt.xlabel('Pattern')
    plt.sca(axs[0])
    plt.title('H=%d,I=%d,\n\#Sampled attractor=%d'%(H,
                                                 attr.shape[1],
                                                 len(attr)))
    plt.ylabel('Symmetric\n hamming dist.')
    sphclu.dendrogram(
        Z,
    #     truncate_mode='lastp',  # show only the last p merged clusters
    #     p=12,  # show only the last p merged clusters
        show_leaf_counts=False,  # otherwise numbers in brackets are counts
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,  # to get a distribution impression in truncated branches
        labels = ['']*len(lab)
    )
    plt.ylim([-.05,None])
    return fig,axs

def snaps_summary(snaps, ax1 = None):
    if ax1 is None:
        ax1 = plt.gca()
    else:
        plt.sca(ax1)
    h=snaps[0]
    theoretical_cutoff =  np.array([0.0,0.03,0.05,0.138]) #* h.size
    lst = []
    for snap in snaps:
#         h, attr,_,_ = snap
        h = snap
        attr = h.out[0]
        uniq, count = np.unique(canonlise(attr),axis = 0,return_counts=1)
    #     spdist.cdist(h.target,metric='hamm')
        isAcc = h.nearest_target(uniq,axis =0 ) == 0
        isMem = h.nearest_target(attr, axis =1 ) == 0
        rateConv = h.isConverged.mean()
        isStable = (h.target==binarise(h.feedforward(h.target))).all(axis = 1)
        lst += [[ len(h.target),len(uniq),isAcc.sum(),rateConv,isStable.mean(),
                isMem.mean()]]
    xs,ys,nAcc,rateConv,isStable,isMem = np.array(lst).T
    xs = xs.astype('float')/h.size
#     ls = []
    plt.plot(xs,ys,'bx-',label = 'Sampled attractors')
#     plt.plot([0,25],[0,25],'b--',label = 'y=x')
    plt.ylabel('Number of distinct sampled attractors')
    plt.xlabel('Loading function (\#Stored patterns / I)')
    plt.title('I = %d, N = %d , K=%d' % (h.size, h.isConverged.size,1))
    plt.legend(loc = 'right',bbox_to_anchor = [1.6,.5])
    plt.twinx()
    plt.ylabel('Fraction of \#stored patterns')
    # plt.sca(axb)
#     plt.plot(xs,xs/ys,'x-')
    plt.plot(xs,isMem,'x-',label = 'CRR')
    plt.plot(xs,nAcc/(h.size * xs.astype(float)),'r.-',label = 'Fraction of sampled memories')
    plt.plot(xs,isStable,'yo-',label = 'Fraction of stable memories')
    plt.vlines(theoretical_cutoff,0,1,linestyles='--',label = 'theoretical switch points')
    plt.legend(loc = 'right',bbox_to_anchor = [1.7,1])
    plt.sca(ax1)
    plt.grid()
    
def plot_CRR(MEAN,STD,xs=None,ys=None,truncate=0,hmap = 1,**kwargs):
    if xs is None:
        xs = np.arange(MEAN.shape[1])
        dx = 1
    else:
        xs = np.array(xs)
        dx = xs[1] - xs[0]
    if ys is None:
        ys = np.arange(MEAN.shape[0])
        dy = 1
    else:
        ys = np.array(ys)
        dy = ys[1] - ys[0]
    fig,axs= plt.subplots(1,2,figsize=[12,4])
    if hmap:
        xedge = np.hstack([xs - dx/2.,xs[-1]+dx/2.])
        yedge = np.hstack([ys - dy/2.,ys[-1]+dy/2.])
        plt.sca(axs[1])
        plt.pcolormesh( xedge, yedge,MEAN,**kwargs)
        plt.colorbar()
    plt.sca(axs[0])
    cmap = plt.get_cmap('Set1')
    for i,(m,s,size) in enumerate(zip(MEAN,STD,ys)[truncate:]):
        col =cmap(i)
        lab = 'I=%d'%size
        plt.plot(xs,m,label =lab,c = col)
        plt.plot(xs,m+2*s,'--',c=col,alpha=.5)
        plt.plot(xs,m-2*s,'--',c=col,alpha=.5)
    plt.legend()
    # plt.plot(zs.T)
    plt.ylabel('Probability of correct retrieval')
#     plt.show()
    plt.grid()
    return fig,axs


def set_canvas():
    plt.rcParams['figure.figsize'] = plt.gcf().get_size_inches()
def mpl_header():
    import prettytable as pt
    import matplotlib.pyplot as plt
#     %matplotlib inline

    from IPython.display import set_matplotlib_formats
    set_matplotlib_formats('pdf', 'png')
    plt.rcParams['savefig.dpi'] = 75

    plt.rcParams['figure.autolayout'] = False
#     plt.rcParams['figure.autolayout'] = True
#     plt.rcParams['figure.figsize'] = plt.gcf().get_size_inches() or 10, 6
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['font.size'] = 16
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['lines.markersize'] = 8
    plt.rcParams['legend.fontsize'] = 14

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = "serif"
    plt.rcParams['font.serif'] = "cm"
    plt.rcParams['text.latex.preamble'] = "\usepackage{subdepth}, \usepackage{type1cm}"

    
def merge_data(d1,d2):
#     d = d1['data']
    d1['data'] = d1['data'] + d2['data']
#     np.concatenate( (d,d2['data']), axis = 1)
    print d1['marginal']['ys'],d2['marginal']['ys']
    d1['marginal']['ys'] = list(d1['marginal']['ys'])
    d1['marginal']['ys'][1] = d1['marginal']['ys'][1].tolist() + d2['marginal']['ys'][1].tolist()
#     d = d.extend(d2['data'].tolist())
    return d1

def load_data(fnames):

    DATA = [np.load(fname).tolist() for fname in fnames]
    data = reduce(merge_data,DATA)
    return data

def unpack_data(data, guess = None):
    ys,out = zip(*data.get('data'))
#     zname = '%s' % dict_callback.get(data.get('meta').get('sampler').get('callback'),'callback')
    msam = data['meta']['sampler']
    tl = '\#Network=%d, H=%d'%(msam['nRepeat'],msam['nSample'])
    zname = msam.get('callback')
    try:
        marg = data.get('marginal')
    #         xname = 
        xname,xs = marg['xs']
        yname,ys = marg['ys']
    except:
        xs = None;ys = None
    out = np.array(out)

    MEAN = np.mean(out,axis = 2)
    STD = np.stderr(out,axis = 2)
    pdata = {'xs':xs,'xname':xname,
            'ys':ys,'yname':yname,
            'zname': zname,
             'title':tl,
           'MEAN':MEAN,'STD':STD}
    if guess:
        pdata = guess_name(pdata,guess)
#     return pdata,data
    return pdata

def guess_name(d,refdict = namedict):
    for k,v in d.items():
        if k.endswith('name'):
            propose = refdict.get(v,None)
            d[k] = propose or v
    return d

# from vis_util import *
def eqn_lm(fit):
    return 'y=%.3fx+%.3f'%(tuple(fit.tolist()))

def plot_CRR(MEAN,STD,xs=None,ys=None,
             xname='xname',yname='yname',zname='zname',hmap = 1,truncate=0,
             title = None,
             ax = None,**kwargs):
    if ax is not None:
        hmap = 0
        axs = [ax]
        fig = plt.gcf()
    else:
        fig,axs= plt.subplots(1,2,figsize=[12,4])
    if xs is None:
        xs = np.arange(MEAN.shape[1])
        dx = 1
    else:
        xs = np.array(xs)
        dx = xs[1] - xs[0]
    if ys is None:
        ys = np.arange(MEAN.shape[0])
        dy = 1
    else:
        ys = np.array(ys)
        dy = ys[1] - ys[0]
    if hmap:
        xedge = np.hstack([xs - dx/2.,xs[-1]+dx/2.])
        yedge = np.hstack([ys - dy/2.,ys[-1]+dy/2.])
        plt.sca(axs[1])
        plt.pcolormesh( xedge, yedge,MEAN,**kwargs)
        plt.colorbar()
    plt.sca(axs[0])
    cmap = plt.get_cmap('Set1')
    for i,(m,s,size) in enumerate(zip(MEAN,STD,ys)[truncate:]):
        col =cmap(i)
        lab = '%.2f' % round(size,2)
        plt.plot(xs,m,label = lab,c = col)
        plt.plot(xs,m+2*s,'--',c=col,alpha=.5)
        plt.plot(xs,m-2*s,'--',c=col,alpha=.5)
#     plt.legend(title = 'test')
    # plt.plot(zs.T)
    plt.xlabel(xname);plt.ylabel(zname)
    plt.title(title)
#     plt.ylabel('Probability of correct retrieval')
#     plt.show()
    plt.grid()
    
    ls= axs[0].lines[::3]
    fig.legend(ls, np.round(ys,2),
               title=yname,
               loc="center right",   )
    plt.subplots_adjust(right=0.85)
    return fig,axs







def wrap_env(s,env=None):
    if env is None:
        return s
    if len(env)==0:
        return s
    return '\\begin{{{env}}} \n {s} \n \\end{{{env}}}'.format(s=s,env=env)

def wrap_math(s):
    return '$%s$'%s

def wrap_table(tab,caption = '',pos = 'h'):
    fmt='''\\begin{{table}}[{pos}]
    {tab}
    \\caption{{ {cap} }}
    \\end{{table}}
    '''
    s = fmt.format(pos=pos,cap = caption,tab = tab)
    return s


import prettytable as pt
from ptable import PrettyTable
def latex_table_tabular(self,hline = '\\hline',env = 'center'):
    latex = ["\\begin{tabular}"]
    latex.append("{"+"|".join((["l"]*len(self[0])))+"}\n")
    latex.extend([hline]*2)
    latex.append('\n')
    for row in self:
        latex.append(" & ".join(map(format, row)))
        latex.append("\\\\ %s \n"%hline)
    latex.append(hline)
    latex.append("\\end{tabular}")
    s = ''.join(latex)
    s = wrap_env(s,env)
    return  s
PrettyTable.latex_table_tabular = latex_table_tabular

data = np.array([[1,2,3],[2,3,4]])
# ?pt.PrettyTable

header = [r"$\frac{a}{b}$", r"$b$", r"$c$"]

tb = PrettyTable(data,header )

# outbuff = open('q1-param.tex','w')
#with open('q1-param.tex','w') as out:
#     print >>out,tb.latex_table_tabular()
#    s = tb.latex_table_tabular()
#    s = wrap_table(s,caption = '\\label{tab:q1-param} Initial parameters for HH model')
#    print >>out,s

    
from matplotlib import pyplot as plt
import numpy as np

def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    
    source: Taken from https://stackoverflow.com/questions/34017866/arrow-on-a-line-plot-with-matplotlib
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )


if __name__=='__main__':
    t = np.linspace(-2, 2, 100)
    y = np.sin(t)
    # return the handle of the line
    line = plt.plot(t, y)[0]

    add_arrow(line,size = 50)

    plt.show()

def phase_plot(sol_lst):
    '''
    Plot a list of T by 2+ matrix on a 2D phase-space
    Taking the first two varaible by default if phase dimension > 2D
    '''
    cmap = plt.get_cmap('Set1')
    for i,(tl,sol) in enumerate(sol_lst):
        col = cmap(i)
        L = len(sol)
        line = plt.plot(sol[:,0],sol[:,1],
                        c = col,
                        label = tl
                       )[0]
        add_arrow(line,position = 30,size = 50)
    #     add_arrow(line,position = sol[-2,0],size = 50)
        plt.plot(sol[0,0],sol[0,1],'o',c = col)
        plt.plot(sol[-1,0],sol[-1,1],'x',c=col)
        x,y = sol.T
    plt.xlabel('$v_E$(Hz)')
    plt.ylabel('$v_I$(Hz)')
    plt.legend()
    plt.grid()    

def dmet_2d(f,Zs= None,asp=1.,bins = 10,span=[-2,2],N = 1000, check_ph = 1,levels = None,log = 0,silent = 0,**kwargs):
    '''
    Plot a real-valued function on a 2D plane
    '''
    origin = 'lower'
    Nx = int(np.sqrt(N))
    Ny = Nx
    spany = [x*asp for x in span]
    X = np.linspace(*(span+[Nx]))
    Y = np.linspace(*(spany+[Ny]))
    Xs,Ys = np.meshgrid(X,Y)
    if Zs is None:
        f = np.vectorize_lazy(f)
    #     print Xs.shape
    #     Zs = map(f,zip(Xs,Ys))
        Zs = f(Xs,Ys)
    if check_ph:
#         print len(Zs)
#         print Zs[0]
#         return Zs
        Zs = map_ph(Zs)
#         print Zs[0]
        Zs = np.array(Zs)
    if log:
        Zs = np.log(1+Zs)
    if silent:
        return Zs,Xs,Ys
#     print len(Zs)
#     print Zs[0]
#     print len(X)
    imin = np.argmin(Zs)
    plt.plot(Xs.flat[imin],Ys.flat[imin],'rx')
    CS = plt.contourf(X,Y,Zs,levels = levels,**kwargs)
    CS2 = plt.contour(CS, levels=CS.levels[::2],
                  colors='r',
#                   origin=origin
                     )
#     Cs = plt.contour(X,Y,Zs)
    plt.clabel(CS2, inline=1, fontsize=10)
    plt.grid()
    return Zs,Xs,Ys
#     return plt.gcf()

def preview(f,xs = None,rg=[0,1],**kwargs):
    '''
    Preview a 1D-function f(x) on an interval "rg"
    '''
    if xs is None:
        xs = np.linspace(*rg,num=100)
    ys = f(xs)
    plt.plot(xs,ys,**kwargs)
    plt.grid()



