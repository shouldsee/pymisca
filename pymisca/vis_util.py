from util import *
# import pymisca.util as pyutil
import numpy as np
import sys
pyvis = sys.modules[__name__]
try: 
	import scipy
	import  scipy.cluster.hierarchy as sphclu
except:
	print ('scipy not installed')

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

try: 
    import plotly.offline as opy
    import plotly.graph_objs as go
except Exception as e:
    print ('[WARN]:%s')
    
import matplotlib.ticker as mticker
def hide_axis(ax,which='both'):
    if which in ['x','both']:
#         ax.get_xaxis().set_visible(False)
        ax.xaxis.set_major_formatter(mticker.NullFormatter())
    if which in ['y','both']:
        ax.yaxis.set_major_formatter(mticker.NullFormatter())
#         ax.get_yaxis().set_visible(False)    
    return ax

def hide_ticks(ax,which='both'):
    if which in ['x','both']:
        ax.set_xticks([])
    if which in ['y','both']:
        ax.set_yticks([])
    return ax


def hide_frame(ax):
    for spine in ax.spines.values():
        spine.set_visible(False) 
    return ax

def hide_Axes(ax,which='both',alpha=0.0):
    hide_axis(ax)
    hide_frame(ax)
    hide_ticks(ax)
    ax.patch.set_alpha(alpha)
    return ax

def line__getLegend(line):
    res = (line,line.get_label())
    return res
getLegend = line__getLegend
def line__getLegends(lines):
    res = zip(*map(getLegend,lines))
    return res
getLegends = line__getLegends


def make_subplots(
    L,
    ncols = 4,
    baseRowSep = 4.,
    baseColSep = 4.,
    gridspec_kw={'hspace':0.45},
    **kwargs
):
    '''
    Create a grid of subplots
'''
    nrows = L//4+1
    fig,axs = plt.subplots(ncols=ncols,nrows=nrows,
                            figsize=[ncols*baseColSep, 
                                     nrows*baseRowSep],
                           gridspec_kw=gridspec_kw,                           
                           **kwargs); 
    axs = np.ravel(axs)
    return fig,axs
get_subplotGrid = make_subplots


def plotArrow(ax):
    hide_Axes(ax)
    ax.arrow(0.,0.5,1.,0.,
             width = 0.15
    #         head_width = 0.5,
            )
    ax.set_xlim(0,2.)
    ax.set_ylim(0,1.,)
    return ax

#### mpatches
import matplotlib.patches as mpatches
def legend4Patch(lst,as_patch=0):
    '''
    Usage:
        leg = legend4Patch(
            (
                ['blue','Short Day (12:12)'],
                ['red','Long Day (20:4)'],
                ['black','Short Day Average']
            )
        )
        plt.gca().legend(*leg)
        
'''
    out = []
    for color,label in lst:
        d = {'color':color,'label':label}
        res = mpatches.Patch(**d)
        out += [res]
    patches = out
    leg = getLegends(patches)
    if as_patch:
        res = patches
    else:
        res = leg
    return res
def test__legend4Patch():
    leg =out = legend4Patch(
        (
            ['blue','Short Day (12:12)'],
            ['red','Long Day (20:4)'],
            ['black','Short Day Average']
        )
    )
    plt.gca().legend(*leg)

def add_harrow(x_tail,x_head,mutation_scale=100,ycent=0.,
               head_length = None, height = 0.5, head_width = None, text=None,
               length_includes_head=True,
               ax=None,**kwargs):
    y_tail = y_head = ycent
#     height = .5
    tail_width = height /2.
    if head_length is None:
        head_length = abs(x_head - x_tail)/5.
    if head_width is None:
        head_width = tail_width
    if ax is None:
        ax = plt.gca()
    dx = x_head - x_tail
    dy = y_head - y_tail
        
#     print dx
#     print x_tail,x_head
    if 0:
        arrowstyle = mpatches.ArrowStyle('Simple',
                                         head_length= head_length, 
                                         head_width = head_width,
                                         tail_width = tail_width)
        arrow = mpatches.FancyArrowPatch( (x_tail, y_tail), (x_head, y_head),
                                         mutation_scale=mutation_scale,arrowstyle=arrowstyle,
                                         shrinkB =0. ,shrinkA=0., 
                                         **kwargs)

#     kwargs.update(arrowStyle)
#     arrow = mpatches.Arrow( (x_tail, y_tail), (x_head, y_head),
#                                      mutation_scale=mutation_scale,arrowstyle=arrowstyle,
#                                      shrinkB =0. ,shrinkA=0., 
#                                      **kwargs)    
    arrow = mpatches.FancyArrow(x_tail, y_tail, dx, dy,
                               head_length= head_length, 
                                head_width = head_width,
                                length_includes_head=length_includes_head,
                                width = tail_width,**kwargs
#                                 tail_width = tail_width
                               )
    ax.add_patch(arrow)
    if text is not None:
        ax.text(x_tail, height * 0.75 ,text)
    return arrow,ax

def add_hbox(xleft,xright,head_length = 0.00001,
             **kwargs):
    return add_harrow(xleft, xright, head_length = head_length,  **kwargs)
    
if __name__ == '__main__':
    pass

#######

# pyvis.getLegends = getLegends

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
        imc = im[ridx][-truncate:].T
    else:
        imc = im[ridx].T
    plt.pcolormesh(imc, antialiased = False)
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









# import prettytable as pt
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

    
import matplotlib as mpl
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

def add_text(xs,ys,labs,ax= None,checkNA =1, **kwargs):
    '''Vectorised text annotation
'''
    if ax is None:
        ax = plt.gca()
    if checkNA:
        xs,ys,labs = pd.concat([xs,ys,labs],axis=1,join='inner').dropna().values.T
    for xx,yy,tt in zip(xs,ys,labs):
        ax.text(xx,yy,tt,**kwargs)
    return ax    
    


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

def dmet_2d(f,Zs= None,asp=1.,bins = 10,xlim = None,ylim = None,
            span=[-2,2],N = 1000,
            check_ph = 1,levels = None,log = 0,silent = 0,
            vectorised=False,
            ax= None,**kwargs):
    '''
    Plot a real-valued function on a 2D plane
    '''
    origin = 'lower'
    Nx = int(np.sqrt(N))
    Ny = Nx
    spany = [x*asp for x in span]
    if xlim is None:
        xlim = span
    if ylim is None:
        ylim = spany
        
    X = np.linspace(*xlim,num = Nx)
    Y = np.linspace(*ylim,num = Ny)
    Xs,Ys = np.meshgrid(X,Y)
    if Zs is None:
        if not vectorised:
            f = np.vectorize_lazy(f)
            vectorised = True
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
    if ax is None:
        fig,axs = plt.subplots(1,2,figsize=[12,6])
        ax  = axs[0]
    plt.sca(ax)
    plt.plot(Xs.flat[imin],Ys.flat[imin],'rx')
    CS = plt.contourf(X,Y,Zs,levels = levels,**kwargs)
    CS2 = plt.contour(CS, levels=CS.levels[::2],
                  colors='r',
#                   origin=origin
                     )
#     Cs = plt.contour(X,Y,Zs)
    plt.clabel(CS2, inline=1, fontsize=10)
    plt.grid()
    return (Zs,Xs,Ys),(ax,CS)
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

    
def square_shape(im):
    '''Reshape an array to R*R
    '''
    im = np.squeeze(im)
    SHAPE = im.shape[0]
    if np.ndim(im)==1:
        im = np.reshape(im,(int(SHAPE**0.5),)*2 )
    else:
        pass
#         im = np.squeeze(im)    
    return im
def show_pics(IN):
    '''Display input as a list of square images
    '''
    SHAPE = IN.shape[1]
    
    fig,axs = plt.subplots(2,5,figsize=[12,4])
    axs = axs.flat
    for i in range(min(10,len(IN))):
    #     plt.axs
        ax=axs[i]
        im = IN[i]
        im=square_shape(im)
        ax.imshow(im,)

def ylim_fromZero(ax):
    '''Assuming positive range 
    '''
    ax.set_ylim(bottom = 0,top = ax.get_ylim()[1]*1.1)
    return ax

def histoLine(xs,bins=None,log= 0, ax = None, xlim =None, transpose= 0, normed=1, **kwargs):
    ''' Estimate density by binning and plot count as line.
'''
    if ax is None:
        ax = plt.gca()
    xlim = pyutil.span(xs,99.9) if xlim is None else xlim
    bins = np.linspace(*xlim,
                      num=100) if bins is None else bins
#     ys,edg = np.histogram(xs,bins,normed=normed)
    ys,edg = np.histogram(xs,bins,density=normed)
    ct = (edg[1:] + edg[:-1])/2
    if log:
        ys = np.log1p(ys)
    else:
        pass
    if transpose:
        ct,ys = ys,ct
        ax.set_xlabel('count')
    else:
        ax.set_ylabel('count')
    l = ax.plot(ct,ys,**kwargs)
    return ax


def cmap4vlim(vlim,debug=0):
    if debug:
        print ('[dbg]Setting cmap ... ')
    if vlim[0] * vlim[1]>=0:
        name = 'viridis'
    else:
        avg = abs( vlim[1] - vlim[0] )/2.
        vlim = -avg,avg
        name = 'PiYG'
    cmap = plt.get_cmap('PiYG')
    if debug:
        print ('[dbg]...to %s' %name  )
    cmap.set_bad('black',1.)
    return cmap

####### Clustering
def heatmap(C,
            ax=None,
            xlab = '',
            ylab = '',
            main = '',
            xtick = None,
            ytick = None,
            transpose=0,
            cname=None,
            tickMax=100,
            vlim = None,
            cmap = None,
            figsize=None,
            marginInSquare=(2,2),
            squareSize = (0.2,0.2),
            interpolation = 'none',
#             antialiased=False,
            
            **kwargs
           ):
    ''' C of shape (xLen,yLen)
    '''
#     print kwargs.keys()
    if transpose:
        C = C.T
#         xtick,ytick = ytick,xtick
#         xlab,ylab   = ylab,xlab
    if ax is None:
        ##### Create default axis
        shape = C.T.shape  #### transpose is crucial due to pcolormesh

        ## Old
#         if figsize is None:
#             figsize = [min(len(C.T)/3.,14),
#                        min(len(C)/5.,14)]
#         fig,ax = plt.subplots(1,1,figsize=figsize)
    
        ## New 
        fig,ax = getMarginedAxis(shape,
            squareSize = squareSize,
            marginInSquare= marginInSquare,figsize=figsize)
        

    if vlim is not None:
        if any([x is None for x in vlim]):
            vlim = None
    if vlim is None:
        vlim = np.span(C[~np.isnan(C)],99)
    elif vlim[0] is None:
        pass
#     else:
    if cmap is None:
        cmap = cmap4vlim(vlim)
        
    C = np.ma.array (C, mask=np.isnan(C))
    if cmap is not None:
        cmap.set_bad('black',1.)
    kwargs['vmin'],kwargs['vmax'] = vlim

    plt.sca(ax)
    im = ax.matshow(C,aspect='auto', cmap = cmap,
#                     antialiased=antialiased,
                    interpolation = interpolation,
                    **kwargs)
    ax.xaxis.tick_bottom()

    if xtick is not None:
        if len(xtick) <= tickMax:
            plt.xticks(range(len(C.T)), xtick,
                      rotation='vertical')
    
    if ytick is not None:
        if len(ytick) <= tickMax:
            plt.yticks(range(len(C)),ytick)

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if cname is not None:
        cbar = plt.colorbar(im)
        cbar.set_label(cname)
    plt.title(main)    
    return im

def linePlot4DF(df,
                xs=None,
                y2 = None,
                label = None,
                ax=None,
             rowName=None,
             colName= None,ylab = '$y$',
                which = 'plot',
#                 cmap=None,
                xlab='$x$',
                xrotation= 'vertical',
                xshift=None,
                cmap = 'Set1',
                **kwargs):
    rowName = df.index if rowName is None else rowName
    colName = df.columns if colName is None else colName
    C = df.values
    if isinstance(y2,pd.DataFrame):
        y2 = y2.values    
    
    if ax is None:
        fig,axs= plt.subplots(1,2,figsize=[14,4])
        ax = axs[0]
    plt.sca(ax)    
    cmap = plt.get_cmap(cmap)
    
    if xs is None:
        xs = np.arange(len(C[0]))
    
    if which =='StemWithLine':
        plotter = pyutil.functools.partial(StemWithLine,
                                           ax=ax)
    elif hasattr(ax, which):        
        plotter =  getattr(ax, which)
#     plotter = pyutil.functools.partial(plotter,)
    
    handles = []
    for i,ys in enumerate(C):
        if xshift is not None:
            kwargs['xshift'] = xshift *i
#         if which=='fill_between'
        if which=='fill_between':
            assert y2 is not None,'y2 must be specified for fill between'
            h = plotter(xs,y1=ys, y2=y2[i], color=cmap(i),**kwargs)
        else:
            h = plotter(xs,ys,label = rowName[i],color=cmap(i),**kwargs)
        handles.append(h)
#         ax.plot(xs,ys,label=rowName[i])
    ax.set_ylabel(ylab)
    ax.grid(1)
    L = len(ys)
    
    xticks = [ x for x in map(int,ax.get_xticks()) if x>=0 and x<L]
#     ax.set_xticks(xticks)
#     ax.set_xticklabels(colName[xticks])
#     ax.tick_params(axis='x', rotation=xrotation)    
    plt.xticks(xticks,colName[xticks],rotation=xrotation,)
#     ax.set_xticks(xticks,)
#     ax.set_xticklabels(colName[xticks])
    ax.set_xlabel(xlab)
    return handles

def StemWithLine(xs=None,ys=None,
                 xshift=0.,
                 color = None,ax = None,
                 bottom=0,**kwargs):  
    if ax is None:
        fig,axs= plt.subplots(1,2,figsize=[14,4])
        ax = axs[0]
    if color is None:
        color = ax._get_lines.get_next_color()

    xs = np.arange(len(ys)) if xs is None else xs
    xs = np.array(xs,dtype='float')
    if xshift:
        xs += xshift
#     bottom = 0.5


    markerline,stemline,_ = ax.stem(xs,ys,linefmt='--',
                                    bottom=bottom)
    [l.set_color(color) for l in [markerline]]
    [l.set_color(color) for l in stemline]
    line = ax.plot(xs,ys,color=color,**kwargs)
    ax.grid(1)
    return line,ax
# pyvis.linePlot=linePlot

def matHist(X,idx=None,XLIM=[0,200],nbin=100):    
    plt.figure(figsize=[12,4])
    if idx is not None:
        X = X[idx]
    MIN,MAX = X.min(),np.percentile(X,99)
    BINS = np.linspace(MIN,MAX,nbin)
    for i in range(len(idx)):
        histoLine(X[i],BINS,alpha=0.4,log=1)
    plt.xlim(XLIM)
    plt.grid()

def abline(k=1,y0=0,x0 = None, color = 'b', ax = None,**kwargs):
    '''Add a reference line
    '''
    if ax is None:
        ax = plt.gca()
    MIN,MAX=ax.get_xlim()
    xs = MIN,MAX
    if x0 is not None:
#         assert k is None
        xs = (x0,x0)
        ys = ax.get_ylim()
    else:
        assert k is not None
        f = lambda x: k*x+y0
        ys =  map(f,xs)
    ax.plot( xs, ys,'--',color=color,**kwargs)    
#     print MIN,MAX
#     ax =plt.gca()
    return ax
    
def qc_2var(xs,ys,clu=None,xlab='$x$',ylab='$y$',
            markersize=None,xlim=None,ylim=None,axs = None,
           xbin = None,
           ybin =None,
            spanPer = 99.9,
            nMax=3000,
#            axis = [0,1,2]
           ):
    ''' Plot histo/scatter/density qc for two variables
'''
    if axs is None:
        fig,axs= plt.subplots(1,4,figsize=[14,4])
    axs = list(np.ravel(axs))
    axs = axs + [None] * (4-len(axs))
    xs = np.ravel(xs)
    ys = np.ravel(ys)
    
    if nMax <0 :
        nMax= len(xs)

    xlim = xlim if xlim is not None else np.span(xs,spanPer)
    ylim = ylim if ylim is not None else np.span(ys,spanPer)
    BX = np.linspace(*xlim, num=30) if xbin is None else xbin
    BY = np.linspace(*ylim, num=50) if ybin is None else ybin    
#         xlim = np.span(BX)
#         ylim = np.span(BY)
    if clu is not None:
        pass
    else:
        clu = [0]*len(xs)
    clu = np.ravel(clu)
    
    df = pd.DataFrame({'xs':xs,'ys':ys,'clu':clu})
#     nMax = 3000
    for key, dfc in df.groupby('clu'):
        if len(dfc)>nMax:
            dfcc = dfc.sample(nMax)
        else:
            dfcc = dfc
#         print k,dfc
#         xs,ys,_ = dfcc.values.T
        xs,ys = dfcc['xs'].values, dfcc['ys'].values
        xs = xs.ravel()
        ys = ys.ravel()
        
        ax = axs[0];
        if ax is not None:
            plt.sca(ax)
            histoLine  (xs,BX,alpha=0.4)    
        ax = axs[1];
        if ax is not None:
            plt.sca(ax)
            plt.scatter(xs,ys,markersize,label=key, marker='.')
        ax = axs[2];
        if ax is not None:
            plt.sca(ax)
            histoLine  (ys,BY,alpha=0.4,transpose=1)            
        ax = axs[3];
        if ax is not None:
            plt.sca(ax)
            ct,BX,BY = np.histogram2d(xs, ys,(BX,BY))
            plt.pcolormesh(BX,BY,np.log2(1+ct).T,)
    [ax.grid(1) for ax in axs[:3] if ax is not None]
    ax = axs[0];
    if ax is not None:
        plt.sca(ax)
        plt.xlabel(xlab)
        plt.xlim(xlim)
    ax = axs[2];
    if ax is not None:
        plt.sca(ax)
        plt.ylabel(ylab)
        plt.ylim(ylim)

    ax = axs[1];
    if ax is not None:
        plt.sca(ax)
        abline()
        plt.xlabel(xlab);plt.ylabel(ylab)
        plt.xlim(xlim);plt.ylim(ylim)

    ax = axs[3];
    if ax is not None:
        plt.sca(ax)
        plt.xlabel(xlab); plt.ylabel(ylab)
    return axs
# pyvis.heatmap=heatmap
import random
def discrete_cmap(N, base_cmap=None,shuffle = 0,seed  = None):
    """Create an N-bin discrete colormap from the specified input map
    Source: https://gist.github.com/jakevdp/91077b0cae40f8f8244a
    """

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    if base_cmap is None:
        base = plt.get_cmap()
    else:
        base = plt.cm.get_cmap(base_cmap)
        
    rg = np.linspace(0, 1, N+1)
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(rg)
    color_list = base(rg)
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N+1)
from mpl_extra import *

def ax_vlines(cutoff,ax = None):
    if ax is None:
        ax = plt.gca()
    lines = ax.vlines(cutoff,*ax.get_ylim())
    return lines

# from pymisca.util import qc_index

try:
    import matplotlib_venn as mvenn
except:
    print ('[IMPORT] cannot import "matplotlib_venn"')
    
def qc_index(ind1=None,ind2=None,
             subsets = None,
    xlab = 'Group A',
    ylab = 'Group B',
    silent= True,
     ax = None,
):
    '''
    compare two sets 
'''
    if subsets is None:
        ind1,ind2 = set(ind1),set(ind2)
        indAny = ind1 | ind2
        indAll = ind1 & ind2
        indnot1 = indAny - ind1
        indnot2 = indAny - ind2
        LCL = locals()
        d = pyutil.collections.OrderedDict()
    #     d = {}
        for key in ['ind1','ind2','indAll','indAny',
                   'indnot1','indnot2']:
            ind = LCL.get(key)
            print (key, len(ind))
            d[key] = ind
        print 
        df = pd.DataFrame(dict([ (k, pd.Series(list(v))) for k,v in d.items() ]))
        df['ind1=%s'%xlab]=np.nan
        df['ind2=%s'%ylab]=np.nan    
        Lnot2 = len(indnot2)
        Lnot1 = len(indnot1)
        Lall  = len(indAll)
        subsets = (Lnot2, Lnot1, Lall)
    else:
        (Lnot2, Lnot1, Lall) = subsets
        df = subsets
    ### Lall is the lenght of intersection
    if not silent:
        if ax is None:
            fig,axs = plt.subplots(1,3,figsize= [16,4])
            ax= axs[0]
        im = mvenn.venn2(subsets = subsets, 
                         set_labels = (xlab, ylab),
                         ax=ax)
        jind = Lall/float(Lnot2 + Lnot1 + Lall)
        ax.set_title('Jaccard_index=%.3f%%'%(100*jind))
    return df,ax

def qc__mapVenn(plotter,indVenn,figsize=[16,6]):
    fig,axs = plt.subplots(1,3,figsize=figsize);
    i= -1
    xlab,ylab = indVenn.columns[-2],indVenn.columns[-1]
    xlab = xlab.split('=',1)[-1]
    ylab = ylab.split('=',1)[-1]
    
    i += 1;ax=axs[i];plt.sca(ax)
    ax = plotter(indVenn.ind1,xlab,ax=ax)

    i += 1;ax=axs[i];plt.sca(ax)
    ax = plotter(indVenn.indAll, ' and '.join([xlab,ylab]),ax=ax)
    
    i += 1;ax=axs[i];plt.sca(ax)
    ax = plotter(indVenn.ind2, ylab ,ax=ax)
    return fig

if 'plotly.offline' in sys.modules:

    def plotly__scatter3D(dataDF=None,x=None,y=None,z=None,
                          text=None,color=None,index=None,
                          filename=None):
        '''
        Generate a Plotly Scatter3D object
        '''
        if dataDF is None:
            if index is None:
                index = color.index
            dataDF =  pd.DataFrame(dict(x=x,y=y,z=z,text=text.tolist(),color=color),index=index)

        colorscale = 'Rainbow'
        colorFct = dataDF.color.astype('category')
        cat = colorFct.cat
        color = cat.codes
        symbol = None
        markerSize=5
        xlim = None
        ylim = None
        zlim = None


        traces = []
        for color,(key,df) in enumerate(dataDF.groupby(colorFct)):
    #         color = vals.categories.get_loc('190RQ')
            trace = go.Scatter3d(
                name = key,
                x=df.x,
                y=df.y,
                z=df.z,
                text=df.text,
                mode='markers',
                marker=dict(
                    size=markerSize,
                    color=color,
                    symbol = symbol,
                    colorscale = colorscale,
                    opacity=0.8
                )
            )
            traces += [trace]
        layout = go.Layout(  
                showlegend = True,
            scene = dict(
            xaxis=dict(
                range=xlim,
                title='PC1',
            ),
            yaxis=dict(
                title='PC2',
                range=ylim
            ),
            zaxis=dict(
                title='PC3',
                range=zlim
            )
            ),
        )
        fig = go.Figure(data=traces, layout=layout)
        return plotly__output(fig,ofname=filename)
#         if filename is not None:
#             graphObj = opy.plot(fig, filename=filename, output_type='file')
#             return filename
#         else:
#             return fig
    def plotly__output(figure, ofname=None,auto_open=False,**kwargs):
        if ofname is not None:
            graphObj = opy.plot(figure, 
                                filename=ofname, 
                                auto_open=auto_open,
                                output_type='file')
            graphObj = ofname
        else:
            graphObj = opy.plot(figure,  
                               auto_open=auto_open,
                                output_type='div',)
        return graphObj
            
    def plotly__heatmap(z,
                        xticks=None,
                        yticks=None,
                       ofname = None,
                       title = '',):
        title = unicode(title)
    #     xticks = list(xticks) if xticks is not None else None
    #     yticks = list(yticks) if yticks is not None else None
        data = [
            go.Heatmap(
                z = z,
                x = xticks,
                y = yticks,
    #             x=date_list,
    #             y=programmers,
                colorscale='Viridis',
            )
        ]

        layout = go.Layout(
            title=title,
    #         xaxis = dict(ticks=xticks,),
    #         yaxis = dict(ticks=yticks )
        )    
        figure = go.Figure(data=data, layout=layout)
        return plotly__output(figure,ofname=ofname)
#         graph = opy.plot(figure, auto_open=False, output_type='div')
#         return graph    

    def plotly__linePlot(dfc,ofname = None, auto_open=False,
                        xticks=None,title = None,showlegend=True,
                        ):
        title = unicode(title)
    # if 1:
        tickSize=16
        if xticks is None:
            xticks = dfc.columns
        data = [
            go.Scatter(
                x=xticks,
                y=rec.values,
                name = key,) 
            for key,rec in dfc.iterrows()
        ]
        layout = go.Layout(
            title=title,
            xaxis = dict(automargin=True,
#                          tickangle=90,
                         tickfont=dict(size=tickSize,
                                      ),
                        ),
            yaxis=  dict(automargin=True,
                tickfont=dict(size=tickSize)
                ),
    #         xaxis = dict(ticks=xticks,),
    #         yaxis = dict(ticks=yticks )
          showlegend = showlegend,
        )    
        figure = go.Figure(data=data,layout=layout,
                          )
        res = plotly__output(figure=figure,
                                   auto_open = auto_open,
                                   ofname=ofname)
        return res
def getFullFigureAxis(**kwargs):
    ''' Create an axis occupying the whole figure without margin
'''
    fig, axs =plt.subplots(
        1,1,
        gridspec_kw=dict(
            left=0,right=1,
            bottom=0,top=1,
            hspace=0.,
            wspace=0.,
           ),
        **kwargs)
    ax = axs
    return fig,ax


def getMarginAxes(ax=None,shape=None,shapeInSquare=(10,10),
                      marginInSquare=(2,2),instMargin=0,
                 ):
    '''subsetting the axis to create space for margin
'''
    if ax is None:
        fig,ax = getFullFigureAxis()
    if shape is not None:
        shapeInSquare=shape
    axs=  mpl.gridspec.GridSpecFromSubplotSpec(
        2,2,ax,
        width_ratios=[marginInSquare[0],shapeInSquare[0],],
        height_ratios = [shapeInSquare[1],marginInSquare[1],],
        hspace=0.,
        wspace=0.,
                                              )
#     return axs
# if 0 :
#     axs = [ [plt.subplot(x) for x in y] for y in axs]
    if instMargin:
        axs = map(plt.subplot, axs)    
        [pyvis.hide_Axes(axs[i]) for i in [0,2,3]]
    else:
        axs = list(axs)
        axs[1] = plt.subplot(axs[1])
    
    axisMain = axs[1]
    axisLeft = axs[0]
    axisBottom = axs[3]
    return (axisMain,axisLeft,axisBottom)
def getMarginedAxis(
    squareShape,
    squareSize = (0.2,0.2),
    marginInSquare=(2,2),
    figsize=None
):
    shape = squareShape
    if figsize is None:
        figsize = np.add( marginInSquare, shape) * squareSize

    fig, ax = pyvis.getFullFigureAxis(figsize=figsize)
    ax = pyvis.getMarginAxes(ax,
                             shape=shape,
                            marginInSquare=marginInSquare,)[0]
    return fig,ax