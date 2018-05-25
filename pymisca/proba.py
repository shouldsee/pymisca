
import scipy.stats as spstat
import scipy.special as spspec 
import numpy as np
import functools
l2_norm = lambda x,keepdims=1:np.sum(np.square(x),axis=-1,keepdims=keepdims)**0.5
# D = 3.
# E = 1.
# K = (D-1)/2.
def make_fbeta(D,Edft=1.):
    '''
    ###### Implement PDF for P(t) where 
    P(t) = ( H(t)*exp(-t^2/E) + (1-H(t)) * 1  ) * Beta(K,K,(t+1)/2) 
    K = (D-1)/2, D is the dimensionality
    H(t) is the heavy-side function
    '''
    K = (D-1)/2.
    
    G = (2**(1-2*K)*np.sqrt(np.pi)*
         spspec.gamma(K)/spspec.gamma(0.5+K)
        )
    bkk = spspec.beta(K,K)
    # C = G + 1.0
#     C = G*f11 + bkk
    def pdf(t,E=None):
        if E is None:
            E = Edft
        H = (t < 0)
        bet = np.power((1-t**2)/4,K-1) ### base beta
        p = (H * np.exp(-t**2/E) + (1-H) * 1) * bet
    #     p = (H * np.exp(-t**2/E) + (1-H) * 1) * spstat.beta.pdf((t+1)/2., K, K,)
        #### Compute hyper geometric according to energy
        f11 = spspec.hyp1f1(0.5, K+0.5,-1./E)
        C = G*f11 + bkk
        return p/C  ##### Divide by normaliser !!! VVVV (do not time it !!!)
    return pdf
def zc_beta(D):
    '''
    Zero centred beta pdf defined on [-1,1]
    '''
    K = (D-1)/2.
    bkk = spspec.beta(K,K)
    if D == 2:
        def pdf(t):
            return np.ones(np.shape(t))*0.5 
    else:
        def pdf(t,**kwargs):
            bet = np.power((1-t**2)/4,K-1) ### base beta
            p  = bet/bkk/2.
            return p
    return pdf
def interp_cdf(pdf,span=[-1,1],num=10000,nCPU = 1):
    '''
    Calculate CDF for the sum (P(X+Y<=x) )
    '''
    xs = np.linspace(*span,num=num)
#     xs = np.linspace(*rg,num=num)
#     ys = mp_map(spdf,xs)
    ys = pdf(xs)
    ys = np.cumsum(ys)*(xs[1]-xs[0])
    xs = (xs[1:]+xs[:-1])/2.
    ys = ys[:-1]
    cdf = functools.partial(np.interp,xp=xs,fp=ys)
    return cdf
def invert_interp(f):
    d = f.keywords
    g = functools.partial(np.interp,fp=d['xp'],xp=d['fp'])
    return g
def interp_ppf(pdf,span=[-1,1],num=10000,**kwargs):
    cdf = interp_cdf(pdf,span=span,num=num)
    ppf = invert_interp(cdf)
    return ppf    

def Rn_basis(D,n):
    v0 = np.zeros(D)
    v0[n] = 1.
    return v0
def mvrnorm(D,n,norm=1):
    v = spstat.multivariate_normal.rvs(mean=np.zeros(D),size=(n,))
    if norm:
        v = v/l2_norm(v)
    return v

def radial_randvector(cosineppf,n=1,D=None,v0=None,E=None):        
    if v0 is None:
        if D is None:
            raise Exception('Please specify one of D or v0')
        v0 = Rn_basis(D,0)
    else:
        D = np.size(v0)    
    if np.ndim(v0)==1:
        v0 = np.expand_dims(v0,0)        
    rvct = spstat.multivariate_normal.rvs(mean=np.zeros(D),size=(n,))
    v =  rvct - np.dot(rvct,v0.T)*v0
    vy = v/l2_norm(v)
    if E is not None:
        cos = cosineppf(np.random.random(n),E=E)[:,None]
    else:
        cos = cosineppf(np.random.random(n),)[:,None]
    sin = np.sqrt(1-cos**2)
    v = (cos*v0+sin*vy)
    return v

if __name__=='__main__':
    from pymisca.util import * 
    from pymisca.vis_util import * 
    D = 3
    n = 6000
    pdf = make_fbeta(D=D,Edft=0.2)
    ppf = interp_ppf(pdf)
    v0 = mvrnorm(D,1)
    v = radial_randvector(ppf,D=3,n=4000,v0=v0)
    plt.plot(v[:,0],v[:,1],'x')
    plt.grid()
    plt.show()

    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.subplot(projection='3d')
    ax.scatter3D(v[:,0],v[:,1],v[:,2],marker='.',s=3)
    plt.show()
    

def rv_gamma(rv=None,alpha=1.,size=1):
    '''
    Fast random variate without error checking
    Alpha is Var(X)/E^2(X)
    MEAN is fixed to 1
    '''
    if rv is None:
        rv = spstat.gamma(a=1./alpha)
    rv.dist._size = size
    val = rv.dist._rvs(1./alpha)     
    return val*alpha

def rv_normal(rv=None,size=1):
    if rv is None:
        rv = spstat.norm(1)
    rv.dist._size=size
    vals = rv.dist._rvs()
    return vals

def gamma_vector(MEAN,disper=1.,rv=None,size=1):
    '''
    disper: is Var(X)/E^2(X)
    '''
    n = np.size(MEAN)
    gammas = rv_gamma(rv=rv,alpha=disper,size=(size,n))
#     if np.ndim(gammas) > np.ndim(MEAN):
    if size > 1:
        MEAN = MEAN[None,:]
    o = np.multiply(MEAN,gammas)
    return o
if __name__=='__main__':
    alpha = 0.5
    xs = rv_gamma(alpha=alpha,size=50000) 
    # xs = rv.dist._rvs(alpha)
    print alpha
    print np.mean(xs),np.var(xs)
    print xs.shape
    
    v0 = mvrnorm(D=2,n=1)
    vals = gamma_vector(v0,size=80,disper=1.)

    plt.plot(*v0,marker='x')
    plt.plot(*vals.T,marker='x',linewidth=0)
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.grid()
    plt.show()

    
    
    
def unif_vector(MEAN,a=None,size=1):
    '''
    "a" (alpha) is the prior belief of the location of variate
    '''
    L1 = np.abs(MEAN)
    if a is None:
        a = np.mean(L1)
    D = np.size(MEAN)
    x = np.random.random((size,D,))
    b = 2 * L1 - a
    y = x * (b-a) + a
    y = (2*(MEAN>0) - 1) * y
    return y

if __name__=='__main__':
    alpha = 0.5
    v0 = mvrnorm(D=2,n=1)
    vals = unif_vector(v0,size=80,a=1.5)

    plt.plot(*v0,marker='x')
    plt.plot(*vals.T[:2],marker='x',linewidth=0)
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.grid()
    plt.show()

    ys = unif_vector(v0,size=100)
    print ys.shape
    # print ys
    plt.hist(ys)
    # plt.hist(ys.ravel())
    # plt.plot(ys.ravel())
    plt.grid()
    plt.show()

# def gauss_vector(MEAN, a=None,size=1):
#     '''
#     "a" (alpha) is the prior belief of the location of variate
#     '''
# #     L1 = np.abs(MEAN)
#     MSQ=np.square(MEAN)
#     L2 = np.sum(MSQ)
#     if a is None:
#         a = np.mean(L2)    
#     D = np.size(MEAN)
#     x = rv_normal(size=(size,D))
#     VAR = (MSQ - a)**2
#     y = x * np.sqrt(VAR) + MEAN
#     return y

def gauss_vector(MEAN, a=None,size=1):
    '''
    "a" (alpha) is the prior belief of the location of variate
    '''
#     L1 = np.abs(MEAN)
    MSQ=np.square(MEAN)
    L2 = np.sum(MSQ)
    if a is None:
        a = np.mean(L2)    
    D = np.size(MEAN)
    x = rv_normal(size=(size,D))
#     scale = np.sqrt((MSQ - a)**2)
    scale = np.abs(MSQ - a)**0.5  #### Square root seems to work better (remove square)
    y = x * scale + MEAN
    return y

if __name__=='__main__':
    alpha = 0.5
    v0 = mvrnorm(D=2,n=1)
    vals = gauss_vector(v0,size=80,a=1.5)

    plt.plot(*v0,marker='x')
    plt.plot(*vals.T[:2],marker='x',linewidth=0)
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.grid()
    plt.show()

    ys = gauss_vector(v0,size=100)
    print ys.shape
    # print ys
    plt.hist(ys)
    # plt.hist(ys.ravel())
    # plt.plot(ys.ravel())
    plt.grid()
    plt.show()

        