#### Functions
#### Potential:http://www.sfu.ca/~ssurjano/optimization.html
import numpy as np
from util import *
from fop import *
from proba import *


from gill import * #### Gillespie Algortihm/PoissonEvent


def ackley(x1,x2):
    '''
    Source: https://gist.github.com/kazetof/de206f55d8a8cacb8600ed355c6ec057
    '''
    a = 20
    b = 0.2
    c = 2*np.pi
    
    sum1 = x1**2 + x2**2 
    sum2 = np.cos(c*x1) + np.cos(c*x2)
    
    term1 = - a * np.exp(-b * ((1/2.) * sum1**(0.5)))
    term2 = - np.exp((1/2.)*sum2)
    return term1 + term2 + a + np.exp(1)

def rosenbrock(*p):
    '''Rosenbrock function
    '''
    x,y=p
    z=(1-x)**2+100*(y-x**2)**2
#     points.append((x,y,z))
    return z
def saddle_classical(x,y):
    '''
    https://arxiv.org/pdf/1406.2572.pdf
    '''
    return 5*x**2 - y**2
def saddle_monkey(x,y):
    '''
    https://arxiv.org/pdf/1406.2572.pdf
    '''
    return x**3-3*x*y**2

f = lambda x,y: -np.exp(-( (x+1)**2 + (y+1)**2 ))
g = lambda x,y: -np.exp(-((x-1)**2 + (y-1)**2))
l1_2d = lambda x,y:abs(x) + abs(y)
gauss_well = addF(f,g) 
def gauss_well(x,y):
    out = -np.exp(-( (x+1)**2 + (y+1)**2 )) -np.exp(-((x-1)**2 + (y-1)**2))
    return out
# dmet_2d()
def forward(x,T,adv=None,D=None,gradF = None,f=None,silent=0):
    '''Iteratively calling an advancer ("adv") to work on a state vector ("x")
    State vector ("x") is of length 2*D, where first D elements are coordinates,
        and last D elements are momenta/velocities
    '''
    if callable(x):
        x = x()
    if D is None:
        D = len(x)//2
#     if gradF is None:
    record_fval = not f is None
    record_grad = not gradF is None
    X = [x]
#     f0 = h(*x[:D]) + 0.000
    if not silent:
        print 'Initial coordinate:',x[:min(D,6)]
        if record_fval:
            print 'Initial  objective:',f(*x[:D])
    data = {'coord':[],'grad':[],'fval':[]}
    for i in range(T):
        data['coord'].append(x)
        if record_fval:
            data['fval'].append(f(*x[:D]))
        if record_grad:
            gd = gradF(*x[:D])
            x = adv(x,i,gd=gd)
            data['grad'].append(gd)
        else:
            x = adv(x,i,)
        if x[-1]==np.inf:
            break
    if not silent:
        print 'Ending  coordinate:',x[:min(D,6)]
        if record_fval:
            print 'Ending  objective:',f(*x[:D])
    return data
def make_adv_unif(h,alpha=None,eta=0.0,
                  dt=0.1,D=None,gradF=None):
    '''
    Prepare a descent functional from an objective function
    Samples random gradients from independent uniform distributions, assume h(X)=h(x1,x2,...,xd),
    grad(h) = \nabla_X {h} is the gradient
    then the stochastic gradient G is constructed so that E(G)=grad(h) and Var(G_i) = 1/3*(grad(h)_i - \alpha)^2,
    where \alpha is the prior belief and can be set to E_i(|grad(h)_i|) (l1 norm of the gradient).
    
    '''
    if D is None:
        D = h.func_code.co_argcount
    if gradF is None:
        gradF = make_gradF(h)
    def adv_descent(IN,i,gd = None):
        x = IN[:D]
        v = IN[D:]
        v = np.array(v)
        if gd is None:
            gd = gradF(*x)
        if 1:
#         if i%2:            
#            vct = -gamma_vector(MEAN=gd,disper=disper,size=1,rv=rv).ravel()
            gd = gd/np.sum(np.square(gd))**0.5
            vct =  -unif_vector(MEAN=gd,
                                a = alpha,size=1).ravel()
            v = (eta * v + vct)/(1.+eta)
#         else:
            x = np.add(x,np.multiply(dt,v))    
        return np.hstack([x,v])
    return adv_descent


def make_adv_descent(h,D=None,dt=0.1,gradF = None):
    '''
    Standard gradient descent
    '''
    if D is None:
        D = h.func_code.co_argcount
    if gradF is None:
        gradF = make_gradF(h)
    def adv_descent(IN,i,gd=None):
        x = IN[:D]
        v = IN[D:]
#         lr = 1.
    #     d = 0.1
    #     dx = np.ones(np.shape(x))*0.1
        if gd is None:
            gd = gradF(*x)
        dx = gradF(*x)
        x = np.add(x,np.multiply(-1*dt,dx))
        
        return np.hstack([x,v])
    return adv_descent


def rand_point(D,rd=6):
    x = np.random.random((D*2,))
    x[:D] = x[:D]*2*rd-rd
    return x


from vis_util import *
def main(h,D=None,gradF= None,x0 = None,nStep=500,
         eta=0.05,alpha=0.2,
        dt = 0.5,adv_maker=None):
    # adv = make_surfer(h,x,D=2)
    # adv = make_adv_descent(h,D=2)
    # adv = make_adv_grandwalk(h,D=2,dt = .5)
    if D is None:
        D = h.func_code.co_argcount
    if x0 is None:
        x0 = rand_point(D)
    if gradF is None:
        gradF = make_gradF(h)    
#     print gradF(*x[:D])
    if adv_maker is None:
        adv_maker = make_adv_unif
    try:
        adv = adv_maker(h,D=D, dt=dt,
                            alpha = alpha,
                            eta=eta,
        #                         alpha=None, 
        #                         eta=.5,
                            gradF=gradF
                           )
    except:
        adv = adv_maker(h,D=D,gradF = gradF,dt=dt)
#     res = forward(x0,1500,adv=adv,D=D,gradF=gradF,f = h)
    res = forward(x0,nStep,adv=adv,D=D,gradF=gradF,f = h)


    fig,axs=plt.subplots(1,3,figsize=[12,4])
    plt.sca(axs[0])
    try:
        dmet_2d(h,span=[-4,4])
    except Exception as e:
        print '[WARN] cannot plot coutour',e
        pass
    X = res['coord']
    X = np.array(X)
    traj_2D(X,ax=plt.gca())
    # pt = X[len(X)//2]
    # add_point(pt)
    plt.plot(X[0,0],X[0,1],marker='v',markersize=10,label='Start')
    try:
        add_point(X[20])
        add_point(X[40])
    except:
        pass
    plt.plot(X[-1,0],X[-1,1],'ys',markersize=10,label='End')
    plt.legend()
    plt.title('Phase over time')
    plt.sca(axs[1])

    # plt.plot(res['grad'])
    plt.plot(res['fval'],label='objective F')
    plt.plot(l2_norm(res['grad']),label='L2(grad)')
    plt.legend()
    # plt.plot(switch)
#     plt.yscale('log')
    plt.grid()
    plt.title('Objective over time')


    plt.sca(axs[2])
    plt.plot(X[:,:D])
    plt.title('Coordinates over time')
    plt.grid()
    plt.show()
    
    return res
preview_optim = main


def add_point(pt):
    plt.plot(pt[0],pt[1],'o',markersize=10)
def traj_3D(X,ax=None):
    if ax is None:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

#     plt.sca(ax)
    ax.plot(X[:,0],X[:,1],X[:,2],'r-')
    pt = X[0]
    ax.scatter3D(pt[0],pt[1],pt[2],marker='o',s=100,c='b')
    pt = X[-1]
    ax.scatter3D(pt[0],pt[1],pt[2],marker='o',s=100,c='orange')
    # plt.plot(X[-1,0],X[-1,1],'ob',markersize=10)
    return ax    
def traj_2D(X,ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111,)

#     plt.sca(ax)
    ax.plot(X[:,0],X[:,1],'r-')
    pt = X[0]
    ax.scatter(pt[0],pt[1],marker='o',s=100,c='b')
    pt = X[-1]
    ax.scatter(pt[0],pt[1],marker='o',s=100,c='orange')
    # plt.plot(X[-1,0],X[-1,1],'ob',markersize=10)
    return ax

