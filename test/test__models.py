
def main(self):
    import pymisca.models as pymod

#     import pymisca.vis_util as pyvis

    m = pymod.Multinomial(3,[0.5,0.5])
    f = m.pdf
    assert np.allclose(f([[3,0],[2,1],[1,2],[0,3]]) , np.array([0.125, 0.375, 0.375, 0.125]))
    
    if 0:
        import pymisca.vis_util as pyvis
        #     %matplotlib inline
        pyvis.dmet_2d(pyvis.arrayFunc2mgridFunc(f),vectorised=True);
        m = Binomial(10,0.5)
        pyvis.preview(functools.partial(m.predict_proba,norm=0,log=0),rg=[0,m.N])
        m = Binomial(10.,0.5,asInt=1)
        pyvis.preview(functools.partial(m.predict_proba,norm=0,log=0),rg=[0,m.N])

        
# reload(pymod)
    m = pymod.NormedMultinomial(10,[0.3,0.7])
    N = 5000
    D = 2

    xs = np.hstack( [np.zeros((N,1)),np.random.random(size=(N,D-1)),np.ones((N,1))])
    xs  = np.diff(xs,axis=1)
    # xs = np.random.random(size=(N,D))
    # proj = lambda X : X/X.sum(axis=1,keepdims=1) 
    # xs = proj(xs)
    intl = m.pdf(xs).mean() 
    print (intl       )
    assert abs(intl - 1.0)<0.1