TF_SEED = 40

#### Adapted from: http://kyleclo.github.io/maximum-likelihood-in-tensorflow-pt-1/

import numpy as np
import pymisca.util as pyutil; reload(pyutil)
import pymisca.vis_util as pyvis; reload(pyvis)
# %matplotlib inline 
NCORE = int(pyutil.os.environ.get("NCORE","2"))
print ("NCORE=",NCORE)
def makeData():
    D = 2

    TRUE_MU = 0.0
    TRUE_SIGMA = 2.0
    SAMPLE_SIZE = (5000,D)

    np.random.seed(0)
    x_obs = np.random.normal(loc=TRUE_MU, scale=TRUE_SIGMA, size=SAMPLE_SIZE)
    rsq = np.square(x_obs).sum(axis=-1)

    keep = rsq > np.mean(rsq)
    x_obs = x_obs[keep]

    #### Random Linear transformation
    # C = pyutil.random_covmat()
    # x_obs = x_obs.dot(C)

    #### Make an ellipse
    x_obs.T[0] *= 0.5
    # x_obs = np.square(x_obs)
    
    return x_obs

x_obs = makeData()
pyvis.qc_2var(x_obs.T[0],x_obs.T[1])

import pymisca.models as pymod
m = pymod.GammaRadialTheta_VIMAP(D=2,K = 12,name= 'test6',NCORE=NCORE)

hist_loss = m.fit(x_obs,
         n_iter = 1500, 
)


sess = m.sess
clu = m.predict(x_obs)
w = m.weights_
pyvis.plt.plot(hist_loss)