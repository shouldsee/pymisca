import pymisca.numpy_extra as pynp

import numpy as np
import pymisca.oop as pyop

def getTraj(
    stepFunc,
    S0,
    lossFunc = None,
    verbose=0,
    print_interval=1,
    speedTol = 1E-4,
    lossTol = 1E-3,
    maxIter =200,
    minIter = 10,
    passDict = False,
):
    '''
    main iteration loop
    '''

    hist = {}
    hist['speed'] = []
    hist['loss'] = []
    hist['xs'] = []
    S = S0
    def getSpeed(Snew,S):
        def _worker((Snew_,S_)):
            speed = np.square(Snew_ - S_).sum()
#             speed = len(S_)-np.sqrt(Snew_*S_).sum()
#             assert 0
            return speed
        
        if isinstance(S, np.ndarray):
            speed = _worker((Snew,S))
#             assert 0
        else:
            speed = np.sum(map(_worker,zip(Snew,S)))
#             assert 0
            
        return speed
        
    for i in range(maxIter):
        if not passDict:
            Snew = stepFunc(S)
        else:
            d = {'X':S,'hist':hist}
            d = stepFunc(d)
            Snew = d['X']
            hist = d['hist']

        speed = getSpeed(Snew,S)
        S = Snew 

        if lossFunc is not None:
            ll = lossFunc(S)
        else:
            ll = None
    #         ll = pynp.distance__hellinger(Y, X.dot(S))
    #         ll = loss(S)
        if i>minIter:
    #             speed= abs(ll - lst[-1])
            lossDiff =  ll - hist['loss'][-1] 
            msg = 'step\t{i}\tloss\t{ll:.3E}\tspeed={speed:.3E}\tlossDiff={lossDiff:.3E}'.format(**locals())
            if verbose>=2:
                if not i % print_interval:
                    print (msg)
#             stop = 0
            if (speed<=speedTol) and (abs(lossDiff)<= lossTol) :
                if verbose>=1:
                    print ('[STOP]Converged: %s' % msg)
                break            
                
            if i + 1== maxIter:
    #                 if verbose>=1:
                print ('[STOP]Failed to converge')
    #         lst += [ll]
    
        hist['loss'].append(ll)
        hist['speed'].append(speed)
        hist['xs'].append(S)
    res = pyop.util_obj(**dict(
        hist=hist,
        last=S,
    ))
    return res    