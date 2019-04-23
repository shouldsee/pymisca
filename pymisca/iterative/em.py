#### originally adapted from https://github.com/sseemayer/mixem
#### 
import numpy as np
import sys
pyem = sys.modules[__name__]

def logsumexp(X,axis=None,keepdims=1,log=1):
    '''
    log( 
        sum(
            exp(X)
            )
        )
'''
    xmax = np.max(X,axis=axis,keepdims=keepdims)
    y = np.exp(X-xmax) 
    S = y.sum(axis=axis,keepdims=keepdims)
    if log:
        S = np.log(S)  + xmax
    else:
        S = S*np.exp(xmax)
    return S


# main(Y,speedTol=0.)

def em(data, distributions, 
       initial_weights=None, 
       max_iterations=100, 
       tol=1e-15, 
       max_iters = None,
       min_iters = 0,
       tol_iters=10,
       sample_weights= None, 
       init_resp= None,
#        weight_callback = callback__weight__entropise,
       fix_weights = None,
       progress_callback = None,
       update_proba = 1.0,
#        callback = None,
      ):
    """Fit a mixture of probability distributions using the Expectation-Maximization (EM) algorithm.

    :param data: The data to fit the distributions for. Can be an array-like or a :class:`numpy.ndarray`
    :type data: numpy.ndarray

    :param distributions: The list of distributions to fit to the data.
    :type distributions: list of :class:`mixem.distribution.Distribution`

    :param initial_weights:  Inital weights for the distributions. Must be the same size as distributions. If None, will use uniform initial weights for all distributions.
    :type initial_weights: numpy.ndarray

    :param max_iterations:  The maximum number of iterations to compute for.
    :type max_iterations: int

    :param tol: The minimum relative increase in log-likelihood after tol_iters iterations
    :type tol: float

    :param tol_iters: The number of iterations to go back in comparing log-likelihood change
    :type tol_iters: int

    :param progress_callback: A function to call to report progress after every iteration.
    :type progress_callback: function or None

    :rtype: tuple (weights, distributitons, log_likelihood)
    """
#     progress_callback = callback
    
    if max_iters is not None:
        max_iterations = max_iters
    n_distr = len(distributions)
    n_data = data.shape[0]

    if initial_weights is not None:
        weight = np.array(initial_weights)
    else:
        weight = np.ones((n_distr,))
    if sample_weights is not None:
        sample_weights = sample_weights / sample_weights.mean()

    last_ll = np.zeros((tol_iters, ))
    log_density = np.empty((n_data, n_distr))
    log_likelihood = log_density.copy()
    def _m_step(wresp):
        # M-step #######
        for d in range(n_distr):
#             if d in idx_active:
            if idx_active[d]:
                if np.random.random() <= update_proba:
                    distributions[d].estimate_parameters(data, wresp[:, d])    
            
    def weight_resp(resp):
        if sample_weights is not None:
            wresp = resp * sample_weights[:,None]
            pass
        else:
            wresp = resp
        return wresp
            
    iteration = 0
    idx_active = range( n_distr )
    if init_resp is None:
        resp = np.empty((n_data, n_distr))
    else:
        resp = init_resp
        _m_step(resp)
        
    wresp  = weight_resp(resp)
    
    while True:
        
        # E-step #######
        # compute responsibilities
        for d in range(n_distr):
            log_density[:, d] = distributions[d].log_density(data)
            #######
#         log_density = np.nan_to_num(log_density)
        
        
        # normalize responsibilities of distributions so they sum up to one for example
#         eps = 1E-10

######
#         weight = np.nan_to_num(weight)
        idx_active = weight != 0.
        
        log_proba = logProba = np.log(weight[None,: ]) + log_density
        logResp = logProba - logsumexp(logProba,axis=1)        
        resp  = np.exp(logResp)
#         log_proba = logProba= 

#         proba = np.log( np.exp(log_density) * weight[None,:] )
#         log_proba = np.log(proba)
#         SUM =  np.sum(proba,axis=1,keepdims=1)
#         SUM[SUM==0.] = 1.
#         resp  = proba /SUM
        
        
        sumWeight = resp.sum(axis=0)
        idx_active = idx_active & (sumWeight != 0.)
        resp[:, ~ idx_active ] = 0.
        wresp = weight_resp(resp)
        ######
#         wresp = np.nan_to_num(wresp)



#         .nonzero()[0]
#         assert (resp.sum(axis=0) != 0).all()
#         resp = weight[np.newaxis, :] * np.exp(log_density)
#         resp /= np.sum(resp, axis=1)[:,  np.newaxis]

        log_likelihood[:,idx_active] = (( wresp[:,idx_active] * log_density[:,idx_active]))

        
        if not fix_weights:
            weight = np.mean(wresp, axis=0)
#             if weight_callback is not None:
#                 weight = weight_callback(weight)
#             weight = weight * 0. + 1.

        callback_args = iteration, weight, distributions, log_likelihood, log_proba,wresp
    
        if progress_callback is not None:
            callback_args = progress_callback(*callback_args)
            
        iteration, weight, distributions, log_likelihood, log_proba, wresp = callback_args
        
        _m_step(wresp)
            
#             if res is not None:
#                 iteration = res[0]
                
        
        #### Convergence check #######
        ll = np.sum(log_likelihood)
#         log_likelihood = np.sum(log_likelihood)
        if np.isnan(ll):
            iteration = -1

        if iteration >= max(min_iters,tol_iters) \
        and (last_ll[-1] - ll) / last_ll[-1] <= tol:
            iteration = -1

        if iteration >= max_iterations:
            iteration = -1
            
        if iteration < 0:
            last_ll[0] = ll
            callback_args = progress_callback(*
                callback_args
            )
            break 
            
        
        # store value of current iteration in last_ll[0]
        # and shift older values to the right
        last_ll[1:] = last_ll[:-1]
        last_ll[0] = ll

        iteration += 1
    ### Return full history for debugging
    return weight, distributions, last_ll[::-1]
