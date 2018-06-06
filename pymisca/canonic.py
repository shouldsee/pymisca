import util as pyutil
import fop as pyfop
def canonic_npy(s):
    f = lambda s: s if s.endswith('.npy') else '%s.npy'%s
    res = pyfop.safeMap(f,s)        
    return res
