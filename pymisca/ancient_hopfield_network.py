try:
#     pass
    import network
    namedict = {network.callback_exp_hamm:'EDNA',
                network.callback_nCorrect:'CRR',
                'sizelst':'Network size\n(I)',
                'nTarlst':'Number of stored patterns',
               'pcor':'$p_c$'}
except:
    namedict = {}
    print "[WARN] %s cannot find network" %__name__


def guess_name(d,refdict = namedict):
    for k,v in d.items():
        if k.endswith('name'):
            propose = refdict.get(v,None)
            d[k] = propose or v
    return d    
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