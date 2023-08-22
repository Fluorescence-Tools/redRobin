# functions written by Oleg Opanasyuk
# in the period jan-Feb 2023
# minor edits by Nicolaas van der Voort (eliminated magic pylab imports)

from numpy import expand_dims, array, hstack, empty, zeros, ones, arange, pi
from numpy import sqrt, mean, expm1, dtype, isscalar, full, modf, log
from numpy.lib.recfunctions import merge_arrays
#from math import modf
from numexpr import evaluate as nv
from scipy.fft import rfft, irfft, rfftfreq
from lmfit import  minimize, Minimizer, Parameters, Parameter, report_fit, fit_report
from pandas import DataFrame, read_csv
from pandas import set_option as pd_set_option
#pd_set_option('display.max_columns', None)
import time


######################################################
# Model functions

# Just for reference and debuging - model functions in a time domain
"""# arbitrary multiexponential decay in time domain
def mexp(t,k,p):
    # t  - 1d array of times (here we do not batch time - it is the same for all data/model)
    # k  - array of exponential rates
    # p  - array of probabilities of k; k and p should have the same dimentions
    # k,p are multidimentional with first index for batch
    
    d = k.ndim; dr = tuple(arange(d))  # dimentions of k,p 
    kx = expand_dims(k,d)  # e(x)panded arrays: add one index at the end for t
    px = expand_dims(p,d)  # -//-
    tx = expand_dims(t,dr) # add indexes at first positions for k,d
    
    return nv('px*exp(-kx*tx)').sum(dr[1:]) # sum all except first and last; result: 2D batch of f(t)


# VV,VH decays in time domain without repetition and convolution
def decay_dfa(t, kd, kf, ka, pdfa, r0):
    # t        - 1d time range: arange(n)
    # kd,kf,ka - 2d arrays of DO, FRET, Anisotropy rates;
    # pdfa     - 4d array of prob. of dfa states, if independent it is outer product of 1d probs.
    # r0       - 1d array (only batch)
    # for k,p  - first index = batch index
    r0pol = r0[:,None]*array([[2.0, -1.0]]) # 2d[batch, vvvh] r0 * ideal polarization factors for VV,VH;
                                            # pol factors can go to parameter (for example, pol=0 for VM)
    kdf = kd[...,None] + kf[:,None,:]       # 3d array of kd+kf rates
    f   = mexp(t,kdf,pdfa.sum(3))           # 2d[batch, t] f(t) fluorescence decays without anisotropy
    k   = kdf[...,None] + ka[:,None,None,:] # 4d array of kd+kf+ka rates
    fr  = mexp(t,k,pdfa)                    # 2d[batch, t] f(t)*r(t) fluorescence*anisotropy decays
    
    return f[:,None,:]+r0pol[...,None]*fr[:,None,:] # 3d[batch, vvvh, t]
""";

# Arbitrary multiexponential decay in frequency domain (fft)
def fmexp(o,k,p,nr):
    # see mexp comments
    # o  - fft frequency (-2.0j*pi*freq(n))
    # k  - array of exponential rates
    # p  - array of probabilities of k; k and p should have the same dimentions
    # nr - repetition period [number of dt]
    
    n  = o.size
    d  = k.ndim; dr = tuple(arange(d))  # dimentions of k,p 
    kx = expand_dims(k,d)  # e(x)panded arrays
    px = expand_dims(p,d)
    ox = expand_dims(o,dr)
    
    return nv('-px*expm1(-kx*n)/expm1(-kx*nr)/expm1(-kx+ox)').sum(dr[1:]) # f(omega)

# VV,VH decays in the fourier domain (o) with repetition (nr)
def fdecay_dfa(o, kd, kf, ka, pdfa, r0, nr):
    # o        - 1d fft frequency range: (-2.0j*pi*fftfreq(n))
    # kd,kf,ka - 2d arrays of DO, FRET, Anisotropy rates;
    # pdfa     - 4d array of prob. of dfa states, if independent it is outer product of 1d probs.
    # r0       - 1d array (only batch)
    # for k,p  - first index = batch index
    r0pol = r0[:,None]*array([[2.0, -1.0]]) # 2d[batch, vvvh] r0 * ideal polarization factors for VV,VH;
                                            # pol factors can go to parameter (for example, pol=0 for VM)
    kdf = kd[...,None] + kf[:,None,:]       # 3d array of kd+kf rates
    f   = fmexp(o,kdf,pdfa.sum(3),nr)       # 2d[batch, t] FT[f(t)](o) fluorescence only
    k   = kdf[...,None] + ka[:,None,None,:] # 4d array of kd+kf+ka rates
    fr  = fmexp(o,k,pdfa,nr)                # 2d[batch, t] FT[f(t)*r(t)](o) fluorescence*anisotropy
    
    return f[:,None,:]+r0pol[...,None]*fr[:,None,:] # 3d[batch, vvvh, o]

# Conditioning of experimental IRF (VV,VH): here bg subtraction, fft and normalization
# Note: The IRF beyond measured signal time range is cutted off in this version
def fftirf(irf,bg,nt):
    # irf - 2d [vvvh, t] array of experimental IRFs;      here the same for whole batch
    # bg  - 1d [vvvh] array of irf background amplitudes; here the same for whole batch
    # nt  - lenth of data (time, omega range) to make fft
    irfcut = irf-bg[:,None]       # irf expanded with batch index; bg is expanded with time index
    irfcut[irfcut<0.0] = 0.0      # remove negative values
    
    firf = rfft(irfcut,nt)        # here real fft is used - result is 2 times shorter
    s  = firf[:, 0].real          # property of FFT: Re(fft[f](0)) = sum(f)

    return firf/(s + (s==0))[:, None] # 2d[vvvh,o] complex normalized firf if sum(t) is not zero

# FT of timeshift
# The shifted function (f) in fourier domain is ft(f)*ft(ts)
def ftimeshift(ts,o,eo):
    # ts - 2d [batch, vvvh] array of timeshifts measured in dt units (bins)
    # o  - 1d array of fft frequency (precalculated):            (-2.0j*pi*rfftfreq(n))
    # eo - 1d array of (exp-1) of fft frequency (precalculated): exp(-2.0j*pi*rfftfreq(n))-1
    ox      =  o[None,None,:]     # o      expanded with dimentions for batch and vvvh
    eox     = eo[None,None,:]     # exp(o) expanded with dimentions for batch and vvvh
    tsf,tsi = modf(ts[... ,None]) # ts expanded with dimention for eo; separate frac.,int. parts
    return nv('exp(ox*tsi)*(1.0+tsf*eox)') # 3d [batch, vvvh, o] of ft_ts 



##################################################################
# High level model routines
def model_dfa(pars, aux): # Here pars.size = n_parts*n_batch
    # pars - dictionary of fittable parameters: records should be scalar parameters
    #        so vector parameters should be written element by element
    # aux  - dictionary of auxliary data: exp. data and non-fittable parameters
    # This form of parameters is required for fitting routine (LMfit)
    
    ###############################################
    # Precalculated
    dt  = aux['dt']
    o   = aux['omega']
    eo  = aux['eo']
    nr  = aux['nr']
    # irf
    firf = aux['firf']
    
    ################################################
    # Read pars dictionary
    # This part should correspond to pars dictionary content
    #  0|  1|   2|   3|   4|   5|  6|   7|   8|  9| 10| 11| 12| 13| 14| 15| 16| 17| 18| 19| 20| 21| 22| 23| 24|
    # a0|  g|bgvv|bgvh|scvv|scvh|xaf|tsvv|tsvh|pd1|pd2|kd1|kd2|pf0|pf1|pf2|pf3|kf1|kf2|kf3| r0|pa1|pa2|ka1|ka2|
    
    v = array(list(pars.valuesdict().values())).reshape(-1,25)   # 2d [batch,par] values from batch of pars records
    n_bat, n_par = v.shape
    # 
    a0  = v[:,0]    # initial amplitude of decay (VM/3), this is not the number of counts
    g   = v[:,1]    # Usually not fittable (only for calibration sample)
    bg  = v[:,2:4]  # BG of signal absolute amplitude
    sc  = v[:,4:6]  # fraction of total counts; normaly only added to VV
    xaf = v[:,6]
    ts  = v[:,7:9]
    
    # DO/DA variants
    kd  = v[:,11:13]
    pd  = v[:,9:11]
    if aux['DO']: # DO:
        p0  = pd.sum(1)   # DO total amplitude saved for final scaling
        pd /= p0[:,None]
        
        pf  =  ones((n_bat,1))  # user input ignored
        kf  = zeros((n_bat,1))  # FRET spectrum reduced to single DO (pf=1, kf=0);
    else:        # DA:
        pd /= pd.sum(1)
        
        kf  = hstack([ zeros((n_bat,1)) , v[:,17:20]])
        pf  = v[:, 13:17] # pf absolute values used:
        p0  = pf.sum(1)                              # FRET total amplitude used for final scaling
        pf /= p0[:,None]     
    
    p0 = (a0*p0)[:,None]*hstack([ones((n_bat,1)),g[:,None]])  # 2d[batch,vvvh]
        
    # anisotropy(depolarization) spectrum
    r0  = v[:,20]                                   # fundamental/initial anisotropy
    pa  = v[:,21:23]
    pa /= pa.sum(1)[:,None]                                  # normalised probabilities
    ka  = v[:,23:25]

    ##########################################################
    # Calculation
    pdfa     = pd[:,:,None,None]*pf[:,None,:,None]*pa[:,None,None,:]
    ft_vvvh  = fdecay_dfa(o, kd*dt, kf*dt, ka*dt, pdfa, r0, nr) # 3d [batch, vvvh, o] array of decays
    
    if 'faf' in aux:
        faf = aux['faf']          # 2d[vvvh,o]
        x = array([xaf, 1.0-xaf]) # 2d[x,batch] array of af and f fractions
        if aux['xaf counts']:      # treat xaf as count fractions
            # transform count fractions to amplitude fractions
            f = array([faf[:,0].real.sum()*ones(n_bat),ft_vvvh[:,:,0].real.sum(1)]) # 2d[x,batch]
            x /= f
            x /= x.sum(0)[None,:]
        ft_vvvh  = x[0,:,None,None]*faf[None,:,:] + x[1,:,None,None]*ft_vvvh
    
    ft_vvvh += (sc*ft_vvvh[:,:,0])[...,None]
    fts      = ftimeshift(ts/dt,o,eo)                      # 3d [batch, vvvh, o]
    f_vvvh = p0[...,None] * irfft(ft_vvvh*fts*firf[None,:,:], axis=2, workers=8) + bg[...,None]
    
    return f_vvvh

# Fitting model residual function - minimised array
def model_dfa_res(pars, aux): # output should be 1d array according to LSQ method of LMfit (change to scalar Method for MLE)
    tsel  = aux['tsel']
    bsel  = aux['bsel'] 
    model = model_dfa(pars, aux)[  : ,:,tsel].flatten() # b-size taken from pars
    data  =          aux['data'][bsel,:,tsel].flatten()
    
    if aux['MLE']:
        #data_lng = aux['data_lng'][bsel,:,tsel].flatten()
        mle = model-data
        in0 = ~(data==0)
        mle[in0] += data[in0]*(log(data[in0])-log(model[in0]))
        return mle#*sqrt(data.size/data.sum())
    else: # LSQ
        # to treat poisson statistics at low number of events, see Gehrels(1986)
        i_1 = data<10.0
        s2  = data.copy()
        s2[i_1] += 0.5
        return (data-model)/sqrt(s2)

def pars_post(values, np, do=True): # parameters values post-processing (here only normalization)
    # This part should correspond to pars dictionary content
    #  0|  1|   2|   3|   4|   5|  6|   7|   8|  9| 10| 11| 12| 13| 14| 15| 16| 17| 18| 19| 20| 21| 22| 23| 24|
    # a0|  g|bgvv|bgvh|scvv|scvh|xaf|tsvv|tsvh|pd1|pd2|kd1|kd2|pf0|pf1|pf2|pf3|kf1|kf2|kf3| r0|pa1|pa2|ka1|ka2|
    v = values.reshape(-1,np).copy() # 2d [batch,par] values from batch of pars records
    nb = v.shape[0]
    
    # DO/DA variants
    pd  = v[:,9:11]
    if do: # DO:
        p0  = pd.sum(1)   # DO total amplitude saved for final scaling
        v[:,9:11] = pd / p0[:,None]
        v[:,13:17]  =  hstack([ ones((nb,1)) ,zeros((nb,3))])  # pf user input ignored
    else:        # DA:
        v[:,9:11] = pd/ pd.sum(1)
        pf  = v[:, 13:17] # pf absolute values used:
        p0  = pf.sum(1)                              # FRET total amplitude used for final scaling
        v[:,13:17] = pf/p0[:,None]     
    
    v[:,0] = p0*v[:,0]  
    
    return v # return array ([batch,user pars]) of parameter values

def find_nearest(array, value):
    i = (abs(array - value)).argmin()
    return i, array[i]


def loadTACs(path, names, ext='_G_PS.dat'): # Here return 3d arrays
    """stand alone function to load decays from server"""
    nfiles = names.size
    for i in arange(nfiles, dtype='int'):
        data = loadtxt(path+names[i]+ext).reshape(2,-1)
        if i==0:
            data_bat = zeros((nfiles,2,data.shape[1]))
        data_bat[i,:] = data
    return data_bat

def tp2xfs(t1,t2,p1): # Two times and fractions to means(x,f) and sd (can be 1d arrays)
    tx = t1*p1+t2*(1.0-p1)
    m2 = t1**2*p1+t2**2*(1.0-p1)
    tf = m2/tx
    sdt = sqrt(m2-tx**2)
    return tx,tf,sdt

###########################################################################################
# Fitting scripts
# Preprocess user input
def fit_prep(aux):
    aux['nb']    = aux['data'].shape[0] # batch size
    aux['nt']    = aux['nr']#aux['data'].shape[2] # t-size
    aux['firf']  = fftirf(aux['irf'],aux['irfbg'],aux['nt'])
    
    aux['t']     = arange(aux['nt'])*aux['dt']         
    aux['omega'] = -2.0j*pi*rfftfreq(aux['nt']) # real fft version is used
    aux['eo']    = expm1(aux['omega'])
    
    imin, tmin = find_nearest(aux['t'], aux['tlim'][0])
    imax, tmax = find_nearest(aux['t'], aux['tlim'][1])
    aux['tsel'] = slice(imin,imax+1)
    aux['tlim'] = (tmin, tmax)
    
    aux['bsel'] = slice(0, aux['nb'])
    
    # Calculate FFT of Auto Florescence from its f,a-spectra (fitted and saved before)
    if 'af' in aux:
        nd,na = aux['af'][0].astype('int')
        af_kd = aux['af'][1:nd+1,0]; af_pd = aux['af'][1:nd+1,1]
        af_ka = aux['af'][nd+2: ,0]; af_pa = aux['af'][nd+2: ,1]; af_r0 = aux['af'][nd+1,1]
        af_kf = array([0.0])       ; af_pf = array([1.0]);   # no FRET
        
        af_p  = af_pd[None,:,None,None]*af_pf[None,None,:,None]*af_pa[None,None,None,:] 
        aux['faf'] = fdecay_dfa(aux['omega'],
                                af_kd[None,:]*aux['dt'],
                                af_kf[None,:]*aux['dt'],
                                af_ka[None,:]*aux['dt'],
                                af_p,
                                array([af_r0]),
                                aux['nr'])[0]
    
    
    # pars template processing
    pt = aux['pars template']
    np = len(pt)                 # pars size
    nb = aux['nb']    # batch size
    names = empty(    np , dtype='O'); spnames = empty(np , dtype='O') # saved user names
    exprs = empty(    np , dtype='O'); spexprs = empty(np , dtype='O') # saved user exprs
    # serial/parallel modifications of name, expr fields
    spmod = empty((nb,np), dtype=dtype([('pname', 'O'),                # precalculated s names
                                        ('sname', 'O'),                # precalculated p names
                                        ('pexpr', 'O'),                # precalculated s exprs
                                        ('sexpr', 'O')]))              # precalculated p exprs
    # intermediate, changeble parameters structure
    pb    = empty((nb,np), dtype=dtype([('name' , 'O'),                # placeholder for s,p names
                                        ('value', 'f8'),
                                        ('vary' , '?'),
                                        ('min'  , 'f8'),
                                        ('max'  , 'f8'),
                                        ('expr' , 'O')]))              # placeholder for s,p exprs.
    for j in arange(np): # for each parameter instance (name) in the template
        # names
        name = pt[j][0]
        names[j] = name
        spmod['sname'][:,j] = name                                                 # not indexed for sesial
        spmod['pname'][:,j] = array([name+'_%d'%i for i in arange(nb)], dtype='O') #     indexed for parallel
        # exprs
        expr = pt[j][5] # user expr string
        if expr==None:
            expr = ''
        exprs[j] = expr
        spmod['sexpr'][:,j] = array([expr.replace('_l',''     ).replace('_g',''  ) for i in arange(nb)], dtype='O')
        spmod['pexpr'][:,j] = array([expr.replace('_l','_%d'%i).replace('_g','_0') for i in arange(nb)], dtype='O')
        # values, min, max: set all the same if user value is scalar, else unfold user array
        v = pt[j][1]; pb['value'][:,j] = full(nb, v) if isscalar(v) else v
        v = pt[j][3]; pb['min'][:,j]   = full(nb, v) if isscalar(v) else v
        v = pt[j][4]; pb['max'][:,j]   = full(nb, v) if isscalar(v) else v
        # vary
        pb['vary'][:,j] = full(nb, pt[j][2], dtype='?') # the same for all batch; user input can be 0,1 instead of bool
   
    aux['pars names'] = names
    aux['pars exprs'] = exprs
    aux['pars modif'] = spmod
    aux['pars structure'] = pb


def getParameters(aux, serial=False, ib=0):
    pars = Parameters()
    
    if serial:
        bsel = slice(ib,ib+1)
        aux['bsel'] = bsel
        # Generate Parameter class for parallel fit
        aux['pars structure']['name'] =  aux['pars modif']['sname']
        aux['pars structure']['expr'] =  aux['pars modif']['sexpr']
    else:
        bsel= slice(0,aux['nb'])
        aux['bsel'] = bsel
        # Generate Parameter class for parallel fit
        aux['pars structure']['name'] =  aux['pars modif']['pname']
        aux['pars structure']['expr'] =  aux['pars modif']['pexpr']
    
    pars.add_many(*aux['pars structure'][bsel].flatten())
    return pars


# one-pass fit (serial/parallel options)
def fit1(aux, serial=False, fitopt={'xtol':1e-6, 'ftol':1e-6}):
    start_extime = time.time()
    
    fit_prep(aux) # prepare butched data arrays
    nb     = aux['nb']
    nt     = aux['nt']
    ntsel  = aux['tsel'].stop-aux['tsel'].start
    pnames = aux['pars names'] 
    
    if serial:
        model_ini     = empty((nb,2,nt))
        model_fit     = empty((nb,2,nt))
        model_fit_res = empty((nb,2,ntsel))
        fit_result    = empty(nb, dtype='O')
        pars_fit      = DataFrame(columns=pnames)

        chi2 = empty(nb)
        chi2vvvh = empty((nb,2))
        
        for i in arange(nb):
            pars = getParameters(aux, serial=True, ib=i)
            model_ini[i] = model_dfa(pars, aux)
            model_min    = Minimizer(model_dfa_res,        #Minimizer
                                  pars,
                                  fcn_kws={'aux': aux} 
                                 )
            fit_result[i]    = model_min.minimize(**fitopt)
            model_fit[i]     = model_dfa(fit_result[i].params, aux)
            model_fit_res[i] = model_dfa_res(fit_result[i].params, aux).reshape((1,2,ntsel))
            pars_fit.loc[i]  = pars_post(array(list(fit_result[i].params.valuesdict().values())), pnames.size, aux['DO'])[0]
            
            chi2[i] = fit_result[i].redchi
            chi2vvvh[i] = mean(model_fit_res[i]**2,axis=-1)
            print('%d'%i, end='|')
        
        chi2_global = mean(chi2)
        
    else:
        pars = getParameters(aux, serial=False) # Prameters class from batch structured array
        model_ini = model_dfa(pars, aux)
        model_min = Minimizer(model_dfa_res,        #Minimizer
                              pars,
                              fcn_kws={'aux': aux} 
                             )
        fit_result    = model_min.minimize(**fitopt)
        model_fit     = model_dfa(fit_result.params, aux)
        model_fit_res = model_dfa_res(fit_result.params, aux).reshape(aux['data'].shape[:2]+(-1,))
        pars_fit = DataFrame(pars_post(array(list(fit_result.params.valuesdict().values())), pnames.size,aux['DO']), columns=pnames)
        
        chi2vvvh = mean(model_fit_res**2,axis=-1)
        chi2     = 0.5*chi2vvvh.sum(-1)
        chi2_global = fit_result.redchi
        
    
    pars_fit.insert(0, 'chivh', chi2vvvh[:,1])
    pars_fit.insert(0, 'chivv', chi2vvvh[:,0])
    pars_fit.insert(0, 'chi', chi2)
    
    print('\nExecution time = %f s'%(time.time()-start_extime))
    print('Global chi2r =',chi2_global)
        
    return model_ini, model_fit, model_fit_res, fit_result, pars_fit

############################################################################
# Plot scripts
# these scipts user and fitting specific - change as you wish
# def fit_and_chi2_plot(aux_pass1, fit_pass1, aux_pass2, fit_pass2, apc, i):
    # #  0|  1|   2|   3|   4|   5|  6|   7|   8|  9| 10| 11| 12| 13| 14| 15| 16| 17| 18| 19| 20| 21| 22| 23| 24|
    # # a0|  g|bgvv|bgvh|scvv|scvh|xaf|tsvv|tsvh|pd1|pd2|kd1|kd2|pf0|pf1|pf2|pf3|kf1|kf2|kf3| r0|pa1|pa2|ka1|ka2|
    # bgvv_ini = aux_pass1['pars structure'][i,2][1]
    # bgvh_ini = aux_pass1['pars structure'][i,3][1]
    
    # data  = aux_pass2['data']
    # dt    = aux_pass2['dt']
    # tdata = arange(data.shape[-1])*dt
    # tmodl = aux_pass2['t']
    # nr    = aux_pass2['nr']
    # tlim  = aux_pass2['tlim']
    
    # tdata = arange(data.shape[-1])*dt
        
    # bgvv_pass2 = fit_pass2[4]['bgvv'].to_numpy()[i]
    # bgvh_pass2 = fit_pass2[4]['bgvh'].to_numpy()[i]
    # chi_pass1  = fit_pass1[4]['chi'].to_numpy()
    # chi_pass2  = fit_pass2[4]['chi'].to_numpy()
    
    # model_pass2 = fit_pass2[1]

    # figure(figsize(17,5))
    
    # subplot(1,2,1)
    # semilogy(tdata,data[i,0],'b.',ms=2,alpha=0.2)
    # semilogy(tdata,data[i,1],'r.',ms=2,alpha=0.2)
    # semilogy(tmodl,model_pass2[i,0],'b',lw=0.7)
    # semilogy(tmodl,model_pass2[i,1],'r',lw=0.7);
    # axhline(bgvv_ini, c='grey', lw=0.5)
    # axhline(bgvh_ini, c='grey', lw=0.5)
    # axhline(bgvv_pass2, c='b', lw=0.5)
    # axhline(bgvh_pass2, c='r', lw=0.5)
    # axvline(tlim[0], c='grey', lw=0.5)
    # axvline(tlim[1], c='grey', lw=0.5)
    # axvline(nr*dt, c='k', lw=0.5)
    # xlabel('t, ns', fontsize=15)
    # ylim(1,);

    # subplot(1,2,2)
    # loglog(apc,chi_pass1,'r.', alpha=0.5, label='Pass1');
    # loglog(apc,chi_pass2,'k.', label='Pass2');
    # axvline(apc[i],c='grey', lw=0.5)
    # axhline(1, c='k', lw=0.5)
    # ylabel(r'$\chi^2_r$', fontsize=15)
    # xlabel('~ Concentration', fontsize=15)
    # legend()

# def FRET2lspec_xFRET__plot(aux, fit, apc):
    # kf    = fit[4][['kf1','kf2']].to_numpy()
    # pf0pf = fit[4][['pf0','pf1','pf2']].to_numpy()

    # xdo = pf0pf[:,0]
    # pf  = pf0pf[:,1:]
    # pf  = pf/pf.sum(-1)[:,None]

    # i = argsort(kf,axis=1).astype('bool') # sort rates
    # kf1 = kf.flatten()[i.flatten()]; kf2 = kf.flatten()[~i.flatten()]
    # pf1 = pf.flatten()[i.flatten()]; pf2 = pf.flatten()[~i.flatten()]

    # taufX, taufF, taufS = tp2xfs(1/kf1,1/kf2,pf1)

    # figure(figsize(17,5))
    # alf = 1

    # subplot(1,3,1)
    # loglog(apc, 1/kf1, 'b.', alpha=alf, label=r'$\tau_{f1}$')
    # loglog(apc, 1/kf2, 'r.', alpha=alf, label=r'$\tau_{f2}$')
    # loglog(apc, taufX, 'g.', alpha=alf, label=r'$\langle\tau_f\rangle_x$')
    # legend( fontsize=12)
    # xlabel('~ Concentration', fontsize=15)
    # ylim(0.08,1000)

    # subplot(1,3,2)
    # loglog(apc, taufF, '.',c='darkmagenta', alpha=alf, label=r'$\langle\tau_f\rangle_F$')
    # loglog(apc, taufS, '.',c='goldenrod', alpha=alf, label=r'$sd(\tau_f)$')
    # legend( fontsize=12)
    # xlabel('~ Concentration', fontsize=15)
    # ylim(0.08,1000)

    # subplot(1,3,3)
    # semilogx(apc, pf1, '.',c='steelblue', alpha=alf, label=r'$p_{f1}$')
    # semilogx(apc, 1-xdo, 'k.', alpha=alf, label=r'$x_{fret}$')
    # axhline(0.5,c='k',lw=0.5)
    # legend( fontsize=12)
    # xlabel('~ Concentration', fontsize=15)
    # ylim(0,1.01);