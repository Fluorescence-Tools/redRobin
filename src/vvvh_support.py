# support functions for vvvh fitting
# when choosing a very different fit model, this file must be modified
# the vvvh_fittingOO can remain unchanged.
# mostly written by Nicolaas van der Voort, some pieces taken from Oleg Opanasyuk
# 10 March 2023

import vvvh_fittingOO as vvvhfit
import numpy as np
import pandas as pd




######## generate the auxiliary information needed for the OO fit routine ######

def fitDO(D0_vvvh, data_irf, data_af, imstats, af_G_norm, tauxD0, bsel = None,
        csvout = None):
    """entry function for RedRobin for Donor only D0_vvvh
    prepares data and runs single 2lt fit
    Subsequentially calculates derived variables
    returns a dataframe with fitted and derived variables"""
    # Select data for fitting
    if not bsel:
        bsel =  np.arange(len(D0_vvvh),dtype='int')#array([153], dtype='int') # arange(195,dtype='int')   #
    D0_vvvh   =  D0_vvvh[bsel]
    a00, bgvv0, bgvh0, xaf0 = get_par_estimates(D0_vvvh, imstats, af_G_norm)
    aux2lt = genAux2lt(D0_vvvh, data_irf, data_af, a00, bgvv0, bgvh0, xaf0)
    fit2lt = vvvhfit.fit1(aux2lt, serial = True, fitopt={'xtol':1e-5, 'ftol':1e-5})
    add_2ltDerivedVars(fit2lt, tauxD0) #add taux, tauf, Eavg

    # rename columns
    dfrm2lt = fit2lt[4] # fourth element is fit dataframe
    dfrm2lt.columns = dfrm2lt.columns.values + "2lt"
    dfrm2lt.index = imstats.index #set index to mask names
    # save dfrm
    if csvout:
        dfrm2lt.to_csv(csvout)
    return dfrm2lt 

def fitDA(DA_vvvh, data_irf, data_af, imstats, af_G_norm, tauxD0, bsel = None,
        csvout = None):
    """entry function for RedRobin for donor acceptor data
    prepares data and runs 2lt fit and D0DA pattern fit
    Subsequentlially derived variables are calculated.
    returns a dataframe with fitted and derived variables"""
    # Select data for fitting
    if not bsel:
        bsel =  np.arange(len(DA_vvvh),dtype='int')#array([153], dtype='int') # arange(195,dtype='int')   #
    DA_vvvh   =  DA_vvvh[bsel]
    a00, bgvv0, bgvh0, xaf0 = get_par_estimates(DA_vvvh, imstats, af_G_norm)
    
    # first do 2 lt fit
    aux2lt = genAux2lt(DA_vvvh, data_irf, data_af, a00, bgvv0, bgvh0, xaf0)
    fit2lt = vvvhfit.fit1(aux2lt, serial = True, fitopt={'xtol':1e-5, 'ftol':1e-5})
    
    # then do D0DA fit
    auxD0DA = genAuxD0DA(DA_vvvh, data_irf, data_af, a00, bgvv0, bgvh0, xaf0)
    da_patternfit = vvvhfit.fit1(auxD0DA, serial = True, fitopt={'xtol':1e-10, 'ftol':1e-10})
    
    #add derived variables
    add_D0DADerivedVars(da_patternfit, auxD0DA, 18)
    xFRET = 1 - da_patternfit[4]['pf0']
    add_2ltDerivedVars(fit2lt, tauxD0, xFRET) #add taux, tauf, Eavg, E_species
    
    #rename columns and merge
    D0DAdfrm = da_patternfit[4]
    D0DAdfrm.columns = D0DAdfrm.columns.values + "D0DA"
    dfrm2lt = fit2lt[4]
    dfrm2lt.columns = dfrm2lt.columns.values + "2lt"
    dfrm_merged = pd.concat([D0DAdfrm, dfrm2lt], axis = 1)
    dfrm_merged.index = imstats.index #set index to mask names
    #save dfrm
    if csvout:
        dfrm_merged.to_csv(csvout)
    return dfrm_merged 
    
def get_par_estimates(data, imstats, af_G_norm):
    max_vvvh = np.amax(data, axis=2)
    a00 = ((max_vvvh[:,0]+2.0*max_vvvh[:,1])/3.0)
    bg_est =  np.mean(data[:,:,:10], axis=2)
    bgvv0  =  bg_est[:,0]
    bgvh0  =  bg_est[:,1]
    xaf0   =  calc_af_est(imstats, af_G_norm)
    return a00, bgvv0, bgvh0, xaf0
    


def calc_af_est(imstats, af_G_norm):
    # DA
    NG = imstats['NG-tot'].to_numpy()
    pic= imstats['surfaceMax'].to_numpy()
    fr = 20
    nG = NG/pic/fr # normalised intensity of DA
    # (NV): I find this definition of xaf confusing, because it is defined as a fraction
    # of the amplitude, rather than as an absolute amount of photons
    # cannot oversee it, not changing it for now.
    xaf = af_G_norm/nG
    return xaf
    
#def aid_genaux(data, data_irf, data_af, af_G_norm, imstats, mode = '2lt', bsel = None):
#    assert mode in ['2lt', 'D0DA'], 'invalid mode'
#    # Select data for fitting
#    if not bsel:
#        bsel =  np.arange(len(data),dtype='int')#array([153], dtype='int') # arange(195,dtype='int')   #
#    data   =  data[bsel]
#    
#    max_vvvh = amax(data, axis=2)
#    a00 = ((max_vvvh[:,0]+2.0*max_vvvh[:,1])/3.0)[bsel]
#    #a00_vvvh =  1.3*amax(data, axis=2)[bsel]
#    bg_est =  mean(data[:,:,:10], axis=2)
#    bgvv0  =  bg_est[bsel,0]
#    bgvh0  =  bg_est[bsel,1]
#    xaf0   =  calc_af_est(imstats, af_G_norm)[bsel]
#    # Fitting session settings
#    if mode == '2lt':
#        return genAux2lt(data, data_irf, data_af, a00, bgvv0, bgvh0, xaf0)
#    if mode == 'D0DA':
#        return genAuxD0DA(data, data_irf, data_af, a00, bgvv0, bgvh0, xaf0)
    
def genAux2lt(data, data_irf, data_af, a00, bgvv0, bgvh0, xaf0,
                kd1 = 1 / 1.64, kd2 = 1 / 2.75):
    """generate auxiliary data for 2 lifetime fit
    The two lifetimes are fixed to EGFP values, this seems to be necessary
    to avoid fitting noise"""
    # the definition of 
    return {
            'data' : data,           # array of data: 3d [batch,vvvh,t] array, even for siglge data: ([0:1], not [0]) !!!  
            'irf'  : data_irf,       # 2d [vvvh, t] array; the same for all DO,DA
            'irfbg': np.array([85,45]), # the same for all DO,DA
            'af'   : data_af,        # fluor./aniso. spectrums - proprientary, see description; the same for all DO,DA
            'dt'   : 0.032,          # time bin width
            'nr'   : 780,            # Excitation repetition period = nr*dt
            'tlim' : (0,24),         # fitting t range [ns]. Altered to the nearest point
            'xaf counts': True,      # Treat xaf as count fractions
            'DO'   : True,           # True: ignore fret (to save calculation time)
            'MLE'  : True,           # True: minimise likelihood instead of LSQ
            'pars template': [       # User template for fitting parameters
            # name   value             vary     min        max              expr 
            ['a0'  , a00             , True  ,  0.0      , 3.0*a00           , None    ], 
            ['g'   , 1/0.92          , False ,  0.0      , 2                 , None    ], 
            # signal BG
            ['bgvv', bgvv0           , True  ,  0.0      , 1.5*bgvv0 + 10    , None    ],
            ['bgvh', bgvh0           , True  ,  0.0      , 1.5*bgvh0 + 10    , None    ],
            # scatter
            ['scvv', 0.05            , True  ,  0.0      , 10.0              , None    ],
            ['scvh', 0.0             , False ,  0.0      , 10.0              , None    ], 
            # AF fcaction:  from 0 to 1
            ['xaf' , xaf0            , False ,  0.0      , 1.0               , None    ],
            # time shifts
            ['tsvv', 0.00            , True  ,  -1.9     , 1.9               , None    ],
            ['tsvh', -0.00           , True  ,  -1.9     , 1.9               , None    ],
            # DO spectrum
            ['pd1' , 0.5             , True  ,  0.0      , 1.0               , None    ],
            ['pd2' , 0.5             , True  ,  0.0      , 1.0               ,'1-pd1_l'],
            ['kd1' , kd1             , False ,  0.0      , 10.0              , None    ],
            ['kd2' , kd2             , False ,  0.0      , 10.0              , None    ],
            # FRET spectrum (including DO fraction pf0)
            ['pf0' , 1.0             , False ,  0.0      , 1.0               , None    ],
            ['pf1' , 0.0             , False ,  0.0      , 1.0               , None    ],
            ['pf2' , 0.0             , False ,  0.0      , 1.0               , None    ],
            ['pf3' , 0.0             , False ,  0.0      , 1.0               , None    ],
            ['kf1' , 0.96            , False ,  0.0      , 10.0              , None    ],
            ['kf2' , 0.25            , False ,  0.0      , 10.0              , None    ],
            ['kf3' , 0.25            , False ,  0.0      , 10.0              , None    ],
            # Depolarization spectrum (including rO)
            ['r0'  , 0.38            , True  ,  0.0      , 0.4               , None    ], 
            ['pa1' , 1.0             , False ,  0.0      , 1.0               , None    ], 
            ['pa2' , 0.0             , False ,  0.0      , 1.0               ,'1-pa1_l'],
            ['ka1' , 1/45.0          , True  ,  0.0      , 10.0              , None    ], 
            ['ka2' , 0.1             , False ,  0.0      , 10.0              , None    ]]}

def genAuxD0DA (data, data_irf, data_af, a00, bgvv0, bgvh0, xaf0):
    return {
        'data' : data,           # array of data: 3d [batch,vvvh,t] array, even for siglge data: ([0:1], not [0]) !!!  
        'irf'  : data_irf,       # 2d [vvvh, t] array; the same for all DO,DA
        'irfbg': np.array([85,45]), # the same for all DO,DA
        'af'   : data_af,        # fluor./aniso. spectrums - proprientary, see description; the same for all DO,DA
        'dt'   : 0.032,          # time bin width
        'nr'   : 780,            # Excitation repetition period = nr*dt
        'tlim' : (0,24),         # fitting t range [ns]. Altered to the nearest point
        'xaf counts': True,      # Treat xaf as count fractions
        'DO'   : False,          # True: ignore fret (to save calculation time)
        'MLE'  : True,           # True: minimise likelihood instead of LSQ
        'pars template': [       # User template for fitting parameters
        # name   value             vary     min        max              expr 
        ['a0'  , a00             , True  ,  0.0      , 3.0*a00           , None    ], 
        ['g'   , 1/0.92          , False ,  0.0      , 2                 , None    ], 
        # signal BG
        ['bgvv', bgvv0           , True  ,  0.0      , 1.5*bgvv0 + 10    , None    ],
        ['bgvh', bgvh0           , True  ,  0.0      , 1.5*bgvh0 + 10    , None    ],
        # scatter
        ['scvv', 0.05            , True  ,  0.0      , 10.0              , None    ],
        ['scvh', 0.0             , False ,  0.0      , 10.0              , None    ], 
        # AF fcaction:  from 0 to 1
        ['xaf' , xaf0            , False ,  0.0      , 1.0               , None    ],
        # time shifts
        ['tsvv', 0.00            , True  ,  -1.9     , 3.0               , None    ],
        ['tsvh', -0.12           , True  ,  -1.9     , 3.0               , None    ],
        # DO spectrum
        ['pd1' , 0.5             , False ,  0.0      , 1.0               , None    ],
        ['pd2' , 0.5             , False ,  0.0      , 1.0               ,'1-pd1_l'],
        ['kd1' , 1/1.68          , False ,  0.0      , 10.0              , None    ],
        ['kd2' , 1/2.75          , False ,  0.0      , 10.0              , None    ],
        # FRET spectrum (including DO fraction pf0)
        ['pf0' , 0.5             , True  ,  0.0      , 1.0               , None    ],
        ['pf1' , 0.375           , True  ,  0.0      , 1.0               ,'(1-pf0_l)*0.75'],
        ['pf2' , 0.125           , True  ,  0.0      , 1.0               ,'1-pf0_l-pf1_l' ],
        ['pf3' , 0.0             , False ,  0.0      , 1.0               , None    ],
        ['kf1' , 0.154          , False  ,  0.0      , 10.0              , None    ],
        ['kf2' , 1.346          , False  ,  0.0      , 10.0              , None    ],
        ['kf3' , 1.0             , False ,  0.0      , 10.0              , None    ],
        # Depolarization spectrum (including rO)
        ['r0'  , 0.375           , True  ,  0.0      , 0.4               , None    ], 
        ['pa1' , 1.0             , False ,  0.0      , 1.0               , None    ], 
        ['pa2' , 0.0             , False ,  0.0      , 1.0               ,'1-pa1_l'],
        ['ka1' , 1/42.0          , True  ,  0.0      , 10.0              , None    ], 
        ['ka2' , 0.1             , False ,  0.0      , 10.0              , None    ]]}
        
def fix_a0_bg_ts_tlim(aux, pars_fit):
    """aux are the settings you want to use for tail fitting
    pars_fit are the fit settings where timeshift, amplitude and background are
    taken from, e.g., from a full-range fit."""
    aux['pars template'][0][1] = pars_fit['a0'].to_numpy()
    aux['pars template'][0][2] = False #fix ts
    aux['pars template'][2][1] = pars_fit['bgvv'].to_numpy()
    aux['pars template'][2][2] = False #fix ts
    aux['pars template'][3][1] = pars_fit['bgvh'].to_numpy()
    aux['pars template'][4][2] = False #fix ts
    aux['pars template'][7][1] = pars_fit['tsvv'].to_numpy()
    aux['pars template'][7][2] = False #fix ts
    aux['pars template'][8][1] = pars_fit['tsvh'].to_numpy()
    aux['pars template'][8][2] = False
    aux['tlim'] = (2, 24)
        
########## add derived values after fit #######################################

def add_D0DADerivedVars(da_fit, aux, max_t):
    fit_df = da_fit[4]
    for i in range(len(fit_df)):
        xFRET_post = get_posterior_xFRET(da_fit[3][i], aux, max_t = max_t)
        da_fit[4].loc[i, 'xFRET_post'] = xFRET_post
    fit_df['<k>'] = (fit_df['kf1'] * fit_df['pf1'] + fit_df['kf2'] * fit_df['pf2']) / \
            (fit_df['pf1'] + fit_df['pf2'])
    return
            
def add_2ltDerivedVars(fit2lt, tauxD0, xFRET = None):
    fit2lt[4]['taux'] = fit2lt[4]['pd1'] / fit2lt[4]['kd1'] + fit2lt[4]['pd2'] / fit2lt[4]['kd2']
    fit2lt[4]['tauf'] = (fit2lt[4]['pd1'] / fit2lt[4]['kd1']**2 + \
                         fit2lt[4]['pd2'] / fit2lt[4]['kd2']**2) / fit2lt[4]['taux']
    fit2lt[4]['Eavg'] = 1 - fit2lt[4]['taux'] / tauxD0
    if xFRET is not None:
        fit2lt[4]['Especies'] = fit2lt[4]['Eavg'] / xFRET
    return

########## various testing / debigging functions ##############################

def calc_eps(t, model_da_result):
    pf0 = model_da_result.params["pf0"].value
    pf1 = model_da_result.params["pf1"].value
    kf1 = model_da_result.params["kf1"].value
    pf2 = model_da_result.params["pf2"].value
    kf2 = model_da_result.params["kf2"].value
    return pf0 + pf1 * np.exp(-kf1*t) + pf2 * np.exp(-kf2*t)
    
def calc_r(t, model_da_result):
    r0 = model_da_result.params["r0"].value
    pa1 = model_da_result.params["pa1"].value
    pa2 = model_da_result.params["pa2"].value
    ka1 = model_da_result.params["ka1"].value
    ka2 = model_da_result.params["ka2"].value
    return r0 * (pa1 * np.exp(-ka1*t) + pa2 * np.exp(-ka2*t))

def get_norm_irf(aux):
    vv = aux['irf'][0] / sum(aux['irf'][0])
    vh = aux['irf'][1] / sum(aux['irf'][1])
    return np.vstack([vv,vh])

def calc_eps_conv(t, model_da_result, aux):
    eps_raw = calc_eps(t, model_da_result)
    irf_vvvh = get_norm_irf(aux)
    vv = np.convolve(irf_vvvh[0], eps_raw)[:len(t)]
    vh = np.convolve(irf_vvvh[1], eps_raw)[:len(t)]
    eps_vvvh = np.vstack([vv, vh])
    return eps_vvvh

def get_posterior_xFRET(pars, aux, min_t = None, max_t = 25):
    """if you don't trust IRF convolution, this function just calculates the
    xFRET decay on the posterior distribution.
    This caries a systematic deviation, but is also very robust, making it usefull
    When one is uncertain about the very compicated fit result."""
    t = np.arange(0, max_t, aux['dt'])
    eps_conv_vv = calc_eps_conv(t, pars, aux)[0]
    return get_Delta_xFRET(aux['dt'], eps_conv_vv, mintime = max_t, maxtime = min_t)
    
def get_Delta_xFRET(dt, epst, mintime = None, maxtime = None):
    #I am not super happy with this functions functionality. Can easily break.
    if mintime == None:
        mintime = np.argmin(epst)
    else:
        mintime = int(mintime / dt)#convert to index
        
    if maxtime == None:
        maxtime = np.argmax(epst)
    else:
        maxtime = int(maxtime / dt)
    return epst[maxtime] - epst[mintime]

def get_xFRETAtTime(fit_dfrm, time):
    return fit_dfrm['pf1'] * (1- np.exp(- fit_dfrm['kf1'] * time) )+ \
        fit_dfrm['pf2'] * (1 - np.exp(- fit_dfrm['kf2'] * time) ) + \
        fit_dfrm['pf3'] * (1 - np.exp(- fit_dfrm['kf3'] * time) )
        

    

