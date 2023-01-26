#import aid_functions as aid
import batchplot as bp
import os
import pandas as pd
#import ImageManipulation as IM
import numpy as np
import fitDA
import gc
import tiffile #pip install tiffile if missing
from PIL import Image
import copy

import warnings
from scipy.ndimage import gaussian_filter
#note name df is blocked for dataframe
debug = False
if debug:
    import matplotlib.pyplot as plt

warnings.simplefilter("default")

from ltImage import LtImage, ImChannel

def trymkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
        


################################################################################
#This class structure is disadvantageous because everytime I test a bugfix,
#I need to reanalyze all data to check it, considering the files are very large
#this takes a long time, solution: make functional

def PS2PandS(TACPS):
    assert len(TACPS) %2 == 0 and len(TACPS.shape) == 1 ,\
        "array not divisible by two or not 1D."
    ntacs = len(TACPS) / 2
    TACP = TACPS[:ntacs]
    TACS = TACPS[ntacs:]
    return TACP, TACS

def PandS2PS(TACP, TACS):
    assert TACP.shape == TACS.shape and len(TACS.shape) ==1, \
        "arrays dimensions mismatch or are not 1D."
    ntacs = len(TACP)
    TACPS = np.zeros(ntacs*2)
    TACPS[:ntacs] = TACP
    TACPS[ntacs:] = TACS
    return TACPS
def calculateRatesFromDf(df, 
                              integrationtime = 1, 
                              Gpower = 1, #Dpower
                              Ypower = 1):#Apower
    """integration time is dwelltime * Nframes"""
    #get max surface
    surfaceCols = [col for col in df if col.startswith("surface")]
    df['surfaceMax'] = df[surfaceCols].max(axis = 1)
    #need to change label to something generic
    intCols = [col for col in df if col.startswith("N_")]
    for col in intCols:
        df['rate' + col[2:]] = df[col] / (integrationtime * df['surfaceMax'])
    #works if original channels are called GS, GP, RS, RP, etc.
    for label, power in zip(['Gtot', 'Rtot', 'Ytot'], [Gpower, Gpower, Ypower]):
        #LPC stands for LaserPowerCorrected
        df['rate' + label + '_LPC'] = df['rate'+label] / power
    return df
    
    
def addConc(df, 
    Acf, BnormG, BnormY, pMembrane, 
    maturationG = 1, maturationY = 1):
    """calculates corrected rates and corresponding concentration from
    spectroscopic parameters:
        df: dataframe containing the data, if the dataframe is missing input
            columns, it is skipped
        Acf: confocal volume in um2, usually from FCS calibration
        BnormD: brightness of donor / EGFP at 1 uW excitation
        BnormA: brightness of acceptor / mCh at 1 uW excitation
        """
    df['concG'] = df['rateGtot_LPC'] * pMembrane / (Acf * BnormG) / maturationG
    df['concY'] = df['rateYtot_LPC'] * pMembrane / (Acf * BnormY) / maturationY
    df['concYconcG'] = df['concY'] / df['concG']
    #In principle BnormEGFP is affected by FRET, which can be corrected for
    #by also including the red signal in this calculation.
    #However, red need not be added because the amount of FRET is neglegible.
    #to add it requires calculation of the effective brightness for the R
    #channel, that also corrects for crosstalk and direct excitation.
    df['concTotal'] = df['concG'] + df['concY']

def addCorrProxFret(df, kprox):
    zeros = df['xFRETD0DA']*0#have zeros were xFRET is a number
    xFRET_cor = df['xFRETD0DA'] - simpleWolberHudson(df['concY'], kprox)
    df['xFRET_cor'] = pd.DataFrame({'zeros': zeros, 'xFRET_cor' : xFRET_cor}).max(axis = 1)
    #return df
def addESpecies(df, tauxD0 = 2.30):
    for columnName in df.columns:
        if columnName.startswith('taux') or columnName.startswith('<t>x'):
            extension = columnName[4:]
            df['E_avg'+extension] = 1-df[columnName] / tauxD0
            #currently only one xFRET is defined, change later to accept multiple xFRET from same extension
            if 'xFRETD0DA' in df.columns:
                df['E_species'+extension] = df['E_avg'+extension]/df['xFRETD0DA']
def simpleWolberHudson(x, k1):
    return 1- np.exp(-k1*x)

def cleanImage(image):
    del(image.baseLifetime)
    del(image.workLifetime)
    return 1

def saveTACs(imChannelLst, TACdir, label):
    for channel in imChannelLst:
        decayout = os.path.join(TACdir, label + channel.name + '.dat')
        np.savetxt(decayout, channel.decay, fmt = '%i')
        #I chose to store VM, r and PS in the parallel channel 'p'
        if hasattr(channel, "VM"):
            #name[:-1] to get rid of the last letter, which should be "P"
            VMout = os.path.join(TACdir, label + channel.name[:-1] + '_VM.dat')
            np.savetxt(VMout, channel.VM, fmt = '%i')
        if hasattr(channel, "PS"):
            PSout = os.path.join(TACdir, label + channel.name[:-1] + '_PS.dat')
            np.savetxt(PSout, channel.PS, fmt = '%i')
        if hasattr(channel, "r"):
            rout = os.path.join(TACdir, label + channel.name[:-1] +'_r.dat')
            np.savetxt(rout, channel.r, fmt = '%.5e')

def genDefaultFitKwargs():
    return { #some dummy mono exponential
            'D0dat' : np.exp(-np.arange(0,25, 0.064) / 2.5),
            'decaytype' : 'VM',
            'fitrange' : (30, 380)}
            
def getMask(fname):
    mask = tiffile.imread(fname)
    #we need masks to be binary
    mask [ mask != 0 ] = 1
    return mask #mask is np array
    
def trySaveToCSV(dfrm, outname):
    try:
        dfrm.to_csv(outname)
    except PermissionError:
        print('could not save stats, is the file open in Excel?')
        
def buildMaskFromIntensityThreshold(imChannel_lst, threshold):
    """sums all photons of intensities in listed channels, 
    Compares the intensity with threshold.
    returns boolean array of dimension image size"""
    for imChannel in imChannel_lst:
        assert hasattr(imChannel, 'intImage'), 'no 2D image in imChannel'
    sumimg = np.zeros(imChannel_lst[0].intImage.shape)
    for imChannel in imChannel_lst:
        sumimg += imChannel.intImage
    return sumimg > threshold
            
def getPSindices(imChannelLst):
    # extract sets of P ans S sets in channel list
    channelNames = [channel.name for channel in imChannelLst]
    PSindices = []
    for Pindex, name in enumerate(channelNames):
        if name.endswith('P'):
            Sname = name[:-1]+'S'
            try:
                Sindex = channelNames.index(Sname)
                PSindices.append([Pindex, Sindex])
            except ValueError:
                pass
    return PSindices
#container class         
class CELFISsample():
    def __init__(self, ltImage, imChannelLst, ptuname, maskid = None, time = None):
        self.ltImage = ltImage
        self.imChannelLst = imChannelLst
        self.ptuname = ptuname
        self.maskid = maskid
        self.time = time
class sampleSet():
    """This class is intended to automate image analysis for cellular data to
        avoid time-consuming manual work in AnI.
        To work, this script neads a functioncal copy of Seidel in the pythonpath
        """
    #set to False after default settings have been applied once
    isApplyDefaultSettings = True 
    def __init__(self,
                 wdir,
                 templateImChannelLst,
                 **settings
                 ):
                 #TODO add sampleset identifiers here
        """
        input:
            ntacs: number of TAC channels
            pulsetime: inverse of laser repetition rate, in ns
            dwelltime: pixel dwell time in seconds
            Nframes: number taken in imreading and for calculating total
                illumination time
            threshold: all pixels below this threshold are set to zero
            TAC_range: set in hydraharp
        """
        self.setDefaultSettings(wdir)
        self.templateImChannelLst = templateImChannelLst
        self.setUserSettings(**settings)
        self.CELFISsampleLst = []
        return
        
    def setDefaultSettings(self, wdir):
        self.wdir = wdir
        self.TACdir = os.path.join(wdir, 'TAC')
        self.resdir = os.path.join(wdir, 'results')
        self.imdir = os.path.join(wdir, 'images')
        self.ntacs =  1024
        self.g_factor = 1
        self.Nframes = -1
        #this allows analyzing only a portion of the ptu files
        self.dataselect = (0, None)
        self.PSshift = 0
        self.line_setup = [1,2]#PIE mode
        self.timeLst = None
        self.cell_phenotypeLst = None
        self.sampleType = 'sample not specified'
        self.sampleAddition = 'sample addition not specified'
        self.experimentId = 'experiment id not specified'
        self.roundId = 0
        self.powerD = 1
        self.powerA = 1
        self.pMembrane = 1
        self.Acf = 1
        self.kprox = 1e-9
        self.maturationD = 1
        self.maturationA = 1
        self.BnormDonor = 1
        self.BnormAcceptor = 1
        
    
    def setUserSettings(self, **settings):
        for setting, settingvalue in zip(settings, settings.values()):
            setattr(self, setting, settingvalue)
        self.completeSetting()
    def completeSetting(self):
        trymkdir(self.TACdir)
        trymkdir(self.resdir)
        trymkdir(self.imdir)
        self.ptufiles = bp.appendOnPattern(self.wdir, 'ptu')\
            [self.dataselect[0]: self.dataselect[1]]

    def analyzeDir(self, maskdirs = None, **kwargs):
        """analyzes a set of N files each with M masks, totalling
        NxM image objects
        
        maskdirs is the directory containing the masks. It can have values:
            'automatic':    Each ptufile gets an own set of masks
                            the names are automatically inferred from the names
                            of the ptufiles. E.g. 
                            cell1Masks, cell2Masks 
                            for cell1.ptu, cell2.ptu
            other strings:  One set of masks is applied to all files. 
                            The string is the directory containing the masks.
                            
        time in seconds
        
        analyzes all ptu files in a directory into GRY image objects

        kwargs:
        isSave:         if True, TAC decays and tiff images are stored to
                        self.TACdir and self.imdir
        isCleanImage:   if True, 3D lifetime arrays are discarded, freeing
                        up space.
        threshold:      all pixels with an intensity value lower than
                        threshold are set to 0."""
        #expand maskdirs into list maskdirs
        #masks are specified in dirs according to naming convention
        if maskdirs == 'automatic':
            maskdirs = [os.path.join( self.wdir, ptufile[:-4] + '_Masks') 
                for ptufile in self.ptufiles]
        elif type(maskdirs) == str: #same mask for all files
            maskdirs = [maskdirs] * len(self.ptufiles)
        elif maskdirs == None:
            maskdirs == [maskdirs] * len(self.ptufiles)
        elif maskdirs == None:
            pass
        else: raise ValueError
            
        if self.timeLst == None:#dummy timeList
            self.timeLst = [None]*len(self.ptufiles)
        
        if maskdirs == None: #no usermask
            for ptufile, time in zip(self.ptufiles, self.timeLst):
                    self.analyzeFile(ptufile, time = time, **kwargs)
        else: #each ptufile has one or multipe masks
            for ptufile, maskdir, time in \
                    zip(self.ptufiles, maskdirs, self.timeLst):
                maskfiles = [os.path.join(maskdir, file)\
                    for file in os.listdir(maskdir) if file.endswith('tif')]
                self.analyzeFile(ptufile, maskfiles, time, **kwargs)
                
        self.genImstatsdf(**kwargs)
        
    def analyzeFile(self, ptufile, maskfiles = None, time = None, isCleanImage = True,
        **kwargs):
        ffile = os.path.join(self.wdir, ptufile).encode()
        ltImage = LtImage.fromFileName(ffile, 
                                       self.line_setup, 
                                       self.ntacs)
        ltImage.genImageIndex()
        #loop over masks, default to one iteration with mask = None if no mask is present.
        ptuname = os.path.splitext(os.path.split(ptufile)[1])[0]
        if maskfiles == None:
            #copy parameters
            imChannelLst = copy.deepcopy(self.templateImChannelLst)
            self.procesImChannelLst(imChannelLst, ltImage, ptuname, time = time, 
                **kwargs)
        else:
            for maskfile in maskfiles:
                imChannelLst = copy.deepcopy(self.templateImChannelLst)
                mask = getMask(maskfile)
                maskid = os.path.splitext(os.path.split(maskfile)[1])[0]
                self.procesImChannelLst(imChannelLst, ltImage, ptuname, 
                    usermask = mask, 
                    time = time, maskid = maskid, **kwargs)
        if isCleanImage:
            ltImage.cleanExpensiveArrays()

    def procesImChannelLst(self, imChannelLst, ltImage, ptuname, intensityThreshold = None, 
        isSave = True, isCleanImage = True, usermask = None, maskid = None, 
        time = None, **kwargs):
        """
        intensityThreshold (int): 
            if given, sum of all images is calculated and the pixels below the
            threshold are discarded.
        usermask: if given, this mask is applied to the data
        """
        #construct 3D ltimage for all channels
        for channel in imChannelLst:
            channel.genltImage(ltImage)
        #construct intensity mask if given
        if intensityThreshold is not None:
            for channel in imChannelLst:
                channel.genIntensity()
            #the intensity mask is determined based on workIntensity
            intMask = buildMaskFromIntensityThreshold(
                    imChannelLst, 
                    intensityThreshold)
            #the masking happens on the 3D lifetime arrays
            for channel in imChannelLst:
                channel.ltMask(intMask)
            
        if usermask is not None:
            for channel in imChannelLst:
                channel.ltMask(usermask)
        #calculate 2D image and decay from masked ltImage
        for channel in imChannelLst:
            channel.genIntensity()
            channel.genDecay()
        #the TAC decays are determined from the worklifetime image
        self.genDerivedDecaysfromChannelLst(imChannelLst)
        if isSave:
            saveTACs(imChannelLst, self.TACdir, ptuname)
            for channel in imChannelLst:
                channel.saveIntensityToTiff(self.imdir, ptuname)
        if isCleanImage: #free memory intensive 3D array
            [channel.clean3Darray() for channel in imChannelLst]
        gc.collect()
        self.CELFISsampleLst.append(CELFISsample(ltImage, imChannelLst, ptuname, maskid, time))

    def genDerivedDecaysfromChannelLst (self, imChannelLst):
        PSindices = getPSindices(imChannelLst)
        for channel in imChannelLst:
            assert hasattr(channel, 'decay'), \
                'channel object must have decay property'
        #add VM and r variables
        for Pindex, Sindex in PSindices:
            PS = PandS2PS(imChannelLst[Pindex].decay, imChannelLst[Sindex].decay)
            VM, r = bp.genFr(PS, self.g_factor, shift = self.PSshift)
            #choice: store these extra properties under the 'P' decay
            imChannelLst[Pindex].PS = PS
            imChannelLst[Pindex].VM = VM
            imChannelLst[Pindex].r = r

    def genImstatsdf(self):
        df = pd.DataFrame()
        for i, sample in enumerate(self.CELFISsampleLst):
            fullname = sample.ptuname + sample.maskid
            df.at[fullname, "ptuname"] = sample.ptuname
            df.at[fullname, "maskid"] = sample.maskid
            df.at[fullname, "time"] = sample.time
            #cell phenotype
            df.at[fullname, "sampleType"] = self.sampleType
            df.at[fullname, "sampleAddition"] = self.sampleAddition
            df.at[fullname, "experimentId"] = self.experimentId
            df.at[fullname, "roundId"] = self.roundId
            for channel in sample.imChannelLst:
                Nphotons = np.sum(channel.decay)
                surface = np.sum(channel.intImage > 0)
                df.at[fullname, "N_"+channel.name] = Nphotons
                df.at[fullname, "surface_" + channel.name] = surface
            #calculate sum of channels for PS pairs
            PSindices = getPSindices(sample.imChannelLst)
            for Pindex, Sindex in PSindices:
                label = sample.imChannelLst[Pindex].name[:-1]
                Ntot = np.sum(sample.imChannelLst[Pindex].decay + sample.imChannelLst[Sindex].decay)
                df.at[fullname, "N_" + label + "tot"] = Ntot
            if self.cell_phenotypeLst is not None:
                df.at[fullname, 'cell_phenotype'] = self.cell_phenotypeLst[i]
        try:
            #add normalized countrates (note this depends partially on certain channel names)
            integrationtime  = self.Nframes * self.CELFISsampleLst[0].ltImage.dwelltime
            calculateRatesFromDf(df, integrationtime, self.powerD, self.powerA)
            addConc(df, self.Acf, self.BnormDonor, self.BnormAcceptor, 
                self.pMembrane, self.maturationD, self.maturationA)
        except KeyError as e:
            print("Warning: could not calculate all derived variables" + 
                "\nSome columns missing. Original Error message:")
            print(e)
        #save
        outname = os.path.join(self.resdir, self.experimentId + 'imstats.csv')
        trySaveToCSV(df, outname)
        self.imstats = df
        return 0
        
    def mergeStatsAndProcess(self, tauxD0 = 2.3):
        self.mergedstats = self.imstats
        for attr in ['D0DA1ltdfrm', 'D0DA2ltdfrm', 'fit2ltdfrm']:
            if hasattr(self, attr):
                self.mergedstats = pd.concat([self.mergedstats, getattr(self, attr)], axis = 1)
        addCorrProxFret(self.mergedstats, self.kprox)
        addESpecies(self.mergedstats, tauxD0)

    def batchFit1ltD0DA(self,
                      D0dat = None,
                      fitrange = (25, 380),
                      channelId = 0,
                      decaytype = 'decay',
                      **kwargs):
        """makes simple Donor Only calibrated Donor Acceptor fits
        """
         #ugly workaround
        assert D0dat is not None, 'must give a Donor only decay'
        #read all DA decays
        DATACs = self.getImChannelProperty(decaytype, channelId)
        fullnames = [sample.ptuname + sample.maskid for sample in self.CELFISsampleLst]

        dfrm = pd.DataFrame()
        #fit and plot DA
        plotout = os.path.join(self.resdir, self.experimentId + 'D0DA1ltplots')
        trymkdir(plotout)
        
        D0snip = D0dat[fitrange[0]:fitrange[1]]
        _, _, _, Donlymodel, chi2red_D0 = fitDA.fitDonly(D0snip, self.dt_glob)
        for name, DATAC in zip(fullnames, DATACs):
            DAsnip = DATAC[fitrange[0]:fitrange[1]]
            popt, pcov, DAmodel, chi2red = \
                fitDA.fitDA1lt (DAsnip, D0snip, self.dt_glob)
            fitDA.pltDA_eps(DAsnip, D0snip, DAmodel, Donlymodel, name, popt,
                            chi2red, chi2red_D0, plotout, **kwargs)
            dfrm.at[name, 'xFRET'] = 1-popt[1]
            dfrm.at[name, 'kFRET'] = popt[2]
            dfrm.at[name, 'chi2red'] = chi2red
        outname = os.path.join(self.resdir, self.experimentId + 'D0DAFitData.csv')
        dfrm.columns = dfrm.columns.values + "D0DA"
        dfrm.to_csv(outname)
        self.D0DA1ltdfrm = dfrm
        return dfrm

    def batchFit2ltD0DA(self,
                      D0dat = None,
                      fitrange = (25, 380),
                      channelId = 0,
                      decaytype = 'decay',
                      **kwargs):
        """makes Donor Only calibrated (2lt) Donor Acceptor (2lt) fits
        """
        #ugly workaround
        assert D0dat is not None, 'must give a Donor only decay'
        #TODO split tauf, taux, E calculation in generic function
        #init and get data from object
        DATACs = self.getImChannelProperty(decaytype, channelId)
        fullnames = [sample.ptuname + sample.maskid for sample in self.CELFISsampleLst]

        #get fullnames, not name, then for all functions
        pnames = ['A_DA', 'xFRET1', 'xFRET2', 'kFRET1', 'kFRET2', 'bg']
        dfrm = pd.DataFrame()
        #fit and plot DA

        plotout = os.path.join(self.resdir, self.experimentId + 'D0DA2ltplots')
        trymkdir(plotout)
        D0snip = D0dat[fitrange[0]:fitrange[1]]
        poptD0, _, _, Donlymodel, chi2red_D0 = fitDA.fitDonly(D0snip, \
            self.dt_glob)
        x1, x2, tau1, tau2, _ = poptD0
        x1, x2 = [x1 / (x1 + x2), x2 / (x1 + x2)]
        k1, k2 = [1/tau1, 1 / tau2]
        tauxD0 = x1 * tau1 + x2 * tau2
        for name, DATAC in zip(fullnames, DATACs):
            DAsnip = DATAC[fitrange[0]:fitrange[1]]
            popt, pcov, DAmodel, chi2red = \
                fitDA.fitDA2lt (DAsnip, D0snip, self.dt_glob)
            fitDA.pltDA_eps(DAsnip, D0snip, DAmodel, Donlymodel, name, popt,
                            chi2red, chi2red_D0, plotout, **kwargs)
            for p, pname in zip (popt, pnames):
                dfrm.at[name, pname] = p
            dfrm.at[name, 'chi2red'] = chi2red
            #     | kDA1  kDA2
            #___________________
            #kDO1 | x11   x12
            #kDO2 | x21   x22
            #tau_ij = (kD0i + kDAj)^-1
            #sum over all species species / fluorescence weighted
            #tau_x = SUMIJ xij * tau_ij
            taux = 0
            tauf = 0
            for xDA, kDA in zip(popt[[1,2]], popt[[3,4]]):
                for xD0, kD0 in zip ([x1, x2], [k1, k2]):
                    taux += xDA * xD0 * (1 / (kDA + kD0))
                    tauf += xDA * xD0 * (1 / (kDA + kD0))**2
            tauf = tauf / taux
            dfrm.at[name, 'taux'] = taux
            dfrm.at[name, 'tauf'] = tauf
            dfrm.at[name, 'E'] = 1-taux / tauxD0

        outname = os.path.join(self.resdir, self.experimentId + 'D0DAFitData.csv')
        dfrm.columns = dfrm.columns.values + "DODA2lt"
        dfrm.to_csv(outname)
        self.D0DA2ltdfrm = dfrm
        return dfrm

    def batchFit2lt(self,
                    fitrange = (20, 380),
                      channelId = 0,
                      decaytype = 'decay',
                    **kwargs):
        """batch fit D0 data assuming two lifetimes
        commonly for D0"""
        #**kwargs can take arguments that are not used
        #TODO split tauf, taux, E calculation in generic function
        pnames = ['x0', 'x1', 'tau0', 'tau1', 'bg']
        fullnames = [sample.ptuname + sample.maskid for sample in self.CELFISsampleLst]
        dfrm = pd.DataFrame()
        plotout = os.path.join(self.resdir, self.experimentId + 'D02ltplots')
        trymkdir(plotout)
        #fit all data
        TACs = self.getImChannelProperty(decaytype, channelId)
        assert len(TACs) != 0, 'TACs is empty'
        for TAC, name in zip(TACs, fullnames):
            D0snip = TAC[fitrange[0]:fitrange[1]]
            popt, _, _, Donlymodel, chi2red = fitDA.fitDonly(D0snip, \
                self.dt_glob)
            fitDA.pltD0(D0snip, Donlymodel, name, plotout, dtime = self.dt_glob)
            #fill dataframe row
            for pname, p in zip(pnames, popt):
                dfrm.at[name, pname] = p
            x0, x1, tau0, tau1, bg = popt
            #calc derived vars
            tauf = (x0 * tau0**2 + x1 * tau1**2) / (x0 * tau0 + x1 * tau1)
            taux = (x0 * tau0 + x1 * tau1) / (x0 + x1)
            dfrm.at[name, 'tauf'] = tauf
            dfrm.at[name, 'taux'] = taux
            dfrm.at[name, 'chi2red'] = chi2red
            print('finished fitting with 2lt set %s' % name)
        outname = os.path.join(self.resdir, self.experimentId + '2ltFitData.csv')
        dfrm.columns = dfrm.columns.values + "2lt"
        dfrm.to_csv(outname)
        self.fit2ltdfrm = dfrm
        return dfrm

    def getImChannelProperty(self, property, channelId):
        List = [getattr(sample.imChannelLst[channelId], property) \
            for sample in self.CELFISsampleLst]
        return List
####################functions to aid generating masks###########################

def tryMakeMaskDirs(wdir):
    #list all ptu files in folder
    imagefiles = [file for file in os.listdir(wdir)\
                  if file.endswith('.ptu')]
    for file in imagefiles:
        #generate a directory for manually placing masks
        maskdir = file[:-4] + '_Masks'
        trymkdir(os.path.join(wdir, maskdir))
def createImagesForMasking(wdir, imChannel, line_setup = [1,2], ntacs = 1024):
    #list all ptu files in folder
    imagefiles = [file for file in os.listdir(wdir)\
                  if file.endswith('.ptu')]
    #make a target directory for saving images for masking
    imagesoutdir = os.path.join(wdir, 'imagesForMasking')
    trymkdir(imagesoutdir)
    for file in imagefiles:
        #export an image for creating masks
        ffile = os.path.join(wdir, file)
        ltImage = LtImage.fromFileName(ffile.encode(), line_setup, ntacs)
        ltImage.genImageIndex()
        imChannel.genltImage(ltImage)
        imChannel.genIntensity()
        preposition = file[:-4]
        imChannel.saveIntensityToTiff(imagesoutdir, preposition = preposition)

def saveNpAsImage(array, outname):
    image = Image.fromarray(array)
    image.save(outname)

def createSeriesHiLoMasks(seriesdir, marker = 'imG.tif'):
    """assumes an existing structure of masks and cell images.
    Finds each pair of mask and cellimg from pre-existing file structure
    calls function that works to create additional masks
    Saves these additional masks to disc as .tiff"""
    #get the names of the tiff 
    cellimgdir = 'imagesForMasking'
    cellimgfilesp = os.listdir(os.path.join(seriesdir, cellimgdir))
    cellimgfiles = [file for file in cellimgfilesp if file.endswith(marker)]
    
    #find the Masks folders
    for entry in os.listdir(seriesdir):
        if entry.endswith('_Masks'):
            #get the tiff file that works with this folder
            basename = entry[:-6]
            for cellimgfile in cellimgfiles:
                if (cellimgfile[:-len(marker)] == basename):
                    fcellname = os.path.join(seriesdir, cellimgdir, cellimgfile)
                    cellarr = np.array(Image.open(fcellname))
                    #there is a legacy bug that where some values are 
                    # truncated in 8 bit tiffs: check for clipping of data
                    maxval = max(cellarr.flatten())
                    if maxval == 255 or maxval == 2**16-1:
                        with warnings.catch_warnings():
                            warnings.simplefilter("always")
                            warnings.warn('saturation detected for %s' % cellimgfile, )
                    break
            #get the masks 
            maskfiles = os.listdir(os.path.join(seriesdir, entry))
            maskfiles = [file for file in maskfiles if file.endswith('.tif')]
            maskfiles = [file for file in maskfiles if 
                         not (file.endswith('_lo.tif') or file.endswith('_hi.tif'))]
            maskffiles = [os.path.join(seriesdir, entry, maskfile) 
                          for maskfile in maskfiles]
            for maskffile in maskffiles:
                maskarr = np.array(Image.open(maskffile))
                #make sure the cell area is 1, other values are 0
                maskarr = maskarr == 0
                #function that does the calculation
                himask, lomask = createHiLoMasks(cellarr, maskarr)
                hiOutname = maskffile[:-4] + '_hi.tif'
                loOutname = maskffile[:-4] + '_lo.tif'
                saveNpAsImage(himask, hiOutname)
                saveNpAsImage(lomask, loOutname)
    print('done generating HiLo masks')
                
def deleteHiLoMasks(seriesdir):
    """in '_Masks' subfolders delete all .tiff files ending on '_hi.tif' or 
    '_lo.tif'.
    """
    #find the Masks folders
    for entry in os.listdir(seriesdir):
        fsubdir = os.path.join(seriesdir, entry)
        if entry.endswith('_Masks'):
            for file in os.listdir(fsubdir):
                if file.endswith('_hi.tif') or file.endswith('_lo.tif'):
                    print('deleting mask %s' % file)
                    ffile = os.path.join(fsubdir, file)
                    os.remove(ffile)


def createHiLoMasks(cellarr, maskarr, verbose = False):
    #smooth against shot noise
    cellarr = gaussian_filter(cellarr, 1, output = float)
    #apply mask
    maskedcell = cellarr * maskarr
    total = np.sum(maskedcell)
    #make an ordering of all smoothed values
    sortedcell = np.sort(maskedcell.flatten())
    cumsum = np.cumsum(sortedcell)
    spliti = np.argmax(cumsum > total / 2)
    #get the threshold value
    splitval = sortedcell[spliti]
    lomask = np.logical_and(maskedcell > 0, maskedcell < splitval)
    himask = maskedcell >= splitval
    if verbose:
        #checks
        lowersum = np.sum(maskedcell * lomask)
        uppersum = np.sum(maskedcell * himask)
        print('lower is up to %.2f\nlowersum is %i\nuppersum is %i' %
              (splitval, lowersum, uppersum))
        print('total sum check: %r' % (np.isclose(lowersum + uppersum, total)))
    return himask, lomask