#file generated on 18 Jan 2023by Nicolaas van der Voort
#this pure-python based file creates 3D lifetime images from ptu files
#it supercedes a previous c only interpretation
#it supports having also P and S channels
import numpy as np
import cpp_wrappers
import os
import aid_functions as aid
from PIL import Image

class LtImage:
    """"this class is used to cluster options needed for reading ptu files into images"""
    #change init to also include fromFileName loading tac, t, can eventN
    def __init__(self,
            line_setup,
            dwelltime,
            counttime,
            numRecords,
            dimX,
            dimY,
            ntacs,
            TAC_range,
            eventN = None,
            tac = None,
            t = None,
            can = None):
        self.line_setup = line_setup
        self.dwelltime = dwelltime
        self.counttime = counttime
        self.numRecords = numRecords
        self.dimX = dimX
        self.dimY = dimY
        self.ntacs = ntacs
        self.TAC_range = TAC_range
        self.eventN = eventN
        self.tac = tac
        self.t = t
        self.can = can
        self.channels = []
        self.imsize = dimX*dimY*ntacs
        #Also consider adding a list of Imchannels to this class
        #This makes it all pretty comparable to the ImageManipulation class. 
        #maybe I can merge the two to simplify.
    @classmethod
    def fromFileName(cls, fname, line_setup, ntacs):
        assert type(fname) == bytes, "fname must be bytes"
        numRecords = cpp_wrappers.ptuHeader_wrap (fname)
        print('number of records is ' + str(numRecords))
        eventN, tac, t, can = cpp_wrappers.ptu_wrap(fname, numRecords)
        fdir, file = os.path.split(fname)
        name, _ = os.path.splitext(file)
        header_name = os.path.join(fdir, b"header", name + b".txt")
        dimX, dimY, dwelltime, counttime, TAC_range = cpp_wrappers.read_header_v2(header_name)
        return cls(line_setup,
                   dwelltime,
                   counttime,
                   numRecords,
                   dimX,
                   dimY,
                   ntacs,
                   TAC_range,
                   eventN,
                   tac,
                   t,
                   can)
   #add function to reload eventN, t, can, tac with saved parameters again from disc.
    def genImageIndex(self):
        """
        finds location of line and frame markers
        and calculates from that the index in the flattened ltImage array of 
        each photon

        Returns
        -------
        None.

        """
        #timeit estimate for 52750671 NumRecords: 200 ms
        self.startlineids = np.where(self.can == 65)[0]
        self.stoplineids = np.where(self.can == 66)[0]
        frameids = np.where(self.can == 68)[0]
        self.frameids = np.append(frameids, 2**63-1) #add missing close frame marker
        #lineids must be smaller than 2**15
        lineIndex = np.full(self.numRecords, -1, np.short)
        pxIndex = np.full(self.numRecords, -1, np.short)
        self.lineIds = np.full(self.numRecords, 0, np.byte)
        #timeit estimate for 800x800x2x21 and 52750671 NumRecords: 370 ms
        frameN = 0
        lineN = 0
        lineId = 0
        pxN = 0

        #use inverse value to maintain int
        pixel2macrot = int(self.dwelltime /self. counttime)
        #for loops are slow in python, but the number of entries is limited, 
        #so this is ok
        for startlineid, stoplineid in zip(self.startlineids, self.stoplineids):
            if startlineid > self.frameids[frameN]:
                frameN = frameN+1
                lineN = 0
            linestartt = self.t[startlineid]

            lineIndex[startlineid:stoplineid] = lineN
            self.lineIds[startlineid:stoplineid] = self.line_setup[lineId]
            #time consuming because of calculation
            pxIndex[startlineid:stoplineid] = \
                (self.t[startlineid:stoplineid]-self.t[startlineid])/pixel2macrot
            
            lineId = lineId + 1
            if lineId == len(self.line_setup):
                lineId = 0
                lineN = lineN+1
        #also time consuming      
        rebin = self.TAC_range / self.ntacs
        tacIndex = (self.tac / rebin).astype(np.short)
        #astype np.int64 needed to cast intermediate results to int64 rather than shorts.
        index = lineIndex * np.array([self.dimX * self.ntacs]).astype(np.int64) \
                + pxIndex * np.array([self.ntacs]).astype(np.int64) \
                + tacIndex
        index[index > self.imsize] = -1
        self.index = index
    def cleanExpensiveArrays(self):
        del self.tac
        del self.t
        del self.can
        del self.eventN

class ImChannel:
    """
    this class describes the properties of the channel of an image and some function
    """
    def __init__(self,
            name,
            can_lst,
            tacmin,
            tacmax,
            line_id,
            tmin = 0,
            tmax = 2**63-1):
        self.name = name
        self.can_lst = can_lst
        self.tacmin = tacmin
        self.tacmax = tacmax
        self.line_id = line_id
        self.tmin = tmin
        self.tmax = tmax
    
    def genltImage(self, ltImage):
        #below four bool arrays: 200ms:  all for 800x800x2x21 and 52750671 NumRecords: 370 ms, can = 0 (less populated)
        inmode = np.isin(ltImage.lineIds, self.line_id) #do some smart parallel calculation to calculate mode
        intac = np.logical_and(ltImage.tac<self.tacmax, ltImage.tac > self.tacmin)
        in_t = np.logical_and(ltImage.t < self.tmax, ltImage.t > self.tmin) #can get rid of for computational time, but is nice to have
        incan = np.isin(ltImage.can, self.can_lst)
        useph = np.logical_and(np.logical_and(np.logical_and(inmode, intac), in_t), incan)#60ms
        chindex = ltImage.index[useph]#80ms
        ltImage = np.full(ltImage.dimX * ltImage.dimY * ltImage.ntacs, 0, np.byte)
        np.add.at(ltImage, chindex, 1)#700ms
        self.ltImage = ltImage.reshape((800,800,1024)) #fast
        
    def gate(self, mingate, maxgate):
        self.ltImage[:,:,:mingate] = 0
        self.ltImage[:,:,maxgate:] = 0
    def ltMask(self, mask):
        self.ltImage = self.ltImage * mask[...,None]
    def intMask(self, mask):
        self.intImage = self.intImage * mask
    def genIntensity(self):
        self.intImage = self.ltImage.sum(axis = 2)
    def genDecay(self):
        self.decay = self.ltImage.sum(axis = (0,1))
    def saveIntensityToTiff(self, outfolder, preposition = '', 
        xmin = 0, xmax = None, ymin = 0, ymax = None): 
        """function takes GRY object and saves to outfolder in tif format in 32bit float .tif format
            preposition allows adding identifier to default filenames.
            xmin. xmax, ymin, ymax allow saving snip of np array"""
        assert type(outfolder) == str and type(preposition) == str,\
            "outfolder and preposition must be string type"
        #convert all strings to bytes
        aid.trymkdir(outfolder)
        im = Image.fromarray(self.intImage[xmin:xmax, ymin:ymax].astype(np.uint16))
        outname = os.path.join(outfolder, preposition + self.name + '.tif')
        im.save(outname)
    def clean3Darray(self):
        del self.ltImage
    
