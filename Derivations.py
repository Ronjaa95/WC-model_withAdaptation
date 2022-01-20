#import necessary packages
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.axes as ax
import pandas as pd
import numpy as np
import glob
import os
import math
import sys

import scipy
from scipy import signal
from scipy.stats import binned_statistic


from neurolib.models.wc-adap import WCModel

from FixedPoints import FixedPoints
from Bistability import Bistability

from neurolib.utils.parameterSpace import ParameterSpace
from neurolib.optimize.exploration import BoxSearch
from neurolib.utils.signal import Signal

class Derivations:
    
    
    def __init__(self, model=WCModel(), model_input=WCModel_input(), params=None):
        
        self.model = model
        self.model_input = model_input
        
        self.params = params
        
        if params is not None:
            for k, v in zip(params.keys(), params.values()):
                assert k in self.model.params, "param not in model params"
                self.model.params[k] = v
                
        if self.model.params.N == 1:
            self.OneDim = True
        else:
            self.OneDim = False
                
         
        
        
    
    # # # - - - # # # - - - Functions needed for analysis - - - # # # - - - # # #
        
        
        
    def getDFT(self, x):
        
        """ Computes the Discrete Fourier Transformation of input x. Uses either np.fft.fft(x) for one-dimensional 
            x or np.fft.nfft(x) for multi-dimensional x tracks.
            INPUT:
            :x: time series of periodic behaviour
            OUTPUT:
            :f_trafo: DFT in the form of [y(0), y(1), ..., y(len(x))] (1d) or [[DFT(x[0])],[DFT(x[1])],...[DFT(x[len(x)])]] (multi-dim)
         """
        
        if np.array(x).ndim == 1:
            f_trafo = np.fft.rfft(x)
        elif np.array(x).ndim > 1:
            f_trafo = np.fft.rfftn(x)
        
        return f_trafo
    
    
    def getStatesInvolvementDistribution(self, states, involvement, nbins=10):
        """"written by https://github.com/caglorithm"""
        invs = {0: [], 1: []}
        lens = {0: [], 1: []}
        for s in states[:]:
                lengths = self.getStateLengths(s)
                current_idx = 0
                for state, length in lengths:
                    state = int(state)
                    # compute average involvement during this state
                    mean_involvement = np.mean(involvement[current_idx : current_idx + length])
                    lens[state].append(length * self.model.params.dt)  # convert to milliseconds
                    invs[state].append(mean_involvement)

                    current_idx += length


        up_bin_means, bin_edges, _ = binned_statistic(invs[1], lens[1], bins=nbins, range=(0, 1))
        down_bin_means, bin_edges, _ = binned_statistic(invs[0], lens[0], bins=nbins, range=(0, 1))
        
        return up_bin_means, down_bin_means, bin_edges
    
    
    def getInvolvement(self, states):
        """Returns involvement (=fraction of nodes participating) in down-states.
        
        INPUT:
        :param states: state array (numpy array: NxT) of up- (=1), and down- (=0) states np.array([[0,0,0,0,1,1,1,1,0,0...],...])
        
        OUTPUT:
        :involvement: Involvement time series (numpy array: 1xT) that shows the fraction of nodes participating in down per time step
        """
        
        up_involvement = np.sum(states, axis=0) / states.shape[0]
        return 1 - up_involvement
        


        
    def getUpDownWindows(self, x, thresh=0.2, filter_long=True, dur=25):
        """
        Function returns an array of 0s and 1s, that indicate, where the system is either in the 
        up-state (>0.25) or down-state (<=0.25). The 
        integers are collected in order.
            :x: = array (time series) of activity pattern
        OUTPUT:
            :up_down: array of states [1,1,1,0,0,0,0,0,1,1,1,...]
        """
        
        
        
        if all(i >= thresh for i in x):
            up_down = np.ones(len(x))
        elif all(i < thresh for i in x):
            up_down = np.zeros(len(x))
        else:
            up_down = (x >= thresh).astype(int)
            
            if filter_long:
                up_down = self.filter_long_states(up_down, dur)
            
        return up_down   
    
    
    def filter_long_states(self, states, dur=25):
        """written by https://github.com/caglorithm"""
        states = states.copy()
        LENGTH_THRESHOLD = int(dur / self.model.params["dt"])  # ms
       # for s in states:
        lengths = self.getStateLengths(states)
        current_idx = 0
        last_long_state = lengths[0][0]  # first state
        for state, length in lengths:
            if length >= LENGTH_THRESHOLD:
                last_long_state = state
            else:
                states[current_idx : current_idx + length] = last_long_state
            current_idx += length
        return states

        
    
    def getStateLengths(self, up_down):
        
        """
        Derive the state lengths of up (1) and down (0) states, by counting the number of integration steps, and grouping them per 
        state into tuples of (state, #time steps).
        Example: getStateLengths([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0])
        Returns: [(0, 4), (1, 4), (0, 2), (1, 1), (0, 1)]
        INPUT:
        :up_down: Input list of up-down states, succesively
        OUTPUT:
        :stateLengths: List of (state, length) tuples
        """
        
        import itertools
        
        if np.array(up_down).ndim == 1:
            stateLengths = [(x, len(list(y))) for x, y in itertools.groupby(up_down)]
        elif np.array(up_down).ndim > 1:
            stateLengths = [self.getStateLengths(xss) for xss in up_down]
            
        return stateLengths
    
    
    
    def getDurations(self, stateLengths):
        """
        Returns the durations of the up- (>0.2), and down-states.
            :stateLengths: list of tuples [(state, duration),...] (e.g. [(0,12), (1,5), (0,8), (1,23)] with 0=down, 1=up)
        OUTPUT:
            :[up, down]: two dimensional array of up-, and down-state durations per state [in ms*dt].
        """
        
        up = [stateLengths[j][1] for j in range(len(stateLengths)) if stateLengths[j][0]==1]
        down = [stateLengths[j][1] for j in range(len(stateLengths)) if stateLengths[j][0]==0]
        
        return [up, down]
    
    
    
    def getDurationsNdim(self, stateLengths):
        """
        Compute the Inter-Event-Intervals, returns a list of IEIs.
            :stateLengths: list of tuples [(state, duration),...] (e.g. [(0,12), (1,5), (0,8), (1,23)] with 0=down, 1=up)
        OUTPUT:
            :[up_durations, down_durations]: two dimensional array of up-, and down-state durations [in ms*dt] per state, per node.
        """
        
      #  up_durations = [self.getDurations(xss)[0] for xss in stateLengths] #entpricht UP-duration
      #  down_durations = [self.getDurations(xss)[1] for xss in stateLengths] #entspricht DOWN-duration
         
        up_durations = []
        down_durations = []
        for i, l in enumerate(stateLengths):
            up_durations.append([u[1] for u in l if u[0]==1]) #entpricht UP-duration
            down_durations.append([u[1] for u in l if u[0]==0]) #entspricht DOWN-duration
            
        
        return [up_durations, down_durations]
   
    
    def getIEIs_up_to_down(self, stateLengths):
        
        IEIs = [(stateLengths[j+1][1]+stateLengths[j][1])*0.0001 for j in range(1, len(stateLengths)-1) 
                if stateLengths[j-1][0]==1 and stateLengths[j][0]==0 and stateLengths[j+1][0]==1]
        
        return IEIs
    
       
    
    def getIEIs_down_to_up(self, stateLengths):
        
        IEIs = [(stateLengths[j+1][1]+stateLengths[j][1])*0.0001 for j in range(1, len(stateLengths)-1) 
                if stateLengths[j-1][0]==0 and stateLengths[j][0]==1 and stateLengths[j+1][0]==0]
        
        return IEIs
    
    
    
    
    def getMean(self, array):
        """
        Computes the mean value of an array.
        
        INPUT:
        :array: array of values, e.g. np.array([100, 110, 90])
        OUTPUT:
        :mean: float of mean value of array, e.g. 100.0
        """
       
        mean = (1/len(array)) * sum(array)
        
        return mean
    
    def getStandardDeviation(self, liste, mean):
        """
        Computes the standard deviation of a list of values.
        
        INPUT:
        :liste: list of values, e.g. [100, 110, 90]
        :mean: the corresponding mean as float of the values of the list, e.g. 100.0
        OUTPUT:
        :std_dev: float of standard deviation of list, e.g. 10.0
        """
        
        
        squared_std_deviation = (1/len(liste)) * sum((liste-mean)**2)
        
        std_dev =  np.sqrt(squared_std_deviation)
        
        return std_dev
        
    
    def getCV(self, std_dev, mean):
        """
        Computes the Coefficient of Variation for the input values
        
        INPUT:
        :std_dev: standard deviation of the array, we're looking at e.g. 10.0
        :mean: mean value of the array, we're looking at e.g. 100
        OUPUT:
        :cv: float value of coefficient of variation, e.g. 0.1
        """

        cv = std_dev/mean
        
        return cv 
        
        
    def checkOneOsc(self, x):
        """
            For the input activity, derive whether oscillations happen.
            
            INPUT:
            :x: = 1-dimensional activity track
            OUTPUT:
            :oss: bool (True if oscillations [max(x) - min(x) > 0.1] happen, False otherwise)
            
            The value 0.1 could be even smaller, while we investigate the deterministic system, since that should not leave a FP.
        """
        
        if max(x)-min(x)>0.1:
            oss=True
        else:
            oss=False

        return oss
    
    def checkMultiOsc(self, x):
        oss = [self.checkOneOsc(xss) for xss in x]
        return oss
    
   # # # # - - - -  DataFrame COMPUTATIONS - Diese Berechnungen wurde neu aufgelegt und dienen nur als template - - - - # # # #
# # # # Sie sind allerdings korrigiert worden, aber nicht nochmal gelaufen. Dementsprechend könnten Tippfehler im Code sein. # # # #
    
    def deriveMulti(self, traj): 
        
        
        paras = self.search.getParametersFromTraj(traj)
    
        self.model.params['exc_ext'] = paras['exc_ext']
        self.model.params['inh_ext'] = paras['inh_ext']
        
        bi=Bistability(model=self.model_input, 
                       ext_input=[self.model.params['exc_ext'], self.model.params['inh_ext']], 
                       fix_params=self.model.params)
        bi.testBistability()
        bi_array = bi.bistable
        if sum(bi_array)>((3*self.model.params.N)/4):
            bistab=True
        else:
            bistab=False
        
        self.model.run()
        
        max_exc = np.max(self.model.exc[:, -int(5000/self.model.params['dt']):])
        max_inh = np.max(self.model.inh[:, -int(5000/self.model.params['dt']):])
        
        exc_act = self.model.exc
        inh_act = self.model.inh
        adap_act = self.model.adap
        
        duration = self.model.params.duration/self.model.params.dt
        cut_off = int(duration - 60*10000)
        
        x = exc_act[:, -cut_off:]
        x_adap = adap_act[:, -cut_off:]
        oss = self..checkMultiOsc(x)
        if any(oss):
            oss_general = True
        else: 
            oss_general=False
        
        LCEI = 0
        LCaE = 0
        
        if oss_general:
            dominant_frequencies = []
            adap_frequ = []
            
            frequencies, psd =  signal.welch(x, 10000, window='hanning', nperseg=int(60000), scaling='spectrum')
            
            
            idx_dominant_frequ = np.argmax(np.sum(psd,axis=0)) 
            domfrequ = frequencies[idx_dominant_frequ]
            
            
            frequencies, power_spectrum =  signal.welch(x_adap, 10000, window='hanning', nperseg=int(60000), scaling='spectrum')
            
            adap_ps_at_most_powerful_frequ = power_spectrum[:,idx_dominant_frequ]
            adap_amp = np.sqrt(max(adap_ps_at_most_powerful_frequ))
    
            if adap_amp < 0.004:
                LCEI = 1.0 #Excitation-Inhibition Limit Cycle (LCEI)
            else:
                LCaE = 1.0 #adaptation-driven Limit Cycle (LCaE)
            
        else:
            domfrequ = False
            adap_amp = 0
            
            
        result = {'max_exc': max_exc,
                  'min_exc': min_exc,
                  'oss': oss_general,
                  'dominant_frequency': domfrequ,
                  'adaptation_amplitude': adap_amp,
                  'bistability': bistab,
                  'LCEI': LCEI,
                  'LCaE': LCaE,}
        
        #safe result dictionary to a table in a hdf-file in the directory 'traj'
        self.search.saveToPypet(result, traj)
        
    
    
    def deriveOne(self, traj):
        paras = self.search.getParametersFromTraj(traj)
    
        self.model.params['exc_ext'] = paras['exc_ext']
        self.model.params['inh_ext'] = paras['inh_ext']
        
        fp = FixedPoints(self.model, 
                         ext_input=[self.model.params['exc_ext'], self.model.params['inh_ext']], 
                         fix_params=self.model.params)
        fp.runAll()
        f_pts = fp.fixed_points
        stab = fp.stability
        bistab = fp.bistable
        fixed_points = np.array(f_pts)
        stabi = np.array(stab)
        
        
        if bistab:
            self.model.params['exc_init'] =  f_pts[0][0] * np.random.uniform(1, 1, (self.model.params.N, 1))
            self.model.params['inh_init'] =  f_pts[0][1] * np.random.uniform(1, 1, (self.model.params.N, 1))
            self.model.params['adap_init'] =  self.model.params.a_adap * fp.S_A(f_pts[0][0]) * np.random.uniform(1, 1, (wc.params.N, 1))
        else:
            self.model.params['exc_init'] =  0.05 * np.random.uniform(0, 1, (self.model.params.N, 1))
            self.model.params['inh_init'] =  0.05 * np.random.uniform(0, 1, (self.model.params.N, 1))
            
        
        self.model.run()
        
        #for the bifurcations in the one node-model: maximal and minimal activity-value in last second of the first node
        max_exc = np.max(self.model.exc[0, -int(5000/self.model.params['dt']):])
        max_inh = np.max(self.model.inh[0, -int(5000/self.model.params['dt']):])
        
        exc_act = self.model.exc
        inh_act = self.model.inh
        adap_act = self.model.adap
        
        duration = self.model.params.duration/wc.params.dt
        cut_off = int(duration - 60*10000)
        
        x = exc_act[0, -cut_off:]
        x_adap = adap_act[0, -cut_off:]
        
        oss = self.checkOneOsc(x)
        
        LCEI = 0
        LCaE = 0
        
        if oss:
            frequencies, power_spectral_density =  signal.welch(x, 10000, window='hanning', nperseg=int(60000), scaling='spectrum')
            idx_dominant_frequ = np.argmax(power_spectral_density)
            dominant_frequency = frequencies[idx_dominant_frequ]
            
            frequencies, power_spectral_density =  signal.welch(x_adap, 10000, window='hanning', nperseg=int(60000), scaling='spectrum')
            adap_amp = np.sqrt(max(power_spectral_density))
            if adap_amp < 0.004:
                LCEI = 1.0 #Excitation-Inhibition Limit Cycle (LCEI)
            else:
                LCaE = 1.0 #adaptation-driven Limit Cycle (LCaE)
            
        else:
            dominant_frequency = False
            adap_amp = 0
            
        #Compute state durations, but cut off the first 2 seconds: 
        
        result = {'max_exc': max_exc,
                  'min_exc': min_exc,
                  'oss': oss,
                  'dominant_frequency': dominant_frequency,
                  'adaptation_amplitude': adap_amp,
                  'fixed_points': fixed_points,
                  'stability': stabi,
                  'bistability': bistab,
                  'LCEI': LCEI,
                  'LCaE': LCaE,}
        
        #safe result dictionary to a table in a hdf-file in the directory 'traj'
        self.search.saveToPypet(result, traj)
        
        
    def runToSave(self, HDF_PATH, filestr, ParamSpace, run=False):
        
        """
        Function to run the DataFrame Computations such that they're saved and callable afterwards.
        
        :HDF_PATH: Directory path, where to store the dataframe (str)
        :filestr: Name of the file, in which this shall be saved. (str)
        :ParamSpace: Values of parameters, over which shall be iterated (dict), 
                     e.g. {'exc_ext': np.linspace(0,6,51), 'inh_ext': np.linspace(0,6,51)}
        :run: bool, if dataframe needs to be derived (True), or already exists, and only needs to be called (False)
        """
       
        
        parameters = ParameterSpace(ParamSpace)
        
        if self.OneDim:
            self.search = BoxSearch(evalFunction = self.deriveOne, model=self.model, parameterSpace=parameters, filename=HDF_PATH + filestr)
        else:
            self.search = BoxSearch(evalFunction = self.deriveMulti, model=self.model, parameterSpace=parameters, filename=HDF_PATH + filestr)
        
            
        print('File is saved in %s%s' %(HDF_PATH, filestr))
        
        if run:
            self.search.run()
        
        return self.search
   
    
