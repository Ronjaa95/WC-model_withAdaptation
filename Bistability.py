#import necessary packages
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
import math

import pandas as pd

from neurolib.models.wc-adap import WCModel
from neurolib.models.wc_input import WCModel_input

from scipy.optimize import root
from scipy.optimize import fsolve
from scipy.linalg import eigvals

import logging, sys
logging.disable(sys.maxsize)


class Bistability:
    """ In the FixedPoints_Adaptation class, the only addition is the adaptation current. Since we're looking at the nullclines, we have 0=-a(t)+b*E(t), which can be reformulated into a(t)=b*E(t) and implemented like that. Therefore, we only extend the exc_rhs in the activity function by -self.model.params['a_adap']*excs and consider the derivative of [b*E(t)]'=b in df_dE, since that's the only derivative of the linearization matrix that is changed by the addition of the adaptation current. The rest stays the same."""
    
    def __init__(self, model, ext_input=[1.0, 0.5], fix_params=None):
       
        self.model = model
        
        
        if fix_params is not None:
            for k, v in zip(fix_params.keys(), fix_params.values()):
                assert k in self.model.params, f"Parameter {k} not found in outputs."
                self.model.params[k] = v
                #print('%s=%.2f' %(k,v))
        
        self.model.params['exc_ext'] = ext_input[0]
        self.model.params['inh_ext'] = ext_input[1]
        
                
        #self.stability = []
        self.bistable = []
        self.oss=False
        self.oss_list = []
        
    def testBistability(self):
        
        
        self.model.params['step_current'] = True
        
        
        self.model.params['neg_from_second'] = 58
        self.model.params['neg_to_second'] = 55
        self.model.params['neg_ext_val'] = -10
        
        self.model.params['from_second'] = 29
        self.model.params['to_second'] = 26
        self.model.params['ext_val'] = 10
        
        self.model.params['duration'] = 60*1000
        
        self.model.run()
        
        exc = self.model.exc
        inh = self.model.inh
        
        s_c = self.model.step_current     
        
        
        down = exc[:,-int(45*10000):-int(30*10000)]
        
        up = exc[:,-int(16*10000):-int(1*10000)]
        
        
        self.tuples = []
        
        for k in range(len(down)):
            down_state = (down[k] >= 0.2).astype(int)
            up_state = (up[k] >= 0.2).astype(int)
            if max(down[k])-min(down[k])>0.2:
                down_state=0.5
                self.oss=True
            elif all(down_state):
                down_state=1
            else:
                down_state=0
                
            if max(up[k])-min(up[k])>0.1:
                up_state=0.5
                self.oss=True
            elif all(up_state):
                up_state=1
            else:
                up_state=0
                
            self.tuples.append((up_state,down_state))
            
            if down_state != up_state:
                self.bistable.append(1)
            else: 
                self.bistable.append(0)
                
        return exc, s_c
    
    def upDownBistability(self):
        
        
        self.model.params['step_current'] = True
        
        
        self.model.params['neg_from_second'] = 120
        self.model.params['neg_to_second'] = 110
        self.model.params['neg_ext_val'] = -10
        
        self.model.params['from_second'] = 60
        self.model.params['to_second'] = 50
        self.model.params['ext_val'] = 10
        
        self.model.params['duration'] = 125*1000
        
        self.model.run()
        
        exc = self.model.exc
        inh = self.model.inh
        adap=self.model.adap
        
        s_c = self.model.step_current     
        
        
        down = exc[:,-int(71*10000):-int(61*10000)]
        
        up = exc[:,-int(11*10000):-int(1*10000)]
        
        
        self.tuples = []
        
        for k in range(len(down)):
            down_state = (down[k] >= 0.2).astype(int)
            up_state = (up[k] >= 0.2).astype(int)
            if all(down_state):
                down_state=1
            else:
                down_state=0
                
            if all(up_state):
                up_state=1
            else:
                up_state=0
                
            self.tuples.append((up_state,down_state))
            
            if down_state != up_state:
                self.bistable.append(1)
            else: 
                self.bistable.append(0)
                
        return exc, adap, s_c
    
    
    def testBistability_long(self):
        
        
        self.model.params['step_current'] = True
        
        
        self.model.params['neg_from_second'] = 158
        self.model.params['neg_to_second'] = 155
        self.model.params['neg_ext_val'] = -10
        
        self.model.params['from_second'] = 79
        self.model.params['to_second'] = 76
        self.model.params['ext_val'] = 10
        
        self.model.params['duration'] = 160*1000
        
        self.model.run()
        
        exc = self.model.exc
        inh = self.model.inh
        
        s_c = self.model.step_current     
        
        
        down = exc[:,-int(95*10000):-int(80*10000)]
        
        up = exc[:,-int(16*10000):-int(1*10000)]
        
        
        self.tuples = []
        
        for k in range(len(down)):
            down_state = (down[k] >= 0.2).astype(int)
            up_state = (up[k] >= 0.2).astype(int)
            if max(down[k])-min(down[k])>0.2:
                down_state=0.5
                self.oss=True
            elif all(down_state):
                down_state=1
            else:
                down_state=0
                
            if max(up[k])-min(up[k])>0.1:
                up_state=0.5
                self.oss=True
            elif all(up_state):
                up_state=1
            else:
                up_state=0
                
            self.tuples.append((up_state,down_state))
            
            if down_state != up_state:
                self.bistable.append(1)
            else: 
                self.bistable.append(0)
                
        return exc, s_c
    
