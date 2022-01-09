#import necessary packages
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
import math

import pandas as pd

from neurolib.models.wc-adap import WCModel

from scipy.optimize import root
from scipy.optimize import fsolve
from scipy.linalg import eigvals

import logging, sys
logging.disable(sys.maxsize)


class FixedPoints:
    """ In the FixedPoints_Adaptation class, the only addition is the adaptation current. Since we're looking at the nullclines, we have 0=-a(t)+b*E(t), which can be reformulated into a(t)=b*E(t) and implemented like that. Therefore, we only extend the exc_rhs in the activity function by -self.model.params['a_adap']*excs and consider the derivative of [b*E(t)]'=b in df_dE, since that's the only derivative of the linearization matrix that is changed by the addition of the adaptation current. The rest stays the same."""
    
    def __init__(self, model, ext_input=[1.0, 0.5], fix_params=None, vary_params=None):
        
        self.model = model
        
        if fix_params is not None:
            for k, v in zip(fix_params.keys(), fix_params.values()):
                assert k in self.model.params, "param not in model params"
                self.model.params[k] = v
                #print('%s=%.2f' %(k,v))
        
        if vary_params is not None:
            for k, v in zip(vary_params.keys(), vary_params.values()):
                assert k in self.model.params, "param not in model params"
                self.model.params[k] = v
        
        self.model.params['exc_ext'] = ext_input[0]
        self.model.params['inh_ext'] = ext_input[1]
                
        self.fixed_points = []
        self.stability = []
        self.bistable = False
        
        self.max_r_e = 1.0
        
        
        # # # - - - # # # - - - transfer functions - - - # # # - - - # # #
        
    def S_E(self, x):
        return 1.0 / (1.0 + np.exp(-self.model.params.a_exc * (x - self.model.params.mu_exc)))
    
    def S_I(self, x):
        return 1.0 / (1.0 + np.exp(-self.model.params.a_inh * (x - self.model.params.mu_inh)))
    
    def S_A(self, x):
        return 1.0 / (1.0 + np.exp(-self.model.params.a_a * (x - self.model.params.mu_a)))
    
    
    # # # - - - # # # - - - derivatives of transfer functions - - - # # # - - - # # #
    
    def derivS_E(self, x):
        return self.model.params.a_exc * self.S_E(x) * (1 - self.S_E(x))
    
    def derivS_I(self, x):
        return self.model.params.a_inh * self.S_I(x) * (1 - self.S_I(x))
    
    def derivS_A(self, x):
        return self.model.params.a_a * self.S_A(x) * (1 - self.S_A(x))
    
    
    # # # - - - # # # - - - inverses of transfer functions - - - # # # - - - # # #
    
    def inverseS_E(self, y):
        return self.model.params.mu_exc - (1/self.model.params.a_exc) * np.log((1/y)-1)
    
    def inverseS_I(self, y):
        return self.model.params.mu_inh - (1/self.model.params.a_inh) * np.log((1/y)-1)
    
    def inverseS_A(self, y):
        return self.model.params.mu_a - (1/self.model.params.a_a) * np.log((1/y)-1)
    
    
    # # # - - - # # # - - - the nullclines- - - # # # - - - # # #
    
    def I(self, E):
        """Returns the excitatory nullcline w.r.t. E"""
        inside = (self.model.params.c_excexc * E - 
                  self.inverseS_E(E) - 
                  self.model.params.a_adap * self.S_A(E) + 
                  self.model.params.exc_ext)
        return (1/self.model.params.c_inhexc) * inside
    
    def E(self, I):
        """Returns the inhibitory nullcline w.r.t. I"""
        inside = (self.model.params.c_inhinh * I + 
                  self.inverseS_I(I) - 
                  self.model.params.inh_ext)
        return (1/self.model.params.c_excinh) * inside
    
    
    # # # - - - # # # - - - linearization matrix (jacobian matrix)- - - # # # - - - # # #
    
    
    def df_dE(self, E, I):
        B_E = (self.model.params.c_excexc * E - 
               self.model.params.c_inhexc * I - 
               self.model.params.a_adap * self.S_A(E) + 
               self.model.params.exc_ext)
        outside = (-1.0 + self.derivS_E(B_E) * (self.model.params.c_excexc - self.model.params.a_adap * self.derivS_A(E)))
        return ((1/self.model.params.tau_exc) * outside)
    
    def df_dI(self, E, I):
        B_E = (self.model.params.c_excexc * E - 
               self.model.params.c_inhexc * I - 
               self.model.params.a_adap * self.S_A(E) + 
               self.model.params.exc_ext)
        return ((1/self.model.params.tau_exc) * (-self.derivS_E(B_E) * self.model.params.c_inhexc))
    
    def dg_dE(self, E, I): 
        B_I = (self.model.params.c_excinh * E - 
               self.model.params.c_inhinh * I +
               self.model.params.inh_ext)
        return ((1/self.model.params.tau_inh) * (self.derivS_I(B_I) * self.model.params.c_excinh))
    
    def dg_dI(self, E, I): 
        B_I = (self.model.params.c_excinh * E - 
               self.model.params.c_inhinh * I +
               self.model.params.inh_ext)
        return ((1/self.model.params.tau_inh) * (-1.0 - self.derivS_I(B_I) * self.model.params.c_inhinh))
    
    def jacobian(self, x):
        E = x[0]
        I = x[1]
        return [[self.df_dE(E, I), self.df_dI(E, I)], [self.dg_dE(E, I), self.dg_dI(E, I)]]
    
    
    
    # # # - - - # # # - - - the activity- - - # # # - - - # # #
    
    def activity(self, x):
        E = x[0]
        I = x[1]
        exc_rhs = (
            1
            / self.model.params.tau_exc
            * (
                - E + self.S_E( #ommiting the refractory period
                  self.model.params.c_excexc * E  # input from within the excitatory population
                - self.model.params.c_inhexc * I  # input from the inhibitory population
                - (self.model.params.a_adap * self.S_A(E))  # spike-frequency adaptation as negative feedback term
                + self.model.params.exc_ext) # external input
            )
        )
        inh_rhs = (
            1
            / self.model.params.tau_inh
            * (
                - I + self.S_I( #ommitting refractory period
                  self.model.params.c_excinh * E  # input from the excitatory population
                - self.model.params.c_inhinh * I  # input from within the inhibitory population
                + self.model.params.inh_ext )  # external input
            )
        )
        
        return [exc_rhs, inh_rhs]
    
    
    
    # # # - - - # # # - - - derive the fixed points- - - # # # - - - # # #
    
    def computeFPs(self):
        """ Derive all fixed points and collect them in the list self.fixed_points """
    
        for i in np.linspace(0.0, 1.0, 101):
            sol = root(self.activity, [i, i], jac=self.jacobian, method='hybr')
           # fix_point = [round(sol.x[0], 8), round(sol.x[1], 8)]
            if sol.success:
                if all(np.isclose(self.activity(sol.x), [0.0, 0.0])):
                    if len(self.fixed_points)==0: #always append the firstly derived fixed point
                        self.fixed_points.append(sol.x)
                    else:
                        already_derived = False
                       # self.fixed_points = self.fixed_points.sort()
                        for k in range(len(self.fixed_points)):
                            if all(np.isclose(sol.x, self.fixed_points[k], atol=1e-9)):
                                already_derived = True
                            else: 
                                pass
                        if already_derived: #all(array): #if fix_point in self.fixed_points:
                            pass #skip the already derived fixed points
                        else:
                            self.fixed_points.append(sol.x)
                        
    def checkFixPtsStability(self, fixed_points):
        self.fixed_points = np.sort(self.fixed_points, axis=0)
        fixed_points = np.sort(fixed_points, axis=0)
        for i in range(len(fixed_points)):
            E0 = fixed_points[i][0]
            I0 = fixed_points[i][1]
            y=[E0, I0]
            A = self.jacobian([E0,I0])
            w = eigvals(A)
            if all(elem.real<0 for elem in w):
               # self.stability.append(1)
                if len(self.stability)==0:
                    self.stability.append(1)
                elif len(self.stability) > 0 and self.stability[len(self.stability)-1]==0:
                    self.stability.append(1)
                else:
                    self.stability.append(0)
            else: 
                self.stability.append(0)
                
                
                
                
    def runAll(self):
        #generates the self.fixed_points-list of all fixed points
        self.computeFPs() #has no input, since all necessary parameters are set in __init__()
    
        self.checkFixPtsStability(self.fixed_points) #generates the self.stability list with 1 for stable 0 otherwise
        
        if sum(self.stability)==2:
            self.fixed_points = np.sort(self.fixed_points, axis=0)
            self.bistable = True
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        