# -*- coding:utf-8 -*-
"""
Created on Jan 27, 2014

@author: alex
"""
from numpy import *
from HJBSolver import * 

class ControlGenerator():
    def __init__(self, params, 
                       alpha_bounds,
                       energy_eps,
                       description = ''):
        self._params = params;
        self._alpha_bounds = alpha_bounds;
        self._energy_eps = energy_eps;
        self._description = description;
    def __call__(self, params, Tf):
         return lambda t,x: .0;

from scipy.interpolate.fitpack2 import RectBivariateSpline
class CLControlGenerator(ControlGenerator):
    def __init__(self, params, 
                       alpha_bounds,
                       energy_eps,
                       max_Tf,
                       description = 'Closed-Loop Controller based on HJB solution'):
        ControlGenerator.__init__(self, params, 
                                  alpha_bounds, 
                                  energy_eps,
                                  description);
        
        self._max_Tf = max_Tf;
        
        xmin = HJBSolver.calculate_xmin(alpha_bounds, params, num_std = 2.0)
        dx = HJBSolver.calculate_dx(alpha_bounds, params, xmin)
        dt = HJBSolver.calculate_dt(alpha_bounds, params, dx, xmin)
        
        #Set up solver
        self._Solver = HJBSolver(dx, dt, self._max_Tf, xmin)
        
        #the v solution:
        xs, ts, vs, cs =  self._Solver.c_solve(params,
                                               alpha_bounds=alpha_bounds,
                                               energy_eps=energy_eps)
        
        #Store the control 
        self._cs = cs;
                                  
    def __call__(self, params_NOT_USED, Tf):
        #ps is not necessary, but is written here 
        #Already overrunning:
        if Tf <= .0: 
            def apply_alphamax(t,x):            
                return self._alpha_bounds[1]
            return apply_alphamax;
        #Too close to the end:
        if 3*self._Solver._dt >= Tf:
            def apply_alphamax(t,x):
                return self._alpha_bounds[1]
            return apply_alphamax; 
        
        'Tf is such that we must calculate HJB soln:'
        ts = self._Solver._ts;
        indxs = where(ts>=self._max_Tf - Tf);
        
        closedloop_xs = self._Solver._xs;
        closedloop_ts = ts[indxs];
        closedloop_ts -= self._max_Tf - Tf;
        
        closedloop_cs =  squeeze(self._cs[:, indxs]); 
        stochasticInterpolator = RectBivariateSpline(closedloop_xs[:-1],
                                                      closedloop_ts,
                                                       closedloop_cs);
        def feedback_alpha(t,x):
            if t > Tf:
                return self._alpha_bounds[1]
            if x < closedloop_xs[0]:
                return .0
            if x > closedloop_xs[-1]:
                raise RuntimeError('Received x = %f > v_thresh = %f - ERROR: Controller should not be asked for what to do for values higher than the threshold!!!'%(x, closedloop_xs[-1]))
            else:
                return stochasticInterpolator(x,t)[0][0]
        
        return feedback_alpha;

from AdjointSolver import FBKSolution, deterministicControlHarness,\
    calculateOptimalControl, FPAdjointSolver
class OLControlGenerator(ControlGenerator):
    def __init__(self, params, 
                       alpha_bounds,
                       energy_eps,
                       description = 'Open-Loop Controller based on Maximum Principle solution'):
        ControlGenerator.__init__(self, params, 
                                  alpha_bounds, 
                                  energy_eps,
                                  description);
    def __call__(self, params_NOT_USED, Tf):
        if Tf <= .0: 
        #Already overrunning:
            def openloop_alpha(t,x):            
                return self._alpha_bounds[1]
            return openloop_alpha;
        else:
            params = self._params;
            alpha_bounds = self._alpha_bounds
            energy_eps = self._energy_eps 
        #Must calculate FBK soln:
            xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, params, num_std = 1.0)
            dx = FPAdjointSolver.calculate_dx(alpha_bounds, params, xmin)
            dt = FPAdjointSolver.calculate_dt(alpha_bounds, params, dx, xmin, factor = 4.)
        
            if 3*dt >= Tf:
                def openloop_alpha(t,x):
                    return self._alpha_bounds[1]
                return openloop_alpha; 
            
            #Set up solver
            xs, openloop_ts, fs, ps,\
             cs_iterates, J_iterates = calculateOptimalControl(params, Tf,
                                                    energy_eps,
                                                     alpha_bounds,
                                                     grad_norm_tol = 5e-4,
                                                     obj_diff_tol = 1e-2)
            openloop_cs = cs_iterates[-1];
            def openloop_alpha(t,x):
                if t>=Tf:
                    return self._alpha_bounds[1]
                else:
                    return interp(t, openloop_ts, openloop_cs)
     
            print Tf, dt
            return openloop_alpha;
                                      
        
        
if __name__ == '__main__':
    tau_char = .5;
    beta_high = 1.5
    beta_low = .3;
    mu_high = 1.5
    mu_low = .1
    params =  [mu_low/tau_char, tau_char, beta_high]
    alpha_bounds = array([-2, 2]);
    energy_eps = .1;
    Tf = 1.;
    
########################CLOSED LOOP:$$$$$$$$$$####
    clCG = CLControlGenerator(params,
                              alpha_bounds, energy_eps, max_Tf=Tf*3)
    fb_new = clCG(None, Tf);
    
    from TrainController import getFeedbackControl
    fb_old = getFeedbackControl(params,
                                 Tf, alpha_bounds, energy_eps)
    
    
    print 'Closed Loop Validate:'
    for t,x in zip([Tf/8., Tf/2.,Tf/2.,Tf/2., Tf+.1],
                   [.8,    -.1,   .4,   .9, .4]):
        print fb_new(t,x), fb_old(t,x)
    
########################OPEN LOOP:$$$$$$$$$$####
    olCG = OLControlGenerator(params, alpha_bounds, energy_eps)
    ol_new = olCG(None, Tf) 
    from TrainController import getOpenloopControl
    ol_old = getOpenloopControl(params, Tf, alpha_bounds, energy_eps)
    print 'Open-Loop Validate:'
    for t,x in zip([Tf/8., Tf/2.,Tf/2.,Tf/2., Tf+.1],
                   [.8,    -.1,   .4,   .9, .4]):
        print ol_old(t,x), ol_new(t,x);
        
    
    
            
        