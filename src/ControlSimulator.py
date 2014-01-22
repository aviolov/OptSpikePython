# -*- coding:utf-8 -*-
"""
@author: alex
"""
from __future__ import division

from numpy import linspace, float, arange, sum,exp
from numpy import sin, sqrt, ones_like, zeros_like, where, diff, pi, log, max , sign, amin, amax
from numpy import zeros, ones, array, c_, r_, float64, matrix, bmat, Inf, ceil, arange, empty, interp, dot, sqrt
from numpy.random import randn, seed
from scipy.sparse import spdiags, lil_matrix
from scipy.sparse.linalg.dsolve.linsolve import spsolve
from copy import deepcopy
from scipy.optimize.zeros import brentq
from matplotlib.pyplot import xlim
from scipy.interpolate.interpolate import interp2d
from scipy.interpolate.fitpack2 import RectBivariateSpline
from HJB_TerminalConditioner import calculateTs_Kolmogorov_BVP
from matplotlib.patches import FancyArrowPatch, FancyArrow, ArrowStyle
from AdjointSolver import FBKSolution, deterministicControlHarness
from matplotlib.font_manager import FontProperties
from PathSimulator import ABCD_LABEL_SIZE

RESULTS_DIR = '/home/alex/Workspaces/Python/OptSpike/Results/ControlSimulator/'
FIGS_DIR = '/home/alex/Workspaces/Latex/OptSpike/Figs/ControlSimulator'

import os
for D in [FIGS_DIR, RESULTS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)
import time

label_font_size = 32
xlabel_font_size = 40

from HJBSolver import HJBSolver, HJBSolution
from collections import deque

class SpikedPath():
    def __init__(self, ts, xs, cs, t_spike):
        self.ts = ts;
        self.xs = xs;
        self.cs = cs;
        self.t_spike = t_spike

class ControlledPath():
    def __init__(self, control_f):
        self.control_function = control_f;
        
        self.ts = deque([.0])
        self.xs = deque([.0])
        self._x_prev = .0;

        self.cs = deque([])
        
        self._t_spike = None
    
    def generateStaticPath(self, skip=5):
        if self._t_spike == None:
            raise RuntimeError('Asking for a non-spiked Generated Path')
        return SpikedPath(array(self.ts)[::skip],
                          array(self.xs)[::skip], 
                          array(self.cs)[::skip], 
                          self._t_spike)
        
    def increment(self, t, x, c):
        for colection, value in zip([self.ts, self.xs, self.cs],
                                    [t,x,c]):
            colection.append(value)
        self._x_prev = x
        
    def set_spike_time(self,t, last_control):
        self._t_spike = t
        self.cs.append(last_control)
        
    def isAlive(self):
        return None == self._t_spike
        
        
def deterministicVsStochasticHarness(tb,
                                      deterministic_alpha,
                                       stochastic_alpha,
                                        dt = 1e-3, x_thresh=1.0):
    tauchar, beta = tb[0], tb[1];
    sqrt_dt = sqrt(dt);
    
    #deterministic time, path
    detPath = ControlledPath(deterministic_alpha) 
    #stochastic control path:
    stochPath = ControlledPath(stochastic_alpha)

    #the dynamics RHS:    
    def compute_dX(alpha, x, xi):
        return (alpha - x/tauchar)*dt + beta * xi *sqrt_dt
    
    
    #THE MAIN INTEGRATION LOOP:
    t = .0;
    
    while detPath.isAlive() or stochPath.isAlive():
        xi = randn()
        
        for controledPath, control_function in zip([detPath, stochPath],
                                                   [deterministic_alpha, stochastic_alpha]):        
            if controledPath.isAlive():
                x_prev = controledPath._x_prev;
                alpha = control_function(t, x_prev)
                dX = (alpha - x_prev / tauchar)*dt + beta * xi *sqrt_dt #compute_dX(alpha, x_prev, xi)
            
                x_next = x_prev + dX
                
                if x_next >= x_thresh:
                    controledPath.set_spike_time(t+dt, alpha)
                else:
                    controledPath.increment(t+dt, x_next, alpha)
        
        t += dt#Not the most elegant way to do things!!!
                    
    
    return detPath, stochPath
######################################
     
def controlHarness(params, controls, dt = 1e-3, x_thresh=1.0,
                   rand_seed = None):
    #returns 'paths' the controlled paths corresponding to the controls
    mu, tauchar, beta = params[0], params[1], params[2];
    sqrt_dt = sqrt(dt);
    
    paths = [];
    for alpha in controls:
        path = ControlledPath(alpha)
        paths.append(path)
    
    #the dynamics RHS:    
#    def compute_dX(alpha, x, xi):
#        return (mu + alpha - x/tauchar)*dt + beta * xi *sqrt_dt
    
    def anyAreAlive():
        for path in paths:
            if path.isAlive():
                return True
        return False
    
    #THE MAIN INTEGRATION LOOP:
    t = .0;
    if None != rand_seed:
        seed(rand_seed)
    while anyAreAlive():
        xi = randn()
        #Increment each path using the SAME RANDN:
        for controledPath in paths:                    
            if controledPath.isAlive():
                x_prev = controledPath._x_prev;
                alpha = controledPath.control_function(t, x_prev)
                dX = (mu + alpha - x_prev / tauchar)*dt + beta * xi *sqrt_dt #compute_dX(alpha, x_prev, xi)
            
                x_next = x_prev + dX
                
                if x_next >= x_thresh:
                    controledPath.set_spike_time(t+dt, alpha)
                else:
                    controledPath.increment(t+dt, x_next, alpha)
        
        t += dt#Not the most elegant way to do things!!!
                    
    
    return paths

def deterministicVsStochasticComparison(tb = [.6, 1.25], Tf = 1.5, energy_eps = .001, alpha_bounds = (-2., 2.),
                                        fig_name = None, visualize=False, N_samples=2):
    mpl.rcParams['figure.subplot.left'] = .05
    mpl.rcParams['figure.subplot.right'] = .975
    mpl.rcParams['figure.subplot.bottom'] = .125
    mpl.rcParams['figure.subplot.top'] = .9
    mpl.rcParams['figure.subplot.hspace'] = .33
    
    #Get the deterministic control:
    deterministic_ts, deterministic_control = deterministicControlHarness(tb, Tf, energy_eps, alpha_bounds) 
        
    #Set up the HJB solver:
    xmin = HJBSolver.calculate_xmin(alpha_bounds, tb, num_std = 1.0)
    dx = HJBSolver.calculate_dx(alpha_bounds, tb, xmin)
    dt = HJBSolver.calculate_dt(alpha_bounds, tb, dx, xmin, factor = 4.)
    
    #TODO: The way you pass params and the whole object-oriented approach is silly. Tf changes for each solve and atb don't, so maybe rething the architecture!!!
    S = HJBSolver(dx, dt, Tf, xmin)
    #the v solution:
    xs, ts, vs, cs =  S.solve(tb, alpha_bounds=alpha_bounds, energy_eps=energy_eps, visualize=False)
    
#    for sample_idx in
    
    def deterministic_alpha(t,x):
        if t>Tf:
            return alpha_bounds[1]
        else:
            return interp(t, deterministic_ts, deterministic_control)
    
#    stochasticInterpolator = interp2d(xs[:-1], ts, cs.transpose(), copy=False, fill_value = .0);
#    stochasticInterpolator = interp2d(xs[:-1], ts, cs.transpose(), kind = 'linear');
    stochasticInterpolator = RectBivariateSpline(xs[:-1], ts, cs);
    
    def stochastic_alpha(t,x):
        if t>Tf:
            return alpha_bounds[1]
        if x < S.getXmin():
            return .0
        if x>S.getXthresh():
            raise RuntimeError('Received x = %f > v_thresh = %f - ERROR: Controller should not be asked for what to do for values higher than the threshold!!!'%(x, S.getXthresh()))
        else:
            return stochasticInterpolator(x,t)[0][0]

#    figure()
#    subplot(211); hold(True)
#    sample_xs = linspace(xs[0], xs[-1], len(xs)*4)
#    plot(xs[:-1], cs[:,3], 'rx');
#    plot(sample_xs, [stochastic_alpha(ts[3], x)  for x in sample_xs], 'b') 
#    subplot(212); hold(True)
#    sample_xs = linspace(xs[0], xs[-1], len(xs)*4)
#    plot(xs[:-1], cs[:,-3], 'rx'); 
#    plot(sample_xs,[stochastic_alpha(ts[-3], x) for x in sample_xs], 'b') 
#    
#    figure(); hold(True)
#    plot(deterministic_ts[::4], deterministic_control[::4], 'rx')
#    sample_ts = linspace(deterministic_ts[0], deterministic_ts[-1], len(deterministic_ts)*4)
#    plot(sample_ts, [deterministic_alpha(t, None) for t in sample_ts], 'b')
    
    errors = empty((N_samples,2));
    detPath, stochPath = None, None #Create lists eventually
    
    seed(2012)
    for sim_idx in xrange(N_samples):
        detPath, stochPath = deterministicVsStochasticHarness(tb, deterministic_alpha, stochastic_alpha,dt = 1e-3)
        errors[sim_idx,:] = array([detPath._t_spike, stochPath._t_spike]) - Tf

        if visualize and sim_idx < 4: #4 being rather arbitrary here
            figure()
            tmax = max(( detPath.ts[-1],stochPath.ts[-1], Tf+.05))
            subplot(211); hold(True)
            plot(detPath.ts, detPath.xs, 'g', label='Deterministic Control'); 
            plot(stochPath.ts, stochPath.xs, label='Stochastic Control');
            xlabel(r'$x$', fontsize = 16) 
            ylim((-1.0,1.0)); xlim((.0, tmax)); vlines(Tf, -1.0, 1.0, colors='r')
            legend(loc='lower right')
            
            subplot(212)
            plot(detPath.ts, detPath.cs, 'g'); 
            plot(stochPath.ts, stochPath.cs); 
            ylim((alpha_bounds[0]-.1,alpha_bounds[1]+.1)); xlim((.0, tmax)); vlines(Tf, alpha_bounds[0]-.1,alpha_bounds[1]+.1, colors='r')
            ylabel(r'$\alpha$', fontsize = 16)
            xlabel('$t$', fontsize = 16)
            
            if fig_name != None:
                get_current_fig_manager().window.showMaximized()
                file_name = os.path.join(FIGS_DIR, fig_name+'id%d.png'%(sim_idx+1))
                print 'saving ', file_name
                savefig(file_name)
        
    squared_errors = errors**2
                
#    print 'Errors:\n', errors
#    print 'Squared Errors:\n', squared_errors

    print 'Realized Deterministic Squared Error = ', sum(squared_errors[:,0])/N_samples
    print 'Realized Stoch Squared Error = ', sum(squared_errors[:,1])/N_samples
    print 'Theoretical Stoch Squared Error (incl. energy)', vs[argmin(abs(xs)),0]
    
    if visualize and N_samples >= 32:
        figure()
        for idx, control_tag in zip(xrange(1,3),
                                    ['Deterministic', 'Stochastic']):
            subplot(2,2,idx);                   
            hist(errors[:,idx-1],20,range = (-2.0, 2.0))
            title(control_tag, fontsize = 32)
            xlabel('$t_{sp} - T^*$', fontsize = 24)
            subplot(2,2,idx+2)
            hist(squared_errors[:,idx-1],20, range = (.0, 4.0))
            xlabel('$(t_{sp} - T^*)^2$', fontsize = 24)
            
        if  fig_name != None:
            get_current_fig_manager().window.showMaximized()
            file_name = os.path.join(FIGS_DIR, fig_name+'hists.png')
            print 'saving ', file_name
            savefig(file_name)
           
           
           
def openLoopVsFeedbackComparison(tb = [.6, 1.25], Tf = 1.5, energy_eps = .001, alpha_bounds = (-2., 2.),
                                        fig_name = None, visualize=False, N_samples=2):
    #Get the deterministic control:
    deterministic_ts, deterministic_control = deterministicControlHarness(tb, Tf, energy_eps, alpha_bounds) 

    from AdjointSolver import calculateOptimalControl
    openloop_alphas, openloop_ts,J_openloop, k = calculateOptimalControl(tb, Tf, energy_eps, alpha_bounds, alpha_step = 1.)
        
    #Set up the HJB solver:
    xmin = HJBSolver.calculate_xmin(alpha_bounds, tb, num_std = 1.0)
    dx = HJBSolver.calculate_dx(alpha_bounds, tb, xmin)
    dt = HJBSolver.calculate_dt(alpha_bounds, tb, dx, xmin, factor = 4.)
    
    S = HJBSolver(dx, dt, Tf, xmin)
    #the v solution:
    xs, ts, vs, cs =  S.solve(tb, alpha_bounds=alpha_bounds, energy_eps=energy_eps, visualize=False)
    
    
    def deterministic_alpha(t,x):
        if t>=Tf:
            return alpha_bounds[1]
        else:
            return interp(t, deterministic_ts, deterministic_control)
        
    def openloop_alpha(t,x):
        if t>=Tf:
            return alpha_bounds[1]
        else:
            return interp(t, openloop_ts,openloop_alphas)
        
    
    stochasticInterpolator = RectBivariateSpline(xs[:-1], ts, cs);
    def feedback_alpha(t,x):
        if t>Tf:
            return alpha_bounds[1]
        if x < S.getXmin():
            return .0
        if x>S.getXthresh():
            raise RuntimeError('Received x = %f > v_thresh = %f - ERROR: Controller should not be asked for what to do for values higher than the threshold!!!'%(x, S.getXthresh()))
        else:
            return stochasticInterpolator(x,t)[0][0]
    
         
    #####The actual comparison:
    errors = empty((N_samples,3));
#    detPath, stochPath = None, None #Create lists eventually
    
    seed(2012)
    controls = [deterministic_alpha, openloop_alpha, feedback_alpha] 
    for sim_idx in xrange(N_samples):
        paths = controlHarness(tb, controls ,dt = 1e-3)
        
        for path_idx in xrange(3):
            errors[sim_idx,path_idx] = (paths[path_idx]._t_spike - Tf)
         
            
        detPath, openLoopPath, feedbackPath = paths[0],paths[1],paths[2]
        if visualize and sim_idx < 0: 
            figure()
            tmax = max(( detPath.ts[-1],openLoopPath.ts[-1], feedbackPath.ts[-1], Tf+.05))
            subplot(211); hold(True)
            plot(detPath.ts, detPath.xs, 'g', label='Deterministic Control'); 
            plot(openLoopPath.ts, openLoopPath.xs, 'r', label='Open-Loop Control');
            plot(feedbackPath.ts, feedbackPath.xs, label='Feedback Control');
            vlines(Tf, -1.0, 1.0, colors='k')
            xlabel(r'$x$', fontsize = 16) 
            ylim((-1.0,1.0)); xlim((.0, tmax)); 
            legend(loc='lower right')
            
            subplot(212)
            plot(detPath.ts, detPath.cs, 'g'); 
            plot(openLoopPath.ts, openLoopPath.cs, 'r'); 
            plot(feedbackPath.ts, feedbackPath.cs); 
            vlines(Tf, alpha_bounds[0]-.1,alpha_bounds[1]+.1, colors='k')
            ylim((alpha_bounds[0]-.1,alpha_bounds[1]+.1)); xlim((.0, tmax)); 
            ylabel(r'$\alpha(t)$', fontsize = 16)
            xlabel('$t$', fontsize = 16)
            
            if fig_name != None:
                get_current_fig_manager().window.showMaximized()
                file_name = os.path.join(FIGS_DIR, fig_name+'id%d.png'%(sim_idx+1))
                print 'saving ', file_name
                savefig(file_name)
        
    squared_errors = errors**2
                
#    print 'Errors:\n', errors
#    print 'Squared Errors:\n', squared_errors

    print 'Realized Deterministic Squared Error = ', sum(squared_errors[:,0])/N_samples
    print 'Realized Open-Loop Squared Error = ', sum(squared_errors[:,1])/N_samples
    print 'Realized Feedback Squared Error = ', sum(squared_errors[:,2])/N_samples
    print 'Theoretical Open-Loop Error (incl. energy)', J_openloop
    print 'Theoretical Stoch Squared Error (incl. energy)', vs[argmin(abs(xs)),0]
    
    if visualize and N_samples >= 32:
        mpl.rcParams['figure.subplot.left'] = .05
        mpl.rcParams['figure.subplot.right'] = .975
        mpl.rcParams['figure.subplot.bottom'] = .15
        mpl.rcParams['figure.subplot.top'] = .9
        mpl.rcParams['figure.subplot.hspace'] = .35
        
        figure()
        for idx, control_tag in zip(xrange(1,4),
                                    ['Deterministic','Open-Loop', 'Stochastic']):
            subplot(2,3,idx);                   
            hist(errors[:,idx-1],20,range = (-2.0, 2.0))
            title(control_tag, fontsize = 26)
            ylim((0, 35)) #a hack!
            xlabel('$t_{sp} - T^*$', fontsize = 24)
            subplot(2,3,idx+3)
            hist(squared_errors[:,idx-1],20, range = (.0, 4.0))
            ylim((0, 100)) #a hack!
            xlabel('$(t_{sp} - T^*)^2$', fontsize = 24)
            
        if  fig_name != None:
            get_current_fig_manager().window.showMaximized()
            file_name = os.path.join(FIGS_DIR, fig_name+'hists.png')
            print 'saving ', file_name
            savefig(file_name) 

           
def openLoopVsclosedLoopComparison(tb = [.6, 1.25], Tf = 1.5, energy_eps = .001, alpha_bounds = (-2., 2.),
                                        fig_name = None, visualize=False, N_samples=2):
    #Get the deterministic control:
    deterministic_ts, deterministic_control = deterministicControlHarness(tb, Tf, energy_eps, alpha_bounds) 

    from AdjointSolver import calculateOptimalControl
    openloop_alphas, openloop_ts,J_openloop, k = calculateOptimalControl(tb, Tf, energy_eps, alpha_bounds, alpha_step = 1.)
        
    #Set up the HJB solver:
    xmin = HJBSolver.calculate_xmin(alpha_bounds, tb, num_std = 1.0)
    dx = HJBSolver.calculate_dx(alpha_bounds, tb, xmin)
    dt = HJBSolver.calculate_dt(alpha_bounds, tb, dx, xmin, factor = 4.)
    
    S = HJBSolver(dx, dt, Tf, xmin)
    #the v solution:
    xs, ts, vs, cs =  S.solve(tb, alpha_bounds=alpha_bounds, energy_eps=energy_eps, visualize=False)
    
    
    def deterministic_alpha(t,x):
        if t>=Tf:
            return alpha_bounds[1]
        else:
            return interp(t, deterministic_ts, deterministic_control)
        
    def openloop_alpha(t,x):
        if t>=Tf:
            return alpha_bounds[1]
        else:
            return interp(t, openloop_ts,openloop_alphas)
        
    
    stochasticInterpolator = RectBivariateSpline(xs[:-1], ts, cs);
    def feedback_alpha(t,x):
        if t>Tf:
            return alpha_bounds[1]
        if x < S.getXmin():
            return .0
        if x>S.getXthresh():
            raise RuntimeError('Received x = %f > v_thresh = %f - ERROR: Controller should not be asked for what to do for values higher than the threshold!!!'%(x, S.getXthresh()))
        else:
            return stochasticInterpolator(x,t)[0][0]
    
         
    #####The actual comparison:
    errors = empty((N_samples,3));
#    detPath, stochPath = None, None #Create lists eventually
    
    seed(2012)
    controls = [openloop_alpha, feedback_alpha] 
    for sim_idx in xrange(N_samples):
        paths = controlHarness(tb, controls ,dt = 1e-3)
        
        for path_idx in xrange(len(controls)):
            errors[sim_idx,path_idx] = (paths[path_idx]._t_spike - Tf)
         
            
        openLoopPath, feedbackPath = paths[0],paths[1]
        if visualize and sim_idx < 8: 
            figure()
            tmax = max(( openLoopPath.ts[-1], feedbackPath.ts[-1], Tf+.05))
            subplot(211); hold(True)
            plot(openLoopPath.ts, openLoopPath.xs, 'r', label='Open-Loop Control');
            plot(feedbackPath.ts, feedbackPath.xs, label='Feedback Control');
            vlines(Tf, -1.0, 1.0, colors='k')
            xlabel(r'$x$', fontsize = 16) 
            ylim((-1.0,1.0)); xlim((.0, tmax)); 
            legend(loc='lower right')
            
            subplot(212)
            plot(openLoopPath.ts, openLoopPath.cs, 'r'); 
            plot(feedbackPath.ts, feedbackPath.cs); 
            vlines(Tf, alpha_bounds[0]-.1,alpha_bounds[1]+.1, colors='k')
            ylim((alpha_bounds[0]-.1,alpha_bounds[1]+.1)); xlim((.0, tmax)); 
            ylabel(r'$\alpha(t)$', fontsize = 16)
            xlabel('$t$', fontsize = 16)
            
            if fig_name != None:
                get_current_fig_manager().window.showMaximized()
                file_name = os.path.join(FIGS_DIR, fig_name+'id%d.png'%(sim_idx+1))
                print 'saving ', file_name
                savefig(file_name)
        
    squared_errors = errors**2
                
#    print 'Errors:\n', errors
#    print 'Squared Errors:\n', squared_errors

    print 'Realized Open-Loop Squared Error = ', sum(squared_errors[:,0])/N_samples
    print 'Realized Closed-Loop Squared Error = ', sum(squared_errors[:,1])/N_samples
    print 'Theoretical Open-Loop Error (incl. energy)', J_openloop
    print 'Theoretical Closed-Loop Squared Error (incl. energy)', vs[argmin(abs(xs)),0]
    
    if visualize and N_samples >= 32:
        mpl.rcParams['figure.subplot.left'] = .05
        mpl.rcParams['figure.subplot.right'] = .975
        mpl.rcParams['figure.subplot.bottom'] = .15
        mpl.rcParams['figure.subplot.top'] = .9
        mpl.rcParams['figure.subplot.hspace'] = .35
        
        figure()
        for idx, control_tag in zip(xrange(1,4),
                                    ['Deterministic','Open-Loop', 'Stochastic']):
            subplot(2,3,idx);                   
            hist(errors[:,idx-1],20,range = (-2.0, 2.0))
            title(control_tag, fontsize = 26)
            ylim((0, 35)) #a hack!
            xlabel('$t_{sp} - T^*$', fontsize = 24)
            subplot(2,3,idx+3)
            hist(squared_errors[:,idx-1],20, range = (.0, 4.0))
            ylim((0, 100)) #a hack!
            xlabel('$(t_{sp} - T^*)^2$', fontsize = 24)
            
        if  fig_name != None:
            get_current_fig_manager().window.showMaximized()
            file_name = os.path.join(FIGS_DIR, fig_name+'hists.png')
            print 'saving ', file_name
            savefig(file_name) 
   
 
def illustrateDesiredVsRealized(tb, Tf, energy_eps, alpha_bounds, ):
    
    def deterministic_alpha(t,x):
        return alpha_bounds[1] * (t> Tf / 4.0)
    paths = controlHarness(tb, [deterministic_alpha] ,dt = 1e-3)
    detPath = paths[0];

    mpl.rcParams['figure.subplot.left'] = .05
    mpl.rcParams['figure.subplot.right'] = .975
    mpl.rcParams['figure.subplot.bottom'] = .15
    mpl.rcParams['figure.subplot.top'] = .9
    mpl.rcParams['figure.subplot.hspace'] = .35

    #Visualize:
    figure()
    tsp = detPath.ts[-1]
    tmax = max((tsp  , Tf+.05))
    ax = subplot(211); hold(True)
    plot(detPath.ts, detPath.xs, 'g', label='$X_t$'); 
    vlines(Tf, -1.0, 1.0, colors='k')
    hlines(1.0, .0, tmax, linestyles='dashed', colors='r', linewidth = 2.0)
    xlabel(r'$x$', fontsize = 16) 
    ylim((-1.0,1.1)); xlim((.0, tmax)); 
    legend(loc='lower right')
    text(0.5 * (tsp + Tf), .5,
            r"$T^* - t_{sp}$", horizontalalignment='center',
            fontsize=20)
    ax.set_xticks((0, tsp, Tf))
    ax.set_xticklabels(('0', '$t_{sp}$','$T^*$'))
#    arrow( tsp, .8, (Tf - tsp - .1), 0, fc="k", ec="k",
#                head_width=0.1, head_length=0.1 )
#    farrow = FancyArrow(tsp, .8, (Tf - tsp - .1), 0, fc="k", ec="k",
#                        head_width=0.1, head_length=0.1 )
    farrow = FancyArrowPatch((tsp, .8), (Tf - .01,.8), 
                              arrowstyle='<->')
    ax.add_patch(farrow);

    
    subplot(212)
    plot(detPath.ts, detPath.cs, 'r', label=r'$\alpha(t)$'); 
    vlines(Tf, alpha_bounds[0]-.1, alpha_bounds[1]+.1, colors='k')
    ylim((alpha_bounds[0]-.1,alpha_bounds[1]+.1)); xlim((.0, tmax)); 
    ylabel(r'$\alpha$', fontsize = 16)
    xlabel('$t$', fontsize = 16)
    legend(loc='lower right')
    
    get_current_fig_manager().window.showMaximized()
    file_name = os.path.join(FIGS_DIR, 'DesiredVsRealized_Tsp.png')
    print 'saving ', file_name
    savefig(file_name)
    

def ControlSimulationDriver(control_params, Tf,
                             N_samples,
                              alpha_bounds = [-2,2.],
                              sim_params = None,
                              save_file_name = None,
                              energy_eps= .001,
                              rand_seeds=None):
    if None== sim_params:
        sim_params = control_params;
         
    #The DET Controller:
#    deterministic_ts, deterministic_control = deterministicControlHarness(control_params, Tf)
    det_alpha_const = deterministicControlHarness(control_params, Tf)
    def deterministic_alpha(t,x):
        if t>=Tf:
            return alpha_bounds[1]
        else:
#            return interp(t, deterministic_ts, deterministic_control)
            return det_alpha_const

    #The open-loop Controller:
    FBKSoln = None; HJBSoln = None;
#    if None != load_file_name:
#        FBKSoln = FBKSolution.load(load_file_name);
#    else:
    FBKSoln = FBKSolution.load(mu_beta_Tf=control_params[::2]+[Tf], energy_eps=energy_eps)
    openloop_ts = FBKSoln._ts;
    openloop_cs = FBKSoln.getControls()
    def openloop_alpha(t,x):
        if t>=Tf:
            return alpha_bounds[1]
        else:
            return interp(t, openloop_ts,openloop_cs)
   
    HJBSoln = HJBSolution.load(mu_beta_Tf =control_params[::2]+[Tf], energy_eps=energy_eps)
    closedloop_xs = HJBSoln._xs;
    closedloop_ts = HJBSoln._ts;
    closedloop_cs = HJBSoln._cs;
    stochasticInterpolator = RectBivariateSpline(closedloop_xs[:-1],
                                                  closedloop_ts,
                                                   closedloop_cs);
    def feedback_alpha(t,x):
        if t > Tf:
            return alpha_bounds[1]
        if x < closedloop_xs[0]:
            return .0
        if x > closedloop_xs[-1]:
            raise RuntimeError('Received x = %f > v_thresh = %f - ERROR: Controller should not be asked for what to do for values higher than the threshold!!!'%(x, closedloop_xs[-1]))
        else:
            return stochasticInterpolator(x,t)[0][0]
    
    #OBTAINED CONTROLS:
    #####The actual comparison:
    Paths = [[],[],[]]
    seed(2012)
#    detPath, stochPath = None, None #Create lists eventually
    controls = [deterministic_alpha, openloop_alpha, feedback_alpha] 
    for sim_idx in xrange(N_samples):
        ###MAIN CALL:
        print 'sim ', sim_idx
        rand_seed = None;
        if None != rand_seeds:
            rand_seed = rand_seeds[sim_idx]
        paths = controlHarness(sim_params,
                                controls,
                                dt = 1e-3,
                                rand_seed = rand_seed)
        ### NOW save it all:
        for strategy_idx in xrange(3):
            spikedPath = paths[strategy_idx].generateStaticPath()
            Paths[strategy_idx].append(spikedPath)

    Paths_det = Paths[0]
    Paths_ol  = Paths[1]
    Paths_cl  = Paths[2]
    
    (ControlSimulation(control_params, Tf, alpha_bounds,
                        Paths_cl, Paths_ol, Paths_det,
                        sim_params)).save(save_file_name)

def simulateRegimes(regimeParams, Tf, N_samples = 8):
    from multiprocessing import Process
    procs = [];
    for params in regimeParams:
        print 'm,tc,b =' , params
#        start = time.clock()
        ControlSimulationDriver(params, Tf, N_samples)
#        print 'one regime time = ', (time.clock() - start) / 60.0;
#        procs.append(Process(target=ControlSimulationDriver,
#                              args=(params, Tf, N_samples)   ))
#        procs[-1].start()
    
#    for proc in procs:
#        proc.join()
            
def simulateMisspecifiedRegimes(regimeParams, Tf,
                                simParams, N_samples = 8):
    for control_params, sim_params in zip(regimeParams,
                                          simParams):
        print 'control m,tc,b =' , control_params
        print 'sim     m,tc,b =' , sim_params
        ControlSimulationDriver(control_params, Tf, N_samples,
                                sim_params = sim_params,
                                save_file_name = 'Misspec_m=%.1f_b=%.1f_Tf=%.1f'%(control_params[0],
                                                                            control_params[2],
                                                                            Tf));
     #####################33
    
def getErrors(pathList):
        errors = empty(len(pathList))
        for idx, P in enumerate(pathList):
            errors[idx] = P.t_spike - Tf;
        return errors
    
regime_tags = {(1.5/.5,1.5):'SUPT_HN',
               (.1/.5 ,1.5):'subt_HN',
               (.1/.5 ,.3):'subt_ln',
               (1.5/.5,.3):'SUPT_ln'}

def latexifyRegimes(regimeParams, Tf, N_samples):    
    for params in regimeParams:
        print 'm,tc,b =' , params
        FBKSoln = FBKSolution.load(mu_beta_Tf=params[::2]+[Tf])
        HJBSoln = HJBSolution.load(mu_beta_Tf=params[::2]+[Tf])
        cSim = ControlSimulation.load(mu_beta_Tf=params[::2] + [Tf])
        
        cl_errors = getErrors(cSim._Paths_cl)
        ol_errors = getErrors(cSim._Paths_ol)
        det_errors = getErrors(cSim._Paths_det)
        
        latex_file_name = os.path.join(FIGS_DIR,
                                    'MeanSquaredErrors__%s.txt'%(regime_tags[(params[0],
                                                                              params[2])]))         
#        N_samples = len(cl_errors)
#        N_half_samples = N_samples / 2
        print 'latex output to ', latex_file_name
        latexFile =  open(latex_file_name, 'w')
        latexFile.write(r'\begin{tabular}{l|' +'p{4.9cm} p{4.5cm}' +'} \n');
#        latexFile.write(r'Control Law &'+ \
#                          r'Squared Error (%d)&  '%N_half_samples +\
#                          r'Squared Error (%d)&  '%N_half_samples +\
#                          r'Squared Error (%d) &'%N_samples +\
#                          r'Theoretical Error\\ \hline'); 
        latexFile.write(r'Control Law &'+ \
#                          r'Error (%d)&  '%N_half_samples +\
#                          r'Error (%d)&  '%N_half_samples +\
                          r'Average  Squared Spike-Time Deviation (N=%d) &'%N_samples +\
                          r'Expected Squared Spike-Time Deviation (theory)\\ \hline');         
        
        value_at_zero = HJBSoln._vs[abs(HJBSoln._xs).argmin(), 0]
        for errors, control_tag, theory_error in zip([det_errors,ol_errors,cl_errors],
                                       ['Deterministic', 'Open-Loop', 'Closed-Loop'],
                                       [.0, FBKSoln._J_iterates[-1], value_at_zero]):
            analytical_err_str = '%.3f '%theory_error
            if 'Deterministic' == control_tag:
                analytical_err_str = '--'

            write_str =  r' %s & %.3f & %s\\ '%(control_tag, 
#                                                        mean(errors[:N_half_samples]**2),
#                                                        mean(errors[N_half_samples:]**2),
                                                mean(errors**2),
                                                analytical_err_str)
            latexFile.write(write_str);
            print(write_str)
        
        latexFile.write(r'\hline');

        latexFile.write(r'\end{tabular}')
        latexFile.close()

def latexifyMisspecified(regimeParams,simParams, Tf):    
    for control_params, sim_params in zip(regimeParams,
                                          simParams):
        print 'm,tc,b =' , control_params
        FBKSoln = FBKSolution.load(mu_beta_Tf=control_params[::2]+[Tf])
        HJBSoln = HJBSolution.load(mu_beta_Tf=control_params[::2]+[Tf])
        cSim = ControlSimulation.load(mu_beta_Tf=control_params[::2] + [Tf])
        mispec_file_name = 'Misspec_m=%.1f_b=%.1f_Tf=%.1f'%(control_params[0],
                                                            control_params[2],
                                                            Tf)
        misSim = ControlSimulation.load(mispec_file_name)
       
        cl_errors  = [getErrors(cSim._Paths_cl),  getErrors(misSim._Paths_cl), ]
        ol_errors  = [getErrors(cSim._Paths_ol),  getErrors(misSim._Paths_ol), ]
        det_errors = [getErrors(cSim._Paths_det), getErrors(misSim._Paths_det), ]
        
        latex_file_name = os.path.join(FIGS_DIR,
                                    'Misspec_MeanSquaredErrors__%s.txt'%(regime_tags[(control_params[0],
                                                                              control_params[2])]))         
        print 'latex output to ', latex_file_name
        latexFile =  open(latex_file_name, 'w')
        latexFile.write(r'\begin{tabular}{l|' +'p{5cm} p{5cm} ' +'} \n');
        latexFile.write(r'Control Law &'+ \
                          r'Mean Squared Spike-Time Deviation (Correct)&  '  +\
                          r'Mean Squared Spike-Time Deviation (Misspecified) \\ \hline');
#                  +                          r'Error (theory)\\ \hline');         
        
        value_at_zero = HJBSoln._vs[abs(HJBSoln._xs).argmin(), 0]
        for errors, control_tag, theory_error in zip([det_errors,ol_errors,cl_errors],
                                       ['Deterministic', 'Open-Loop', 'Closed-Loop'],
                                       [.0, FBKSoln._J_iterates[-1], value_at_zero]):
            write_str =  r' %s & %.3f & %.3f \\ '%(control_tag, 
                                                        mean(errors[0]**2),
                                                        mean(errors[1]**2))
            latexFile.write(write_str);
            print(write_str)
        
        latexFile.write(r'\hline');

        latexFile.write(r'\end{tabular}')
        latexFile.close()
    

def visualizeRegimes(regimeParams, Tf, N_samples = 8,
                     Nbins = 100, fig_name=None):
#    label_font_size = 16
    xlabel_font_size = 32
    YMAX = 3.0
    for params in regimeParams:
        print 'm,tc,b =' , params
        
        cSim = ControlSimulation.load(mu_beta_Tf=params[::2] + [Tf])
        
        cl_errors = getErrors(cSim._Paths_cl)
        ol_errors = getErrors(cSim._Paths_ol)
        det_errors = getErrors(cSim._Paths_det)
        
        figure(figsize = (17, 5))
        subplots_adjust(hspace = .82, wspace = .15,
                        left=.025, right=.975,
                        top = .9, bottom = .2)   
                          
        for err_idx, (errors, control_tag) in enumerate(zip([det_errors,ol_errors,cl_errors],
                                                            ['Deterministic', 'Open-Loop', 'Closed-Loop'])):
            ax1 = subplot(2,3, 1 + err_idx)
            hist(errors,normed=True, bins = Nbins, range = [-2.0, 2.0]);
#            hist(errors[:,idx-1],20,range = (-2.0, 2.0))
            title(control_tag, fontsize = 32)
            ylim((.0, YMAX))
            ax1.vlines(.0, .0, YMAX, 'r', linestyle='dashed')
            if True: # 1 == err_idx:
                xlabel('$(T_{sp} - t^*)$', fontsize = xlabel_font_size)            

            ax2 = subplot(2,3, 1 + err_idx + 3)
            hist(errors**2,bins = Nbins,normed=True,  range = [.0, 4.0]);
            ylim((.0,YMAX*2))
            if True: #1 == err_idx:
                xlabel('$(T_{sp} - t^*)^2$', fontsize = xlabel_font_size)
            
            
#            for ax in [ax1, ax2]:
#                for ticklabels in [ax.xaxis.get_majorticklabels(),
#                                   ax.yaxis.get_majorticklabels()]:
#                    for label in ticklabels:
#                        label.set_fontsize(label_font_size)
 
            ax1.set_xlim((-1.75,1.75))
            ax2.set_xlim((.0,2.))
            for ax, x_ticks in zip([ax1, ax2],
                                    [[-1.5, .0, 1.5],  [0,1,2]]):
#                ax.set_yticks(tdiff_yticks)
#                ax.set_yticklabels(['$%.0f$'%v for v in tdiff_yticks],
#                                    fontsize = label_font_size)
                yticks = ax.get_yticklabels()
                for t in yticks:
                    t.set_visible(False)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(['$%.1f$'%v for v in x_ticks],
                                    fontsize = label_font_size)
        
    
        if None != fig_name:
            lfig_name = os.path.join(FIGS_DIR,
                                      fig_name + '_%s_errors_hist.pdf'%regime_tags[(params[0],
                                                                                    params[2])])
            print 'saving to ', lfig_name
            savefig(lfig_name)
 
 
def visualizeControls(regimeParams, Tf, N_samples = 8,
                     Nbins = 24, fig_name=None):
    label_font_size = 24
    xlabel_font_size = 32
    
    
    for params in regimeParams:
        print 'm,tc,b =' , params

        ts = arange(.0, Tf, .1);
#        xs = zeros_like(ts);

        #The DET Controller:
#        deterministic_ts, deterministic_control = deterministicControlHarness(params, Tf)
        det_c = deterministicControlHarness(params, Tf)
#        det_cs = interp(ts, deterministic_ts, deterministic_control)
        det_cs = det_c * ones_like(ts)
        
        #The open-loop Controller:
        FBKSoln = FBKSolution.load(mu_beta_Tf=params[::2]+[Tf])
        openloop_ts = FBKSoln._ts;
        openloop_cs = FBKSoln.getControls()
        ol_cs = interp(ts, openloop_ts,openloop_cs)
            
        #The closed-loop Controller:
        HJBSoln = HJBSolution.load(mu_beta=params[::2])
        closedloop_xs = HJBSoln._xs;
        closedloop_ts = HJBSoln._ts;
        closedloop_cs = HJBSoln._cs;
        stochasticInterpolator = RectBivariateSpline(closedloop_xs[:-1],
                                                      closedloop_ts,
                                                       closedloop_cs);
        cl_cs =  stochasticInterpolator(.0, ts).flatten()        
        figure();
        title('REGIME=%s'%regime_tags[(params[0],params[2])])
        plot(ts, det_cs, label='det'); 
        plot(ts, ol_cs, label='ol');
        plot(ts, cl_cs, label='cl');
        print cl_cs
        vlines(Tf, -2.0, 2.0, colors='k')
#        hlines(1.0, .0, path.ts[-1], linestyles='dashed', colors='r', linewidth = 2.0)
        ylabel(r'$\alpha_t$', fontsize = xlabel_font_size)
        xlabel(r'$t$', fontsize = xlabel_font_size) 
        ylim((-2.5,2.5));
#             xlim((.0, tmax)); 
        legend(loc='lower right')
        
    
        if None != fig_name:
            lfig_name = os.path.join(FIGS_DIR,
                                      fig_name + '_%s_.pdf'%regime_tags[(params[0],
                                                                        params[2])])
            print 'saving to ', lfig_name
            savefig(lfig_name)
 
 
def visualizeTrajectories(params, Tf, N_samples,
                          fig_name = None, 
                          N_paths_to_show = 8,
                          subsample = 2):
    YLABEL_PATH = 0
    print 'm,tc,b =' , params
    cSim = ControlSimulation.load(mu_beta_Tf=params[::2] + [Tf])
    
    xmax = 2.5
    for path_idx, (detPath, olPath, clPath) in enumerate(zip(cSim._Paths_det[:N_paths_to_show],
                                                             cSim._Paths_ol[:N_paths_to_show],
                                                             cSim._Paths_cl[:N_paths_to_show])):
        figure(figsize = (6, 8));
        subplots_adjust(hspace = .3, wspace = .05,
                     left=.2, right=.95,
                     top = .975, bottom = .1) 
        for path_label,path in zip(['det', 'OL', 'CL'],
                                   [detPath, olPath,clPath]):
            
            axX = subplot(211); hold(True)
            xplot = path.xs[::subsample]; xplot[-1] = 1.
            plot(path.ts[::subsample],xplot,
                 label=path_label,
                 linewidth = 1.5); 
            vlines(Tf, -1.0, 1.0, colors='k')
            hlines(1.0, .0, xmax, linestyles='dashed', colors='r', linewidth = 2.0)
            
            xlabel(r'$t$', fontsize = xlabel_font_size) 
            ylim((-.5,1.2));
            xlim((.0, xmax)); 
#            ax.set_xticks((0, tsp, Tf))
#            ax.set_xticklabels(('0', '$t_{sp}$','$T^*$'))
        
            axA = subplot(212); hold(True)
            plot(path.ts[::subsample], path.cs[::subsample],
                 label=path_label,
                 linewidth = 2);
            hlines(alpha_bounds, .0, xmax,
                    linestyles='dashed', colors='r', linewidth = 2.0)
             
#                vlines(Tf, alpha_bounds[0]-.1, alpha_bounds[1]+.1, colors='k')
            ylim((alpha_bounds[0]-.2,alpha_bounds[1]+.2));
            xlim((.0, xmax)); #anoather hack in the xmax 
            xlabel('$t$', fontsize = xlabel_font_size)
            
            axA.set_yticks((alpha_bounds[0], .0, alpha_bounds[1]))
            axA.set_yticklabels(('$%d$'%alpha_bounds[0],
                                '$0$',
                                '$%d$'%alpha_bounds[1] ), 
                                fontsize = label_font_size)
            axX.set_yticks((.0, 1.))
            axX.set_yticklabels(('$0$', '$1$'), 
                                fontsize = label_font_size)
            
            for ax in [axX, axA]:
                ax.set_xticks((.0, 1.5))
                ax.set_xticklabels(('$0$', '$1.5$'), 
                                fontsize = label_font_size)
#            
#            for ax in [axA, axX]:
#                for ticklabels in [ax.xaxis.get_majorticklabels(),
#                                   ax.yaxis.get_majorticklabels()]:
#                    for label in ticklabels:
#                        label.set_fontsize(label_font_size)
            
#            %massive hack b/c I happen to know that 5 is the left most
            if path_idx == YLABEL_PATH:
                fontP = FontProperties()
                fontP.set_size(20)
                axX.legend(loc='lower right',
                           prop = fontP, fancybox = True)
                axA.legend(loc='lower right',
                           prop = fontP, fancybox = True)
                axX.set_ylabel(r'$X_t$', fontsize = xlabel_font_size)
                axA.set_ylabel(r'$\alpha$', fontsize = xlabel_font_size)

        get_current_fig_manager().window.showMaximized()
        if fig_name != None:
            file_name = os.path.join(FIGS_DIR, fig_name + '_Traj%d.pdf'%path_idx)
            print 'saving ', file_name
            savefig(file_name)
        
            get_current_fig_manager().window.showMaximized()

 
def visualizeTrajectoriesPaper(params, Tf, 
                               loweps_file_name = 'LowEps',
                               higheps_file_name = 'HighEps', 
                               fig_name = None, 
                               paths_to_show = [0,1,2,3,4,7],
                               subsample = 2,
                               epsilon_value = [.001, .1]):
    YLABEL_PATH = 0
    print 'm,tc,b =' , params
    cSimLowEps = ControlSimulation.load(loweps_file_name)
    cSimHighEps = ControlSimulation.load(higheps_file_name)
    xmax = 2.5
    
    N_paths_shown = len(paths_to_show)


    figure(figsize = (17, 20));
    subplots_adjust(hspace = .25, wspace = .3,
                 left=.1, right=.975,
                 top = .95, bottom = .1) 
    for path_idx in xrange(0,N_paths_shown):
        for eps_idx, cSim in enumerate([cSimLowEps, cSimHighEps]):
            
            detPath, olPath, clPath= cSim._Paths_det[paths_to_show[path_idx]], \
                                         cSim._Paths_ol[paths_to_show[path_idx]],\
                                             cSim._Paths_cl[paths_to_show[path_idx]]
            for path_label,path in zip(['det', 'OL', 'CL'],
                                       [detPath, olPath,clPath]):
                
                
                subplot_idx = path_idx * 4 + eps_idx + 1
                axX = subplot(N_paths_shown*2, 2, subplot_idx); hold(True)
                xplot = path.xs[::subsample]; xplot[-1] = 1.
                plot(path.ts[::subsample],xplot,
                     label=path_label,
                     linewidth = 1.5); 
                vlines(Tf, -1.0, 1.0, colors='k')
                hlines(1.0, .0, xmax, linestyles='dashed', colors='r', linewidth = 2.0)
                
                
#                xlabel(r'$t$', fontsize = xlabel_font_size) 
                ylim((-.5,1.2));
                xlim((.0, xmax));
                if 0 == path_idx:
                    axX.set_title(r'$\epsilon = %.3f $'%epsilon_value[eps_idx],
                                  fontsize = xlabel_font_size + 10)
                 
        #            ax.set_xticks((0, tsp, Tf))
        #            ax.set_xticklabels(('0', '$t_{sp}$','$T^*$'))
                
                
                subplot_idx = path_idx * 4 + eps_idx + 2 +1
                axA = subplot(N_paths_shown*2, 2, subplot_idx);  hold(True)
                plot(path.ts[::subsample], path.cs[::subsample],
                     label=path_label,
                     linewidth = 2);
                hlines(alpha_bounds, .0, xmax,
                        linestyles='dashed', colors='r', linewidth = 2.0)
                 
        #                vlines(Tf, alpha_bounds[0]-.1, alpha_bounds[1]+.1, colors='k')
                ylim((alpha_bounds[0]-.2,alpha_bounds[1]+.2));
                xlim((.0, xmax)); #anoather hack in the xmax 
                
                
                axX.text(-.15, 1.0, '(%s)'%chr(65 + path_idx*2 + eps_idx),
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform=axX.transAxes,
                         fontsize = ABCD_LABEL_SIZE)
                if path_idx == N_paths_shown-1:
                    axA.set_xlabel('$t$', fontsize = xlabel_font_size)
                
                axA.set_yticks((alpha_bounds[0], .0, alpha_bounds[1]))
                axA.set_yticklabels(('$%d$'%alpha_bounds[0],
                                     '$0$',
                                     '$%d$'%alpha_bounds[1] ), 
                                    fontsize = label_font_size)
                axX.set_yticks((.0, 1.))
                axX.set_yticklabels(('$0$', '$1$'), 
                                    fontsize = label_font_size)
                
                for ax in [axX, axA]:
                    ax.set_xticks((.0, 1.5))
                    ax.set_xticklabels(('$0$', '$1.5$'), 
                                    fontsize = label_font_size)
           
            if eps_idx == 0:
                fontP = FontProperties()
                fontP.set_size(20)
                axX.legend(loc='lower right',
                           prop = fontP, fancybox = True)
                axA.legend(loc='lower right',
                           prop = fontP, fancybox = True)
                axX.set_ylabel(r'$X_t$', fontsize = xlabel_font_size)
                axA.set_ylabel(r'$\alpha$', fontsize = xlabel_font_size)

    get_current_fig_manager().window.showMaximized()
    if fig_name != None:
        file_name = os.path.join(FIGS_DIR, fig_name + '_Traj.pdf')
        print 'saving ', file_name
        savefig(file_name)
    
        get_current_fig_manager().window.showMaximized()
            

def simulateHighEpsilon(params, Tf, N_samples,
                        save_file_name='HighEps',
                        energy_eps = .1,
                        rand_seeds=None):
    ControlSimulationDriver(params, Tf,
                            N_samples,
                            save_file_name = save_file_name,
                            energy_eps = energy_eps,
                            rand_seeds = rand_seeds)

def visualizeHighEpsilonTrajectories(load_file_name = 'HighEps', 
                                     fig_name = None, 
                                     N_paths_to_show = 8,
                                     subsample = 1):
#    xlabel_font_size = 32
#    label_font_size = 20
    
    cSim = ControlSimulation.load(load_file_name)
    
    xmax = 2.5
    for path_idx, (detPath, olPath, clPath) in enumerate(zip(cSim._Paths_det[:N_paths_to_show],
                                                             cSim._Paths_ol[:N_paths_to_show],
                                                             cSim._Paths_cl[:N_paths_to_show])):
        figure(figsize = (6, 8));
        subplots_adjust(hspace = .3, wspace = .05,
                     left=.2, right=.95,
                     top = .975, bottom = .1) 
        for path_label,path in zip(['det', 'OL', 'CL'],
                                   [detPath, olPath,clPath]):
            
            axX = subplot(211); hold(True)
            plot(path.ts[::subsample], path.xs[::subsample],
                 label=path_label,
                 linewidth = 2); 
            vlines(Tf, -1.0, 1.0, colors='k')
            hlines(1.0, .0, xmax, linestyles='dashed', colors='r', linewidth = 2.0)
            
            xlabel(r'$t$', fontsize = xlabel_font_size) 
            ylim((-.5,1.2));
            xlim((.0, xmax)); 
#            ax.set_xticks((0, tsp, Tf))
#            ax.set_xticklabels(('0', '$t_{sp}$','$T^*$'))
        
            axA = subplot(212); hold(True)
            plot(path.ts[::subsample], path.cs[::subsample],
                 label=path_label,
                 linewidth = 2);
            hlines(alpha_bounds, .0, xmax,
                    linestyles='dashed', colors='r', linewidth = 2.0)
             
#                vlines(Tf, alpha_bounds[0]-.1, alpha_bounds[1]+.1, colors='k')
            ylim((alpha_bounds[0]-.2,alpha_bounds[1]+.2));
            xlim((.0, xmax)); #anoather hack in the xmax 
            xlabel('$t$', fontsize = xlabel_font_size)
            
            axA.set_yticks((alpha_bounds[0], .0, alpha_bounds[1]))
            axA.set_yticklabels(('$%d$'%alpha_bounds[0],
                                '$0$',
                                '$%d$'%alpha_bounds[1] ), 
                                fontsize = label_font_size)
            axX.set_yticks((.0, 1.))
            axX.set_yticklabels(('$0$', '$1$'), 
                                fontsize = label_font_size)
            
            for ax in [axX, axA]:
                ax.set_xticks((.0, 1.5))
                ax.set_xticklabels(('$0$', '$1.5$'), 
                                fontsize = label_font_size)            
#            %massive hack b/c I happen to know that 5 is the left most

            if path_idx == 5:
                fontP = FontProperties()
                fontP.set_size(20)
                axX.legend(loc='lower right',
                           prop = fontP, fancybox = True)
                axA.legend(loc='lower right',
                           prop = fontP, fancybox = True)
                axX.set_ylabel(r'$X_t$', fontsize = xlabel_font_size)
                axA.set_ylabel(r'$\alpha$', fontsize = xlabel_font_size)

        get_current_fig_manager().window.showMaximized()
        if fig_name != None:
            file_name = os.path.join(FIGS_DIR, fig_name + '_Traj%d.pdf'%path_idx)
            print 'saving ', file_name
            savefig(file_name)
        
            get_current_fig_manager().window.showMaximized()
    
    
def visualizeTrajectoriesPresentation(params, Tf, N_samples,
                          fig_name = None, 
                          N_paths_to_show = 8,
                          subsample = 2):
    xlabel_font_size = 32
    label_font_size = 20
    print 'm,tc,b =' , params
    cSim = ControlSimulation.load(mu_beta_Tf=params[::2] + [Tf])
    
    xmax = 2.5
    for path_idx, (detPath, olPath, clPath) in enumerate(zip(cSim._Paths_det[:N_paths_to_show],
                                                             cSim._Paths_ol[:N_paths_to_show],
                                                             cSim._Paths_cl[:N_paths_to_show])):
        figure(figsize = (8, 6));
        subplots_adjust(hspace = .25, wspace = .05,
                     left=.2, right=.95,
                     top = .975, bottom = .1) 
        for path_label,path in zip(['det', 'OL', 'CL'],
                                   [detPath, olPath,clPath]):
            
            axX = subplot(211); hold(True)
            plot(path.ts[::subsample], path.xs[::subsample],
                 label=path_label,
                 linewidth = 2); 
            vlines(Tf, -1.0, 1.0, colors='k')
            hlines(1.0, .0, xmax, linestyles='dashed', colors='r', linewidth = 2.0)
            
#            xlabel(r'$t$', fontsize = xlabel_font_size) 
            ylim((-.5,1.2));
            xlim((.0, xmax)); 
            axX.set_yticks((0., 1.0))
            axX.set_yticklabels(('$0$', '$1$'))
        
            axA = subplot(212); hold(True)
            plot(path.ts[::subsample], path.cs[::subsample],
                 label=path_label,
                 linewidth = 2);
            hlines(alpha_bounds, .0, xmax,
                    linestyles='dashed', colors='r', linewidth = 2.0)
            axA.set_yticks((alpha_bounds[0], .0, alpha_bounds[1]))
            axA.set_yticklabels(('$%d$'%alpha_bounds[0],
                                '$0$',
                                '$%d$'%alpha_bounds[1] ))
        
             
#                vlines(Tf, alpha_bounds[0]-.1, alpha_bounds[1]+.1, colors='k')
            ylim((alpha_bounds[0]-.2,alpha_bounds[1]+.2));
            xlim((.0, xmax)); #anoather hack in the xmax 
            xlabel('$t$', fontsize = xlabel_font_size)
            
            for ax in [axA, axX]:
                for ticklabels in [ax.xaxis.get_majorticklabels(),
                                   ax.yaxis.get_majorticklabels()]:
                    for label in ticklabels:
                        label.set_fontsize(label_font_size)
            
#            %massive hack b/c I happen to know that 5 is the left most
            fontP = FontProperties()
            fontP.set_size(label_font_size)
            axX.legend(loc='lower right',
                       prop = fontP, fancybox = True)
            axA.legend(loc='lower right',
                       prop = fontP, fancybox = True)
            axX.set_ylabel(r'$X_t$', fontsize = xlabel_font_size)
            axA.set_ylabel(r'$\alpha$', fontsize = xlabel_font_size)

        get_current_fig_manager().window.showMaximized()
        if fig_name != None:
            file_name = os.path.join(FIGS_DIR, fig_name + '_Traj%d.pdf'%path_idx)
            print 'saving ', file_name
            savefig(file_name)
        
            get_current_fig_manager().window.showMaximized()
########################################################################
class ControlSimulation():
    def __init__(self, params, Tf, alpha_bounds,
                  Paths_cl, Paths_ol, Paths_det,
                  sim_params):
        self._mu = params[0]
        self._tau_char = params[1]
        self._beta = params[2]
        self._Tf = Tf
        self._alpha_bounds = alpha_bounds
        self._Paths_cl = Paths_cl
        self._Paths_ol = Paths_ol
        self._Paths_det = Paths_det
                
    def save(self, file_name=None):
#        path_data = {'path' : self}
        if None == file_name:
            file_name = 'ControlSim_m=%.1f_b=%.1f_Tf=%.1f'%(self._mu,
                                                         self._beta,
                                                         self._Tf);
        print 'saving path to ', file_name
        file_name = os.path.join(RESULTS_DIR, file_name + '.cs')
        import cPickle
        dump_file = open(file_name, 'wb')
        cPickle.dump(self, dump_file, 1) # 1: bin storage
        dump_file.close()
        
    @staticmethod
    def load(file_name=None, mu_beta_Tf=None):
        '''not both args can be None!!!'''
        if None == file_name:
            mu,beta,Tf = [x for x in mu_beta_Tf]
            file_name = 'ControlSim_m=%.1f_b=%.1f_Tf=%.1f'%(mu,
                                                         beta,
                                                         Tf);

        file_name = os.path.join(RESULTS_DIR, file_name + '.cs') 
        print 'loading ', file_name
        import cPickle
        load_file = open(file_name, 'r')
        soln = cPickle.load(load_file)        
        return soln
########################

def archivemain():
    pass
#   tb = [.75, 1.25]; Tf = 1.5; energy_eps = .001; alpha_bounds = (-2., 2.);
#    deterministicControlHarness(tb, Tf, energy_eps, alpha_bounds)
#    deterministicVsStochasticComparison(tb, Tf, energy_eps, alpha_bounds, fig_name = 'example_controlled_trajectories_', visualize = True, N_samples=3)
#    deterministicVsStochasticComparison(tb, Tf, energy_eps, alpha_bounds, 
#                                        fig_name = 'example_controlled_trajectories_',
#                                        visualize = True, N_samples=64)
    
#    deterministic_ts, deterministic_control = deterministicControlHarness(tb, Tf, energy_eps, alpha_bounds)

#    openLoopVsFeedbackComparison(tb, Tf, energy_eps, alpha_bounds, 
#                                fig_name = '3controls_example_trajectories_',
#                                visualize = True, N_samples=128)

#    openLoopVsclosedLoopComparison(tb, Tf, energy_eps, alpha_bounds, 
#                                fig_name = 'open_vs_closed_example_trajectories_',
#                                visualize = True, N_samples=8)
    
#    openLoopVsclosedLoopComparison(tb, Tf, energy_eps, alpha_bounds, 
#                                visualize = False, N_samples=1024)

#    illustrateDesiredVsRealized(tb, Tf, energy_eps, alpha_bounds)

########################

def analyzeSuperTHighNoise():
            
    regimeParams = [ [mu_high/tau_char, tau_char, beta_high]]
    
    N_samples = 50
    for Tf in [.5, 1.0, 1.2, 2.0]:
        simulateRegimes(regimeParams, Tf, N_samples)
        
        visualizeRegimes(regimeParams, Tf, N_samples, fig_name = 'Regimes',
                     Nbins = 150)
def trajectoriesBox(params):
    N_samples = 12
#    randseeds = randint(1e12, size=N_samples)
#    randseeds = [50145379756,58039389690,51422188979,87827186063,320667215726,
#                 7461410797,86345881860,35554610571,48527676965,236333792982, 4731787355,701379482434]
#    assert(len(unique(randseeds) == N_samples))
#    for energy_eps, file_name in zip([.001, .1],
#                                      ['LowEps_Trajs',
#                                       'HighEps_Trajs']):
#        simulateHighEpsilon(params, Tf, N_samples, 
#                            save_file_name = file_name,
#                            energy_eps=energy_eps,
#                            rand_seeds = randseeds)
    
#    stride = 3;
#    for idx in xrange(0,N_samples,stride):
#        pts = arange(idx, idx+stride, dtype = int)
#        visualizeTrajectoriesPaper(params, Tf, 
#                               loweps_file_name='LowEps_Trajs',
#                               higheps_file_name='HighEps_Trajs',
#                               paths_to_show = pts)
    visualizeTrajectoriesPaper(params, Tf, 
                               loweps_file_name='LowEps_Trajs',
                               higheps_file_name='HighEps_Trajs',
                               paths_to_show = [6,8,11],                               
                               fig_name = 'Composite')

##################################################################
##################################################################
##################################################################

if __name__ == '__main__':
    from pylab import *
    
    Tf = 1.5; 
    energy_eps = .001; alpha_bounds = (-2., 2.);
    
    mu_high = 1.5;    mu_low = .1
    tau_char = .5;
    beta_high = 1.5;  beta_low = .3;
#    regimeParams = [ [mu_low/tau_char, tau_char, beta_low] ]
    
    regimeParams = [ [mu_high/tau_char, tau_char, beta_low],
                     [mu_high/tau_char, tau_char, beta_high],
                     [mu_low/tau_char, tau_char, beta_low], 
                     [mu_low/tau_char, tau_char, beta_high] ]
    N_samples = 10000
    
#    simulateRegimes(regimeParams, Tf, N_samples)
    visualizeRegimes(regimeParams[0:], Tf, N_samples,
                     fig_name = 'Regimes', Nbins = 150)
#    latexifyRegimes(regimeParams, Tf, N_samples)

    
#    visualizeTrajectoriesPresentation(regimeParams[0], Tf, N_samples,
#                          N_paths_to_show = 8,
#                          fig_name = 'SubTHighNoise_pres')
        
#    regimeParams = [ [mu_low/tau_char, tau_char, beta_high] ]
#    simParams    = [ [mu_low/tau_char, tau_char * 1.5, beta_high*1.5] ]
#    simulateMisspecifiedRegimes(regimeParams,
#                                 Tf,
#                                  simParams,
#                                   N_samples)   
#    latexifyMisspecified(regimeParams, simParams, Tf)

#    visualizeControls(regimeParams, Tf, N_samples, fig_name = 'Controls')

#    visualizeTrajectories(regimeParams[1], Tf, N_samples,
#                          N_paths_to_show = 8,
#                          fig_name = 'SubTHighNoise',
#                          subsample = 2)


####################################
#  The effect of Varying Epsilong (energy_eps)
####################################
#    highEpsParams =  [mu_low/tau_char, tau_char, beta_high]
#    trajectoriesBox(params = highEpsParams)
    
    
    show()
    