# -*- coding:utf-8 -*-
"""
@author: alex
"""
from __future__ import division

from numpy import linspace, float, arange, sum,exp,double
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
from AdjointSolver import FBKSolution, deterministicControlHarness,\
    calculateOptimalControl, FPAdjointSolver
from matplotlib.font_manager import FontProperties

RESULTS_DIR = '/home/alex/Workspaces/Python/OptSpike/Results/TrainController/'
FIGS_DIR = '/home/alex/Workspaces/Latex/OptSpike/Figs/TrainController'

label_font_size = 32
xlabel_font_size = 40

import os
for D in [FIGS_DIR, RESULTS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)
import time

from HJBSolver import HJBSolver, HJBSolution 
from collections import deque

label_font_size = 24
xlabel_font_size = 32
########################
class SpikeTrain():
    FILE_EXT = '.spt'
    def __init__(self, spike_times):
        self._spike_times = spike_times;
                
    def save(self, file_name):
        print 'saving spike train to ', file_name
        file_name = os.path.join(RESULTS_DIR, file_name + SpikeTrain.FILE_EXT)
        import cPickle
        dump_file = open(file_name, 'wb')
        cPickle.dump(self, dump_file, 1) # 1: bin storage
        dump_file.close()
        
    @staticmethod
    def load(file_name):
        file_name = os.path.join(RESULTS_DIR, file_name + SpikeTrain.FILE_EXT)
        import cPickle
        print 'loading ', file_name
 
        load_file = open(file_name, 'r')
        path = cPickle.load(load_file)        
        return path
    
def generateTargetTrainFromModel(params, 
                                 N_spikes = 17,
                                 path_tag = 'model',
                                 visualize = False):
    mu, tau_char, beta = params[:]

    dt = 1e-4
    sqrt_dt = sqrt(dt);

    #Preallocate space for the solution and the spike times:
    spikes = [];
    
    v_prev = .0
    t = .0;
    #THE MAIN INTEGRATION LOOP:
    while len(spikes) < N_spikes:
        dB = randn()*sqrt_dt;
        dv = (mu - v_prev / tau_char)*dt + beta*dB;
        v_prev += dv
        t += dt
        if v_prev >= 1.0:        
            spikes.append(t)
            v_prev = .0;
    
    if visualize:
        figure()
        plot(spikes, zeros_like(spikes), 'b.'); 
        ylim(-.25,.25); xlim(-.25, spikes[-1] + .25)

    SpikeTrain(spikes).save('target_%s_%d'%(path_tag,N_spikes));
       
def generateTargetTrain(N_spikes = 20,
                       mean_ISI = 1.5,
                       path_name = 'target',
                       visualize=False):
    from numpy.random import exponential
    
    ISIs = exponential(scale=mean_ISI, size=N_spikes)
    
    spikes = cumsum(ISIs)
    
    if visualize:
        figure()
        subplot(3,1,1) ; plot(spikes)
        subplot(3,1,2); stem(arange(len(ISIs)), ISIs)
        subplot(3,1,3); plot(spikes, zeros_like(spikes), 'b.'); 
        ylim(-.25,.25); xlim(-.25, spikes[-1] + .25)

    
    SpikeTrain(spikes).save('target_%.1f_%d'%(mean_ISI, N_spikes));    


def generateAhmadianTrain(file_tag):
    
    N_spikes = 17;
    T_mean = 1.5
    Ttot = 800;
    
    spikes =array([70,85,125,150,
                   300,310,385,390,
                   400,420,435,525,
                   605,620,625,690,
                   705], dtype = double)/ Ttot  ;
    
    N_spikes = len(spikes)
    spikes *= N_spikes *  T_mean
    
    print mean(spikes), spikes[-1]
    
    ISIs = diff(r_[0.,
                   spikes])
    
    figure()
    subplot(3,1,1) ; plot(spikes)
    subplot(3,1,2); stem(arange(len(ISIs)), ISIs)
    subplot(3,1,3); plot(spikes, zeros_like(spikes), 'b.'); 
    ylim(-.25,.25); xlim(-.25, T_mean* N_spikes + .25)

    
    SpikeTrain(spikes).save('target_%s'%(file_tag));    

def visualizeTargetTrain(file_tag = None, N_spikes = 20, mean_ISI = 1.5,
                         save_fig_name = None):
    label_font_size = 28
    xlabel_font_size = 36
    file_name = None;
    if file_tag == None:
        file_name = 'target_%.1f_%d'%(mean_ISI, N_spikes)
    else:
        file_name = 'target_%s'%(file_tag)
    ST = SpikeTrain.load(file_name)
    spikes = ST._spike_times;
    ISIs = diff(r_[0,spikes])
    
    if sum(ISIs) < 8:
        return
    print 'mean, min,max, total', mean(ISIs), amin(ISIs), amax(ISIs),sum(ISIs)
    print sort(ISIs)
    
    ###Triple-Fig:
#    figure()
#    subplot(3,1,1) ; plot(spikes)
#    subplot(3,1,2); stem(arange(len(ISIs)), ISIs)
#    subplot(3,1,3); plot(spikes, zeros_like(spikes), 'b.'); 
#    ylim(-.25,.25); xlim(-.25, spikes[-1] + .25)
    
    figure(figsize = (17, 5))
    subplots_adjust(left=.05, right=.975,
                    top = .95, bottom = .3)
    ax2 = subplot(1,1,1) ; 
    plot(spikes, zeros_like(spikes), 'b.', markersize = 16)
    setp(ax2.get_yticklabels(), visible=False)
    xlabel('$t$', fontsize = xlabel_font_size)
    title(file_tag)
    for label in ax2.xaxis.get_majorticklabels():
            label.set_fontsize(label_font_size)
    if None != save_fig_name:
        lfig_name = os.path.join(FIGS_DIR, 'target_train_%s.pdf'%save_fig_name)
        print 'saving to ', lfig_name
        savefig(lfig_name, dpi = 300)
        
class TargetTrainSimulator():
    def __init__(self, params, Tf, alpha_bounds,
                 target_spikes,
                 spikes_cl, spikes_ol):
        self._mu = params[0]
        self._tau_char = params[1]
        self._beta = params[2]
        self._Tf = Tf
        self._alpha_bounds = alpha_bounds
        self._target_spikes = target_spikes
        self._spikes_cl = spikes_cl
        self._spikes_ol = spikes_ol
                
    def save(self, file_name=None):
        if None == file_name:
            file_name = 'TargetTrain_m=%.1f_b=%.1f_Tf=%.1f_Ns=%d'%(self._mu,
                                                         self._beta,
                                                         self._Tf,
                                                         len(self._target_spikes));
        print 'saving path to ', file_name
        file_name = os.path.join(RESULTS_DIR, file_name + '.tts')
        import cPickle
        dump_file = open(file_name, 'wb')
        cPickle.dump(self, dump_file, 1) # 1: bin storage
        dump_file.close()
    @staticmethod
    def load(file_name=None, mu_beta_Tf_N_spikes=None):
        '''not both args can be None!!!'''
        if None == file_name:
            mu,beta,Tf, N_spikes = [x for x in mu_beta_Tf_N_spikes]
            file_name = 'TargetTrain_m=%.1f_b=%.1f_Tf=%.1f_Ns=%d'%(mu,
                                                         beta,
                                                         Tf,
                                                         N_spikes)

        file_name = os.path.join(RESULTS_DIR, file_name + '.cs') 
        print 'loading ', file_name
        import cPickle
        load_file = open(file_name, 'r')
        soln = cPickle.load(load_file)        
        return soln

def getOpenloopControl(params, Tf,
                       alpha_bounds = (-2,2),
                       energy_eps = .001):
    if Tf <= .0: 
    #Already overrunning:
        def openloop_alpha(t,x):            
            return alpha_bounds[1]
        return openloop_alpha;
    else:
    #Must calculate FBK soln:
        xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, params, num_std = 1.0)
        dx = FPAdjointSolver.calculate_dx(alpha_bounds, params, xmin)
        dt = FPAdjointSolver.calculate_dt(alpha_bounds, params, dx, xmin, factor = 4.)
    
        if 3*dt >= Tf:
            def openloop_alpha(t,x):
                return alpha_bounds[1]
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
                return alpha_bounds[1]
            else:
                return interp(t, openloop_ts,openloop_cs)
 
        print Tf, dt
        return openloop_alpha;

def getFeedbackControl(params, Tf,
                       alpha_bounds = (-2,2),
                       energy_eps = .001):
    if Tf <= .0: 
    #Already overrunning:
        def feedback_alpha(t,x):            
            return alpha_bounds[1]
        return feedback_alpha;
    else:
    #Must calculate HJB soln: 
        xmin = HJBSolver.calculate_xmin(alpha_bounds, params, num_std = 2.0)
        dx = HJBSolver.calculate_dx(alpha_bounds, params, xmin)
        dt = HJBSolver.calculate_dt(alpha_bounds, params, dx, xmin)
        
        if 3*dt >= Tf:
            def feedback_alpha(t,x):
                return alpha_bounds[1]
            return feedback_alpha; 
        
        #Set up solver
        S = HJBSolver(dx, dt, Tf, xmin)
        
        #the v solution:
        xs, ts, vs, cs =  S.solve(params,
                                   alpha_bounds=alpha_bounds,
                                    energy_eps=energy_eps,
                                     visualize=False)
        closedloop_xs = xs;
        closedloop_ts = ts;
        closedloop_cs = cs;
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
        print Tf, dt
        return feedback_alpha;


def runSingleTarget(params, Tf,
                    control_generator,
                    dt = 1e-3, x_thresh=1.0):
    #Hit a single target:  

    #get Control: 
    control_func = control_generator(params, Tf)
    mu, tauchar, beta = params[0], params[1], params[2];
    sqrt_dt = sqrt(dt);
    
    #THE MAIN INTEGRATION LOOP:    
    t = .0;
    x = .0;
    while True:
        xi = randn()
        #Increment each path using the SAME RANDN:
        alpha = control_func(t, x)
        dX = (mu + alpha - x / tauchar)*dt + beta * xi *sqrt_dt #compute_dX(alpha, x_prev, xi)

        t += dt;
        x += dX
                
        if x >= x_thresh:
            return t

def runSinlgeTargetTrain(spikes, 
                         params,
                         control_generator,
                         dt = 1e-3, x_thresh=1.0,
                         skip_missed = False):

    
    last_spike = .0
    achievedTrain = []
    
    for spike in spikes:
        #Calculate target:
        T_target = spike - last_spike
        
        #Skip missed?
        missed_at_start = (T_target <= .0);
        if skip_missed and missed_at_start:
            print 'Negative Target = %.2f'%T_target, ' skipping'
            continue
            
        #generate next ISI
        currentISI = runSingleTarget(params, 
                                     T_target,
                                     control_generator,
                                     dt = dt)
        #convert to global time:
        last_spike = last_spike + currentISI
         
        #store:
        achievedTrain.append(last_spike)
    
#    print spikes
#    print achievedTrain
    return achievedTrain

def TargetTrainDriver(params,
                      target_train_tag,
                      N_samples = 2,
                      N_spikes = 3,                      
                      dt = 1e-3, x_thresh=1.0,
                      control_generator = getFeedbackControl,
                      tag = 'cl',
                      skip_missed = False): 

    file_name = 'target_%s'%(target_train_tag)   
    ST = SpikeTrain.load(file_name)
    spikes = ST._spike_times;
    print 'Target Train Driver '
    sampleTrains = []
    seed(2012)
    for sample_idx in xrange(N_samples):
        print 'Running tracking train :: %d / %d'%(sample_idx,
                                                   N_samples)
        sampleTrain = runSinlgeTargetTrain(spikes,
                                            params,
                                           control_generator = control_generator,
                                           dt = dt,
                                           skip_missed=skip_missed)
        sampleTrains.append(sampleTrain)
    
    file_name = 'TargetTrainList_%d_%d_%s'%(N_spikes, N_samples, tag)
    print 'saving target train to ', file_name
    file_name = os.path.join(RESULTS_DIR, file_name + '.lst')
    dump_file = open(file_name, 'wb')
    import cPickle
    cPickle.dump(sampleTrains, dump_file, 1) # 1: bin storage
    dump_file.close()
    
def generateProbabilityOfFire(TargetTrainList, window_delta = .05):
    #Do it naively:
    N_trains = len(TargetTrainList)
    Tf = .0
    
    for train in  TargetTrainList:
        train = array(train)
        Tf = amax(r_[Tf,
                     amax(train)])
    Tf += window_delta;
    
    ts = arange(.0, Tf, window_delta);
    spike_prob = zeros_like(ts)
    for tk, t in enumerate(ts):
        prob_of_fire = .0
        for train in TargetTrainList:
            train = array(train)
            train1 = train [train <  t + window_delta/2.]
            train2 = train1[train1 >= t - window_delta/2.]
            prob_of_fire += len(train2)
        spike_prob[tk] = prob_of_fire / N_trains / window_delta
    
    return ts, spike_prob

def windowSmoothTargetTrain(TargetTrain,
                            sigma_window = .05):

    sigma_squared = sigma_window*sigma_window;

    ts = arange(.0, TargetTrain[-1] + 2* sigma_window, sigma_window / 4.);
    smooth_window  = zeros_like(ts)
    for t_spike in TargetTrain:
        local_ts = ts - t_spike
        smooth_window += exp(- (local_ts *local_ts) / (2.* sigma_squared)) / sqrt( 2.*pi * sigma_squared)

    return ts, smooth_window
 
def visualizeTrackingTrains(params,   
                          target_train_tag,                       
                          N_samples = 1,
                          N_plot_samples = 1,
                          N_spikes = 3,
                          tag = 'cl',
                          t_max = 7.,
                          save_fig = False):
    
    if N_plot_samples > N_samples:
        raise RuntimeError('N_plor_samples > N_samples')
    
    #the target train:
    target_file_name = 'target_%s'%(target_train_tag)   
    ST = SpikeTrain.load(target_file_name)
    TargetTrain = ST._spike_times;
    
    #the tracking trains:
    file_name = 'TargetTrainList_%d_%d_%s'%(N_spikes, N_samples, tag)
    print 'loading target train ', file_name
    file_name = os.path.join(RESULTS_DIR, file_name + '.lst')
    import cPickle
    load_file = open(file_name, 'r')
    TargetTrainsList = cPickle.load(load_file)        
    N_samples = len(TargetTrainsList)
    load_file.close()
    
    file_name = None;
    file_name = 'target_%s'%(target_train_tag) 
    ST = SpikeTrain.load(file_name)
    spikes = ST._spike_times;
    
    sigma_window=.1
    smoothed_ts, smoothedTarget = windowSmoothTargetTrain(TargetTrain,
                                                          sigma_window=sigma_window) 
    window_delta = .1
    ts, probOfFire = generateProbabilityOfFire(TargetTrainsList,
                                               window_delta = window_delta)
    
    figure(figsize = (17, 8)); hold(True)
    subplots_adjust(left=.05, right=.975,
                    top = .95, bottom = .2,
                    wspace = .02) 
    
    ax1 = subplot(211)
    plot(ts, probOfFire, 'b', label='Empirical')
    plot(smoothed_ts, smoothedTarget, 'r', label='Smoothed Target')
    y_up = 5.0
    ylim(.0, y_up)#    ylim(.0, amax(probOfFire))
    ax1.vlines(spikes, .0, y_up, colors='r', linestyles='dashed')
    
#    xlim(-.25, 30.)
    ax1.yaxis.tick_right()
    ax1.set_yticks((0., 5.0))
    ax1.set_yticklabels(('$0$', '$5$'), fontsize = label_font_size)
           
    setp(ax1.get_xticklabels(), visible=False)
    ylabel('Firing Rate', fontsize =xlabel_font_size)
    legend(loc='upper right')

    #############################################                    
    ax2 = subplot(212)
    plot(spikes, ones_like(spikes), 'r.');
    
    ax2.vlines(spikes, -N_samples-1., 2, colors='r', linestyles='dashed')
    for idx in xrange(N_plot_samples):
        generated_spikes = TargetTrainsList[idx]
        plot(generated_spikes,
              (- 1 -idx) * ones_like(generated_spikes), 'b.')
    
    ylabel('induced spikes', fontsize =xlabel_font_size)  
    xlabel('$t$', fontsize = xlabel_font_size) 
    ylim(-N_plot_samples-1., 2); 
#    xlim(-.25, max([spikes[-1],
#                    ts[-1] + window_delta]))
#    xlim(-.25, 30.)
    
    for ax in ax1, ax2:
#        ax.set_xlim(.0, max([spikes[-1],
#                             ts[-1] + window_delta]))
        ax.set_xlim(.0, t_max)
    
    hlines(0, 0, t_max, colors='k')
    
    setp(ax2.get_yticklabels(), visible=False)
    for label in ax2.xaxis.get_majorticklabels():
            label.set_fontsize(label_font_size)
    if save_fig:
        lfig_name = os.path.join(FIGS_DIR, tag + '_trains_sim_%d.pdf'%N_samples)
        print 'saving to ', lfig_name
        savefig(lfig_name, dpi = 300)
        
        
def analyzeGeneratedTrains(params):
    import cPickle

    mean_ISI = 1.5; N_spikes = 20;
    
    tag = 'SUBTHN_cl'
    file_name = 'TargetTrainList_%.1f_%d_%s'%(mean_ISI, N_spikes, tag)
    print 'loading target train ', file_name
    load_name = os.path.join(RESULTS_DIR, file_name + '.lst')
    load_file = open(load_name, 'r')
    TargetTrainsList_cl = cPickle.load(load_file)

    tag = 'SUBTHN_ol'
    file_name = 'TargetTrainList_%.1f_%d_%s'%(mean_ISI, N_spikes, tag)
    print 'loading target train ', file_name
    load_name = os.path.join(RESULTS_DIR, file_name + '.lst')
    load_file = open(load_name, 'r')
    TargetTrainsList_ol = cPickle.load(load_file)
    
    
    for ol_train, cl_train in zip(TargetTrainsList_ol,
                                  TargetTrainsList_cl):
    
        print array(ol_train) - array(cl_train)
        
def analyzeTargetTrains():
    N_spikes = 16
    for path_tag in ['crit' ,'subthn']:
        print path_tag
        file_tag= '%s_%d'%(path_tag,N_spikes)
        visualizeTargetTrain(file_tag= file_tag)
    
    print 20*'--'
    alphas = [-2,2]  
    
    mu_high = 1.5;    mu_low = .1; mu_crit = 1.;
    tau_char = .5;
    beta_high = 1.5;  beta_low = .3; 
    regimeParams = [ [mu_high/tau_char, tau_char, beta_low],
                     [mu_high/tau_char, tau_char, beta_high],
                     [mu_low/tau_char, tau_char, beta_low],  
                     [mu_low/tau_char, tau_char, beta_high]  ]

    from scipy.special import gamma, psi  
    N = 16.;
    ns = arange(1.,N+1)
    a_ks = .5*gamma(ns/2.) / gamma(ns+1)
    def phi_1(z):
        return dot( (sqrt(2)*z)**ns, a_ks)

    def getMean(mu, tau_char, beta):
#        if mu*tau_char < 1.0:
#            return Infinity
#            
        xi = -mu*tau_char*sqrt(2./ beta**2 / tau_char)
        eta= (1.-mu*tau_char)*sqrt(2./ beta**2 / tau_char)
        return tau_char * (phi_1(eta) - phi_1(xi))
        
    def getDeterministic(mu, tau_char):
        if mu <= 1.0:
                return Infinity
        return tau_char * log(mu*tau_char / (mu*tau_char - 1))
    print 'mean inhibited SuperT LN = %.2f'%getMean(mu_high*tau_char + alphas[0],
                                                    tau_char,beta_low)
    print 'mean inhibited SuperT HN = %.2f'%getMean(mu_high*tau_char + alphas[0],
                                                    tau_char, beta_high )

    print 20*'--'
    print 'SLOWEST deterministic SubT = %.2f'%getDeterministic(mu_low/tau_char + alphas[0],
                                                       tau_char)
    print 'FASTEST deterministic SubT = %.2f'%getDeterministic(mu_low/tau_char + alphas[1],
                                                       tau_char)
    print 'SLOWEST inhibited SuperT = %.2f'%getDeterministic(mu_high/tau_char + alphas[0],
                                                       tau_char)
    print 'FASTEST inhibited SuperT = %.2f'%getDeterministic(mu_high/tau_char + alphas[1],
                                                       tau_char)
    

#    for params in regimeParams:
#        mu_base, tau_char,beta= params[:];
#        time_to_reach = empty(2)
#        for idx, mu in enumerate([mu_base + alphas[0], mu_base+alphas[1]]):
#            print mu, tau_char, beta
#            
#            if mu < 1.0:
#                means[idx] = Infinity
#            
#            
#        print 'nonoise_inhibited: %.2f'%means[0],\
#              'nonoise_excitory: %.2f'%means[1]
                        
        
    
    
    
    
########################
if __name__ == '__main__':
    from pylab import *
    
    
    mu_high = 1.5;    mu_low = .1; mu_crit = 1.;
    tau_char = .5;
    beta_high = 1.5;  beta_low = .3;
#    regimeParams = [ [mu_crit/tau_char, tau_char, beta_low],
#                     [mu_crit/tau_char, tau_char, beta_high] 
#                   ]
    regimeParams = [ [mu_high/tau_char, tau_char, beta_low],
                     [mu_high/tau_char, tau_char, beta_high] 
                   ]
#    regimeParams = [ [mu_low/tau_char, tau_char, beta_low],
#                     [mu_low/tau_char, tau_char, beta_high] 
#                   ]
    regimeParams = [ [mu_high/tau_char, tau_char, beta_low],
                     [mu_high/tau_char, tau_char, beta_high],
                     [mu_low/tau_char, tau_char, beta_low],  
                     [mu_low/tau_char, tau_char, beta_high]  ]

    N_spikes  = 16; #17  #3,4,20
    N_samples = 50;

###### ### ### ### ### ##### 
### GENERATE TARGET:
###### ### ### ### ### ###### 
#    target_params = [mu_high/tau_char, tau_char, beta_high]
#    for gen_idx in arange(32):
#        target_regime = 'supthn_%d'%gen_idx
##        generateTargetTrainFromModel(target_params,
##                                  N_spikes,
##                                  path_tag=target_regime)
#        file_tag= '%s_%d'%(target_regime,N_spikes)
#        visualizeTargetTrain(file_tag= file_tag)

    
    target_regime ='supthn_8' #'crit' #'suptln' # 'supthn' #'crit' #'suptln' #'crit' #'subthn'
#    target_train_tag = '%s_%d'%(target_regime, N_spikes)
#    visualizeTargetTrain(file_tag= target_train_tag)
    
#        analyzeTargetTrains()
#
# SIMULATE:  
    from multiprocessing import Process
    procs = [];
    for params, param_tag in zip(regimeParams[:2],
                             ['SUPT_ln', 'SUPT_HN']):
#    for params, param_tag in zip(regimeParams[2:],
#                                 ['subt_ln', 'subt_HN']):
#        for control_tag, c_function in zip(['cl', 'ol'],
#                                       [getFeedbackControl,
#                                        getOpenloopControl ]):
        for control_tag, c_function in zip(['ol'],
                                       [getOpenloopControl]):

                tag = param_tag + '_' + control_tag
                file_tag = '%s_%d'%(target_regime, N_spikes)
                #Simulate:
                procs.append( Process(target=TargetTrainDriver,
                                      args=(params,),
                                      kwargs = {'target_train_tag':file_tag,
                                              'N_samples':N_samples,
                                              'N_spikes':N_spikes,
                                              'control_generator':c_function,
                                              'tag' :tag}) )
#                TargetTrainDriver(params,
#                                  target_train_tag=file_tag,
#                                  N_samples = N_samples,
#                                  N_spikes = N_spikes,
#                                  control_generator = c_function,
#                                  tag = tag,
#                                  dt = 1e-3)
                procs[-1].start()
                
    for proc in procs:
        proc.join()


#################        
##### VISUALIZE:
#################    
##    for params, param_tag in zip(regimeParams[2:],
##                                  ['subt_ln', 'subt_HN']):
    for params, param_tag in zip(regimeParams[:2],
                                 ['SUPT_ln', 'SUPT_HN']):
#        for control_tag, c_function in zip(['cl', 'ol'],
#                                       [getFeedbackControl,
#                                        getOpenloopControl ]):
                control_tag = 'ol'
                c_function = getOpenloopControl
                tag = param_tag + '_' + control_tag
                #Visualize:
                file_tag = '%s_%d'%(target_regime, N_spikes)
                visualizeTrackingTrains(params,
                                      target_train_tag=file_tag,                         
                                      N_spikes=N_spikes,
                                      N_samples = N_samples,                         
                                      N_plot_samples=amin([N_samples,10]),
                                      tag = tag,
                                      save_fig = True,
                                      t_max=11.)
                

############################
###Increased Power Bounds:
###########################
    params = regimeParams[1]
    param_tag = 'SUPT_HN'
    control_tag = 'cl'
#    
    c_function = lambda params, Tf: getFeedbackControl(params, Tf,
                                                       alpha_bounds = (-4,4))
    tag = param_tag + '_' + control_tag + '_' +'aplus'
    target_train_tag = '%s_%d'%(target_regime, N_spikes)
    #Simulate:
#    TargetTrainDriver(params,
#                      target_train_tag=target_train_tag,
#                      N_samples = N_samples,
#                      N_spikes = N_spikes,
#                      control_generator = c_function,
#                      tag = tag,                      
#                      dt = 1e-3)
#    visualizeTrackingTrains(params,
#                          target_train_tag=target_train_tag,
#                          N_spikes=N_spikes,
#                          N_samples = N_samples,                         
#                          N_plot_samples=amin([N_samples,10]),
#                          tag = tag,
#                          save_fig = True,
#                          t_max=11.)


#    skip_missed = False;
#    file_tag = 'Ahmadian'
#    generateAhmadianTrain(file_tag)
#    visualizeTargetTrain('Ahmadian',
#                         save_fig_name='Ahmadian')
##    generateTargetTrain(N_spikes)
#    visualizeTargetTrain(file_tag='model_%d'%N_spikes)
    
#    params = regimeParams[0]; tag = 'SUBTHN_cl_skip'
#    TargetTrainDriver(params,
#                              N_samples = N_samples,
#                              N_spikes = N_spikes,
#                              control_generator = getFeedbackControl,
#                              tag = tag,
#                              dt = 1e-4,
#                              skip_missed=True)
#    visualizeTrackingTrains(params,
#                          N_spikes=N_spikes,
#                           tag = tag,
#                           save_fig=True)

#    for params, param_tag in zip(regimeParams,
#                                 ['CRITLN', 'CRITHN']):
#        for control_tag, c_function in zip(['cl', 'ol'],
#                                       [getFeedbackControl,
#                                        getOpenloopControl ]):
#                tag = param_tag + '_' + control_tag
#                
#                #Simulate:
#        #        TargetTrainDriver(params,
#        #                          target_train_tag=file_tag,
#        #                          N_samples = N_samples,
#        #                          N_spikes = N_spikes,
#        #                          control_generator = c_function,
#        #                          tag = tag,
#        #                          dt = 1e-3)
#                
#                #Visualize:    
#                visualizeTrackingTrains(params,                         
#                                      N_spikes=N_spikes,
#                                      tag = tag,
#                                      save_fig = True)


#    params = regimeParams[0];
##    param_tag = 'SUPERTLN_Ahmadian' #'CRITLN_Ahmadian'    
#    for control_tag, c_function in zip(['cl', 'ol'],
#                               [getFeedbackControl,
#                                getOpenloopControl ]):
#        
#        #How to treat missed targets?
#        tag = param_tag + '_' + control_tag
#        if skip_missed:
#            tag += '_skip' 
#        
#        #Simulate:
##        TargetTrainDriver(params,
##                          target_train_tag=file_tag,
##                          N_samples = N_samples,
##                          N_spikes = N_spikes,
##                          control_generator = c_function,
##                          tag = tag,
##                          dt = 1e-3)
#        
#        #Visualize:    
#        visualizeTrackingTrains(params,    
#                              target_train_tag=file_tag,                         
#                              N_spikes=N_spikes,
#                              tag = tag,
#                              save_fig = True)
    show()
    