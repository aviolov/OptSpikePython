# -*- coding:utf-8 -*-
"""
Created on Oct 22, 2012

@author: alex
"""
from __future__ import division

from SpikeTrainSimulator import SpikeTrain, OUParams, RESULTS_DIR as SPIKE_TRAINS_DIR,\
    simulate_spike_train

from numpy import linspace, float, arange, sum
from numpy import sin, sqrt, ones_like, zeros_like, where, diff, pi, log, max , sign, amin, exp
from numpy import zeros, ones, array, c_, r_, float64, matrix, bmat, Inf, ceil, arange, empty, interp, dot, sqrt
from scipy.sparse import spdiags, lil_matrix
from scipy.sparse.linalg.dsolve.linsolve import spsolve
from copy import deepcopy
from scipy.stats import norm

import os;
FIGS_DIR = '/home/alex/Workspaces/Latex/LIFEstimation/Figs/Fortet_Estimates'
import os
for D in [FIGS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)



def FortetSanityCheck(alpha, tauchar, beta, N_spikes, save_fig_name= None):
    from scipy.stats import norm 
    xth = 1.0;
    
    path_tag = '1'; 
    path_tag = OUParams.generatePathTag(alpha, tauchar, beta) + '_' + path_tag;
    filename = 'spike_train_N=%d%s'%(N_spikes, path_tag)
    T = None
    try:
        T = SpikeTrain.load(filename)  
    except IOError:
        print 'Simulating '
        simulate_spike_train(N_spikes, save_path=True, path_tag = '1',                                        
                                    params = [alpha, tauchar, beta])
        T= SpikeTrain.load(filename)
        
    T0 = .1; Tf = T.getMaxInterval(); #amin([20.0, T.getMaxInterval()]);
    
    Is = T._intervals; N = T.getNumSamples();
    
    ts = linspace(T0, Tf, 64);
    
    LHS_numerator = xth - alpha*tauchar *(1. - exp( - ts /tauchar))
    LHS_denominator = beta *  sqrt( tauchar / 2.0 * (1. - exp(- 2. *ts / tauchar ) ) ) 
    LHS = 1. - norm.cdf( LHS_numerator / LHS_denominator );
            
    RHS = empty_like(LHS);
    for t, idx in zip(ts, xrange(len(ts))):
        factor = (xth - alpha * tauchar)  / (beta * sqrt(tauchar / 2.0));
        xis = -(t - Is [ Is < t] ) / tauchar;
        exponents_term = sqrt( (1 - exp(xis))/ (1. + exp(xis)) )
        RHS[idx] = sum(1. - norm.cdf(factor * exponents_term)) / N
    
    figure(); hold (True)
    plot(ts, LHS, 'b', linewidth=3, label = 'Analytical (LHS)'); 
    plot(ts, RHS, 'r+', markersize=12, label = 'Data Convolved (RHS)');
    xlim((.0, max(ts)))
    ylim((.0, max([max(LHS), max(RHS)])))
    title(r"$\alpha, \tau, \beta = (%.2g,%.2g,%.2g)$" %(alpha, tauchar, beta), fontsize = 42)
    legend(loc = 'lower right');
    xlabel('$t$', fontsize = 18)
    
    get_current_fig_manager().window.showMaximized()
    
    if save_fig_name != None:
        file_name = os.path.join(FIGS_DIR, save_fig_name+'_N=%d'%N_spikes + '.png')
        print 'saving to ', file_name
        savefig(file_name)
    
 
def JointEstimator(T, atb_init, grad_norm_tol= 1e-2, step_tol = 1e-2, Max_Iters=100, verbose = False):
    from FPSolver import FPSolver
    from numpy.linalg.linalg import norm as vector_norm
    
    uIs, SDF = T.getSDF()
    Is = T._intervals; N = T.getNumSamples();
    T0 = .1; Tf = T.getMaxInterval();
    Tf = T.getMaxInterval()
    xth = 1.0;
    ts = linspace(T0, Tf, 64);
    
    from scipy.stats import norm
    def FortetLineEstimator(atb_origin, atb_direction):
        max_step = 1.0            
        s_tau_crit = -atb_origin[1] / atb_direction[1];
        if s_tau_crit <= 1.0 and s_tau_crit >= .0:
            max_step = min([max_step, 
                            (step_tol - atb_origin[1]) / atb_direction[1]])
        s_beta_crit = -atb_origin[2] / atb_direction[2];
        if s_beta_crit <= 1.0 and s_beta_crit >= .0:
            max_step = min([max_step, 
                            (step_tol - atb_origin[2]) / atb_direction[2]])    
              
        def loss_function(s):
            latb = atb_origin + s*atb_direction;
            
            alpha = latb[0]; tauchar= latb[1] ; beta = latb[2];
            
            LHS_numerator = xth - alpha*tauchar *(1. - exp( - ts /tauchar))
            LHS_denominator = beta *  sqrt( tauchar / 2.0 * (1. - exp(- 2. *ts / tauchar ) ) ) 
            LHS = 1. - norm.cdf( LHS_numerator / LHS_denominator );
                    
            RHS = empty_like(LHS);
            for t, idx in zip(ts, xrange(len(ts))):
                factor = (xth - alpha * tauchar)  / (beta * sqrt(tauchar / 2.0));
                xis = -(t - Is [ Is < t] ) / tauchar;
                exponents_term = sqrt( (1 - exp(xis))/ (1. + exp(xis)) )
                RHS[idx] = sum(1. - norm.cdf(factor * exponents_term)) / N
            
#            loss = max( abs(LHS-RHS) ) 
#            figure()
#            plot(ts, LHS, 'b')
#            plot(ts, RHS, 'rx'); title('a=%.2f,t=%.2f,b=%.2f; s = %.2f, loss = %.3f'%(alpha,tauchar,beta, s, loss))
                
            loss = vector_norm(LHS - RHS)

            return loss;
    
        from scipy.optimize import fminbound
        step_len = fminbound(loss_function, .0, max_step, xtol=step_tol/2.0, maxfun=64,  disp=1);
        return step_len   
    
    atb = atb_init;
    for iteration_idx in xrange(Max_Iters):
        xmin = FPSolver.calculate_xmin(Tf, atb)
        dx = FPSolver.calculate_dx(atb, xmin)
        dt = FPSolver.calculate_dt(dx, atb, xmin, factor = 4.)

        S = FPSolver(dx, dt, Tf, xmin)

        #the F solution:
        fts, F =  S.solve(atb)
        Fth = F[:,-1];
        #The s solutions:
        ss = S.solve_forward_sensitivities(atb, F)
        sth = ss[:,:,-1];
        
        grad_L = FPSolver.calcGradL(S._ts, S._dt, Fth, sth, uIs, SDF)
        grad_norm = norm(grad_L) 
        
        if (grad_norm <= grad_norm_tol):
            if verbose:
                print 'Converged after %d iters: grad norm is too small'%iteration_idx
            break

        grad_L_normalized = grad_L/ vector_norm(grad_L) 
        if verbose:
            print 'grad L_normalized = ', grad_L_normalized
        
        step_length = FortetLineEstimator(atb, -grad_L_normalized)
        
        if (step_length <= step_tol):
            if verbose:
                print 'Converged after %d iters: step length too small'%iteration_idx
            break
        
        atb = atb - step_length*grad_L_normalized
        if verbose:
            print '%d: atb = (%.2g, %.2g, %.2g)'%(iteration_idx, atb[0],atb[1],atb[2])

    if iteration_idx ==100 and verbose:
        print 'Did not converge after %d iterations' %iteration_idx
        
    return atb
    
def FPEstimator(T, atb_init, verbose = False):
    from scipy.optimize import fmin_bfgs, fmin_l_bfgs_b
    from FPSolver import FPSolver
    Tf = T.getMaxInterval()
    uIs, SDF = T.getSDF();
    
    def loss_function(atb):
        xmin = FPSolver.calculate_xmin(Tf, atb)
        dx = FPSolver.calculate_dx(atb, xmin)
        dt = FPSolver.calculate_dt(dx, atb, xmin, factor = 4.)
    
        S = FPSolver(dx, dt, Tf, xmin)
    
        #the F solution:
        fts, F =  S.solve(atb)
        Fth = F[:,-1];
        
        return FPSolver.calcLossFunction(fts, S._dt, Fth, uIs, SDF)
    def grad_loss_function(atb):
        xmin = FPSolver.calculate_xmin(Tf, atb)
        dx = FPSolver.calculate_dx(atb, xmin)
        dt = FPSolver.calculate_dt(dx, atb, xmin, factor = 4.)
    
        S = FPSolver(dx, dt, Tf, xmin)
    
        #the F solution:
        fts, F =  S.solve(atb)
        Fth = F[:,-1];
        #The s solutions:
        ss = S.solve_forward_sensitivities(atb, F)
        sth = ss[:,:,-1];
        
        return FPSolver.calcGradL(fts, S._dt, Fth, sth, uIs, SDF)
        
#    abg_est, fopt, gopt, Bopt, func_calls, grad_calls, warnflag  = fmin_bfgs(loss_function, 
#                                                                             atb_init,
#                                                                             grad_loss_function,  
#                                                                             gtol = 1e-2*Tf, full_output = 1)
#    if verbose:
#        print 'gopt = ', gopt, ': fopt = ', fopt, ': func_calls = ', func_calls, ': grad_calls = ', grad_calls 

    abg_est, f_opt, info_dict  = fmin_l_bfgs_b(loss_function,  atb_init, grad_loss_function,
                                             bounds = [(None,None), (.0, None), (.0, None)],  
                                             pgtol = 1e-1*Tf, factr = 1e12, m = 20)

    return abg_est

    
def FortetEstimator(T, atb_init):
    from scipy.stats import norm
    xth = 1.0;
    T0 = .1; Tf = T.getMaxInterval(); #amin([20.0, T.getMaxInterval()]);
    
    Is = T._intervals; N = T.getNumSamples();
    
    ts = linspace(T0, Tf, 64);
    
    def loss_function(atb):
        alpha = atb[0]; tauchar= atb[1] ; beta = atb[2];
        
        LHS_numerator = xth - alpha*tauchar *(1. - exp( - ts /tauchar))
        LHS_denominator = beta *  sqrt( tauchar / 2.0 * (1. - exp(- 2. *ts / tauchar ) ) ) 
        LHS = 1. - norm.cdf( LHS_numerator / LHS_denominator );
                
        RHS = empty_like(LHS);
        for t, idx in zip(ts, xrange(len(ts))):
            factor = (xth - alpha * tauchar)  / (beta * sqrt(tauchar / 2.0));
            xis = -(t - Is [ Is < t] ) / tauchar;
            exponents_term = sqrt( (1 - exp(xis))/ (1. + exp(xis)) )
            RHS[idx] = sum(1. - norm.cdf(factor * exponents_term)) / N
            
        return max(abs(LHS-RHS));

    
    from scipy.optimize import fmin
    atb = fmin(loss_function, atb_init, maxiter = 100, xtol=1e-2, ftol = 1e-2,  disp=0);
    
    return atb
   

def FortetVsJointComparison(atb):
    #Load Train:
    path_tag = '1'; 
    path_tag = OUParams.generatePathTag(atb[0],atb[1],atb[2]) + '_' + path_tag;
    filename = 'spike_train_N=%d%s'%(N_spikes, path_tag)

    T = None
    try:
        T = SpikeTrain.load(filename)  
    except IOError:
        print 'Simulating '
        simulate_spike_train(N_spikes, save_path=True, path_tag = '1',                                        
                                    params = [atb[0],atb[1],atb[2]])
        T= SpikeTrain.load(filename)
        
    #Estimate:
    atb_init = [.5, .5, .5] 
#    atb_fortet = FortetEstimator(T, atb_init)
#    atb_joint = JointEstimator(T, atb_init, step_tol = 1e-2, Max_Iters = 100, verbose =False)
    atb_fp = FPEstimator(T, atb_init, verbose = True)
#    
#    #Display results:
    print '%.2f,%.2f,%.2f & '%(atb[0],atb[1],atb[2]) + \
            r'%.2f,%.2f,%.2f \\'%(atb_fp[0],atb_fp[1],atb_fp[2])
#            '%.2f,%.2f,%.2f & '%(atb_fortet[0],atb_fortet[1],atb_fortet[2]) + \
#            r'%.2f,%.2f,%.2f \\'%(atb_joint[0],atb_joint[1],atb_joint[2])
    
if __name__ == '__main__':
    from pylab import *
    
    for N_spikes in [2000]:
        for atb,fig_name  in zip( [[2.0, .5, .75],
                                    [1., .5, 1.0],
                                    [.5, 1.0, 1.0]], 
                                ['high_alpha', 'mid_alpha', 'low_alpha']):
            FortetVsJointComparison(atb)
#            FortetSanityCheck(atb[0],atb[1], atb[2], N_spikes, save_fig_name = fig_name)
        
#        FortetVsJointComparison([.5, 1.0, 1.0])
        
    
        

    show()
    