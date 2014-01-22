# -*- coding:utf-8 -*-
"""
Created on Oct 19, 2012

@author: alex
"""

from __future__ import division
import numpy as np
from numpy import zeros, ones, array, r_, float64, matrix, bmat, Inf, ceil, arange, cumsum
#from scipy.linalg import lu_solve, inv, lu_factor
from numpy import zeros_like
from copy import deepcopy
from numpy import sin, sqrt
from numpy.random import randn, rand
from scipy.integrate.odepack import odeint
from scipy.interpolate.interpolate import interp1d

from SpikeTrainSimulator import SpikeTrain, OUParams, RESULTS_DIR as SPIKE_TRAINS_DIR, simulate_spike_train
from numpy.lib.function_base import linspace

#RESULTS_DIR = '/home/alex/Workspaces/Python/OptSpike/SpikeTrains/'
FIGS_DIR = '/home/alex/Workspaces/Latex/OptSpike/Figs/'
import os

for D in [FIGS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)

def calculateT2_vectorized(x0,alpha,beta):
    '''Does not work!!!'''
    
    D_reciprocal  = 2./ beta/beta; 
    xi  = (alpha - x0) * sqrt(D_reciprocal)
    eta = (1.0 - alpha)* sqrt(D_reciprocal)
    
    z = max(xi,eta)
    print 'z =' ,z
    
    def root_Nmax(x):
        return x*(log(2*z) - log(x)/2.) + 2.0*log(10.) 
        
    from scipy.optimize import newton, bisect
#    N_max = ceil(newton(root_Nmax, max(2.0, z*z), tol= 1.e-1))
    
    N_max = 10.0    
    if root_Nmax(N_max) > .0 and z>1.0:
        N_max = ceil(bisect(root_Nmax, 1.0, 10.0*z*z*z*z, xtol= 1.e-1))
    print 'N_max =' ,N_max
    
    ns = arange(1.,N_max+1)

    from scipy.special import psi, gamma 
    a_ks = .5*gamma(ns/2.) / gamma(ns+1)
    b_ks = (a_ks * ( psi(ns/2) - psi(1.) ) )
    
    def phi_1(z):
        return dot( (sqrt(2)*z)**ns, a_ks)
       
    def phi_2(z):
        return dot( (sqrt(2)*z)**ns, b_ks)

    T2 =  2*phi_1(eta)**2 - phi_2(eta) - 2*phi_1(eta)*phi_1(xi) + phi_2(xi);  
    
    return T2

def calculateTs_Ricciardi(x0,alpha,beta, tauchar = 1.0, xth= 1.0):
    sqrt_D_reciprocal  = sqrt(2.) / beta; 
    xi  = (x0 - alpha) * sqrt_D_reciprocal
    eta = (xth - alpha)* sqrt_D_reciprocal
    
    from scipy.special import psi, gamma 
    def phi_12_direct(z):
        n = 1; dphi1 = .0; dphi2 = .0
        phi1 = .0;
        phi2 = .0;
        n = 1
        while (dphi2 > 1e-4 or dphi1 > 1e-4) or n == 1:

            dphi1 = (z*sqrt(2))**n / gamma(n+1.0) * gamma(n/2.0)
            dphi2 = dphi1 * (psi(n/2.0) - psi(1.0)) 
            
            phi1 += dphi1
            phi2 += dphi2;
            
            n += 1;
            
        return phi1/2.0, phi2/2.0        
        
#    phi_1_eta, phi_2_eta, phi_1_xi, phi_2_xi = .0,.0,.0,.0;

    Z_MAX = 7.0; Z_MIN = -7.0;
#    print eta, xi
    def compute_direct(eta,xi):
        return ( (eta > Z_MIN) and (eta < Z_MAX) and (xi > Z_MIN) and (xi < Z_MAX) )
     
    phi_1_xi, phi_2_xi, phi_1_eta, phi_2_eta = 4*[.0]
    if compute_direct(eta,xi):
        phi_1_xi, phi_2_xi = phi_12_direct(xi)
        phi_1_eta, phi_2_eta = phi_12_direct(eta)
    else:
        return -1.0, -1.0
    
    T1 = phi_1_eta - phi_1_xi
    T2 =  2*phi_1_eta**2 - phi_2_eta - 2*phi_1_eta*phi_1_xi + phi_2_xi;  
    
    return T1, T2
     
    
def calculateTs_Kolmogorov_quad(x0s,alpha, beta, tauchar = 1.0, xth= 1.0, xmin = -1.0):
    from scipy.integrate import quad
    
    D = beta*beta /2.
    
    def S1(y):
        integral, err = quad(lambda xi: exp( -(alpha *xi - xi*xi/ 2. / tauchar) / D) / D , xmin, y)
        
        if err > 1e-2:
            raise RuntimeError('Integrate error is too high')
        
        return -exp( (alpha*y - y*y/ 2. / tauchar) / D ) * integral 
        
    def T1(x0):
        integral, err = quad(S1, x0, xth)
        
        if err > 1e-2:
            raise RuntimeError('Integrate error is too high')
        
        return -integral
        
    T1s = empty_like(x0s)
    for idx, x0 in zip(xrange(len(x0s)), x0s):
        T1s[idx] = T1(x0)
    
#    def S2_integrand(xi):
#        return T1(xi) / D
    
    T2s = empty_like(x0s);
    
#    for idx, x0 in zip(xrange(len(x0s)), x0s):
#        T2[s] = T1(x0)
    
    return T1s, T2s  

     
def calculateTs_Kolmogorov_BVP(alpha, beta, tauchar = 1.0, xth= 1.0, xmin = -1.0):
    D = beta*beta /2.
    
    def dS(S,x):
        return  -( (alpha-x/tauchar)/D ) * S -1.0/D;
    
    S_0 = .0;
    
    xs = linspace(xmin, xth, 1000); dx = (xs[1]-xs[0])
    Ss = odeint(dS, S_0, xs);
    if max(Ss) > .0:
        raise RuntimeError('Ss should be  negative')
    
    T1s = -cumsum(Ss[-1::-1]) * dx
    T1s = T1s[-1::-1]
    
    T1_interpolant = interp1d(xs, T1s, bounds_error=False, fill_value = T1s[-1]);
    def dS2(S,x):
        T1 = T1_interpolant(x)
        return  -( (alpha-x/tauchar)/D ) * S -2.0*T1/D;
    
    Ss = odeint(dS2, S_0, xs);
    if max(Ss) > .0:
        raise RuntimeError('Ss should be  negative')
    T2s = -cumsum(Ss[-1::-1]) * dx
    T2s = T2s[-1::-1]
    
    return xs, T1s, T2s  
    
    
def moments_analysis():
    N_spikes = 128
    x0s = [.0,.5,.9] #linspace(-.5, .99, 6)
    alphas = [2.]#[2.33];
    betas = [1.25]; 
    taus = [.75 ] #1.25
    
    for alpha in alphas:
        for tauchar in taus:
            for beta in betas:
                print 'a, b=', alpha, beta            
                
                x0s_dense, T1_analytical, T2_analytical = calculateTs_Kolmogorov_BVP(alpha,beta, tauchar, xmin = -2.5)
                
                T1_empirical = empty_like(x0s)
                T2_empirical = empty_like(x0s)
                
                for idx, x0 in zip(xrange(len(x0s)), x0s):
                    path_tag = OUParams.generatePathTag(alpha,tauchar, beta, x0) + '_'
                    filename = os.path.join(SPIKE_TRAINS_DIR, 'spike_train_N=%d%s'%(N_spikes, path_tag))
                    T = None
                    try:
                        T = SpikeTrain.load(filename)  
                    except IOError:
                        print 'Simulating '
                        simulate_spike_train(N_spikes, save_path=True, path_tag = '',                                        
                                                    params = [alpha, tauchar, beta], x0 = x0)
                        T= SpikeTrain.load(filename)
                    
                    T1_empirical[idx] = T.getEmpiricalMoment(1)
                    T2_empirical[idx] = T.getEmpiricalMoment(2)
                
                #POST-PROCESSING:
                figure()
                subplot(211); hold(True)
                plot(x0s, T1_empirical, 'r+', markersize = 12, markeredgewidth=3, label='Empirical')
                plot(x0s_dense, T1_analytical, 'b', linewidth = 4, label='Analytic');
                title(r'$\alpha=%.2f, \tau=%.2f, \beta=%.2f$'%(alpha,tauchar, beta), fontsize=24)
                ylabel('$T_{(1)}$', fontsize=20); legend()
                
                subplot(212); hold(True)
                plot(x0s, T2_empirical, 'rx',  markersize = 12, markeredgewidth=3, label='Empirical')
                plot(x0s_dense, T2_analytical, 'b', linewidth = 4,label='Analytic');
                legend()
                xlabel('$x_0$', fontsize=24);ylabel('$T_{(2)}$', fontsize=20)
                
                save_fig_name = 'Moments_a=%.0f_b=%.0f_N=%d'%(10*alpha,10*beta,N_spikes)
                file_name = os.path.join(FIGS_DIR, save_fig_name+'.png')
#                print 'saving to ', file_name
#                savefig(file_name) 

def TCsBox():
#    alpha = 1.0;  tauchar = .9; beta = .75;
    alpha = 2.0; tauchar = .75; beta = 1.25; 
    
    x0s_dense, T1_analytical, T2_analytical = calculateTs_Kolmogorov_BVP(alpha,beta, tauchar, xmin = -100.)
    figure()
    subplot(211); hold(True)
#    plot(x0s_dense, T1_analytical, 'b', linewidth = 4, label='Analytic');
#    title(r'$\alpha=%.2f, \tau=%.2f, \beta=%.2f$'%(alpha,tauchar, beta), fontsize=24)
#    ylabel('$T_{(1)}$', fontsize=20); legend()
    plot(x0s_dense[x0s_dense > -1.5], T2_analytical[x0s_dense > -1.5], 'b', linewidth = 4, label='Analytic');
    title(r'$\alpha=%.2f, \tau=%.2f, \beta=%.2f$'%(alpha,tauchar, beta), fontsize=24)
    ylabel('$T_{(2)}$', fontsize=20);
    
    subplot(212); hold(True)
    plot(x0s_dense, T2_analytical, 'b', linewidth = 4,label='Analytic');
    xlabel('$x_0$', fontsize=24);ylabel('$T_{(2)}$', fontsize=20)
    save_fig_name = 'KolmogorovBVPMoments_a=%.0f_t=%.0f_b=%.0f'%(10*alpha,10*tauchar, 10*beta)
    file_name = os.path.join(FIGS_DIR, save_fig_name+'.png')
    print 'saving to ', file_name
#    savefig(file_name) 
    
if __name__ == '__main__':
    from pylab import *
    
    moments_analysis()
#    TCsBox()
    
    show()