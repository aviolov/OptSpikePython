# -*- coding:utf-8 -*-
"""
Created on Oct 31, 2013

@author: alex
"""

from numpy import *
from pylab import *
from pygsl import chebyshev
from AdjointSolver import *

def chebyshevOptControls(regimeParams,
                         regimeTitles, 
                         Tf=  1.5, 
                         energy_eps = .001,
                         fig_name = None):
    label_font_size = 24
    xlabel_font_size = 32
    
    
    fig = figure(figsize = (17, 20))
#    subplots_adjust(hspace = .2,wspace = .2,
#                     left=.1, right=.975,
#                     top = .95, bottom = .05)
    N_regimes = len(regimeParams)
    c_max = 2.0; c_min = -2.0;
    for pidx, params in enumerate(regimeParams):
        fbkSoln = FBKSolution.load(mu_beta_Tf = params[::2]+[Tf],
                                   energy_eps = energy_eps)
#        print 'mu,tc,b,  energy_e = %.2f,%.2f,%.2f,%.3f'%(fbkSoln._mu,
#                                          fbkSoln._tau_char,
#                                          fbkSoln._beta,
#                                          fbkSoln._energy_eps) 
        ts,cs = fbkSoln._ts, fbkSoln._cs_iterates[-1]
        f= lambda x,params : interp(x,ts,cs)
        gslf = chebyshev.gsl_function(f, None) 
        K = 4
        s = chebyshev.cheb_series(K)
        s.init(gslf, ts[0], ts[-1]);
        cheb_cs = empty_like(ts);
        
        for k, t in enumerate(ts):
            cheb_cs[k] = s.eval(t)
        
        ax = fig.add_subplot(N_regimes, 1, 1+ pidx)
        ax.plot(ts, cs, linewidth = 3)
        ax.plot(ts, cheb_cs, label='Cheb_%d'%K)
        ax.set_title('energy_e = %.3f'%fbkSoln._energy_eps);
        ax.legend( );
        
        
        mu,beta = params[::2]
        xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, params, num_std = 1.0)
        dx = FPAdjointSolver.calculate_dx(alpha_bounds, params, xmin)
        dt = FPAdjointSolver.calculate_dt(alpha_bounds, params, dx, xmin, factor = 4.)
        S = FPAdjointSolver(dx, dt, Tf, xmin)
        xs, ts, fs, ps, J_cheb, minus_grad_H =  S.solve(params,
                                                    cheb_cs,
                                                     alpha_bounds[1],
                                                      energy_eps=fbkSoln._energy_eps,
                                                       visualize=False)
        
        dt = FPAdjointSolver.calculate_dt(alpha_bounds, params, dx, xmin, factor = 1.)
        S = FPAdjointSolver(dx, dt, Tf, xmin)
        ts = S._ts;
        cs = interp(ts, fbkSoln._ts, cs);    
        xs, ts, fs, ps, J_refined, minus_grad_H =  S.solve(params,
                                                    cs,
                                                     alpha_bounds[1],
                                                      energy_eps=fbkSoln._energy_eps,
                                                       visualize=False)
        
        print 'mu,tc,b,  energy_e = %.2f,%.2f,%.2f,%.3f: '%(fbkSoln._mu,
                                          fbkSoln._tau_char,
                                          fbkSoln._beta,
                                          fbkSoln._energy_eps)
        print '\t: J = %.3f, J_cheb = %.3f, J_refined = %.3f'%(fbkSoln._J_iterates[-1],
                                                               J_cheb,
                                                               J_refined)   
        
    
    get_current_fig_manager().window.showMaximized()        
    if None!= fig_name:
        lfig_name = os.path.join(FIGS_DIR, fig_name + '_cs.pdf')
        print 'saving to ', lfig_name
        savefig(lfig_name, dpi=300)
        
if __name__ == '__main__':
    Tf = 1.5; 
    energy_eps = .001; alpha_bounds = (-2., 2.);
    
    tau_char = .5;
    beta_high = 1.5
    beta_low = .3;
    mu_high = 1.5
    mu_low = .1
    regimeParams = [[mu_high/tau_char, tau_char, beta_low],
                    [mu_high/tau_char, tau_char, beta_high],
                     [mu_low/tau_char, tau_char, beta_low],
                     [mu_low/tau_char, tau_char, beta_high]]
    regimeTitles = {(mu_high/tau_char, beta_low) :'SuperT, low-noise',
                      (mu_high/tau_char, beta_high):'SuperT, high-noise', 
                     (mu_low/tau_char, beta_low)  :'SubT, low-noise', 
                     (mu_low/tau_char, beta_high) :'SubT, high-noise'}
    
    for energy_eps in [.001]:
        chebyshevOptControls(regimeParams,
                             regimeTitles, 
                             Tf,
                             energy_eps)
    
    show()