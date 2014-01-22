# -*- coding:utf-8 -*-
"""
@author: alex
"""
from __future__ import division

from numpy import *
from scipy.sparse import spdiags, lil_matrix
from scipy.sparse.linalg.dsolve.linsolve import spsolve
from scipy.optimize.zeros import brentq
from copy import deepcopy
from scipy.optimize.optimize import fminbound
from HJBSolver import add_inner_title
from PathSimulator import ABCD_LABEL_SIZE
from matplotlib.pyplot import vlines

regime_tags = {(1.5/.5,1.5):'SUPT_HN',
               (.1/.5 ,1.5):'subt_HN',
               (.1/.5 ,.3):'subt_ln',
               (1.5/.5,.3):'SUPT_ln'}

RESULTS_DIR = '/home/alex/Workspaces/Python/OptSpike/Results/FP_Adjoint/'
FIGS_DIR = '/home/alex/Workspaces/Latex/OptSpike/Figs/FP_Adjoint'

import os
for D in [FIGS_DIR, RESULTS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)
import time


#import ext_fpc
class FPSolver():    
    def __init__(self, dx, dt, Tf, x_min, x_thresh = 1.0):  
        #DISCRETIZATION:
        self.rediscretize(dx, dt, Tf, x_min, x_thresh)

    #Grid management routines:    
    def rediscretize(self, dx, dt, Tf, x_min, x_thresh = 1.0):
        self._xs, self._dx = self._space_discretize(dx, x_min, x_thresh)
        self._ts, self._dt = self._time_discretize(Tf,dt)
    
    def _space_discretize(self, dx, x_min, x_thresh = 1.0):
        xs = arange(x_thresh, x_min - dx, -dx)[-1::-1];
        return xs,dx
#        num_steps = ceil( (x_thresh - x_min) / dx)
#        xs = linspace(x_min, x_thresh, num_steps)
#        dx = xs[1]-xs[0];
#        return xs, dx
    
    def _time_discretize(self, Tf, dt):
        num_steps = ceil( Tf/ dt )
        ts = linspace(.0, Tf, num_steps)
        dt = ts[1]-ts[0];
        return ts, dt
    
    def getTf(self):
        return self._ts[-1]
    def setTf(self, Tf):
        self._ts, self._dt = self._time_discretize(Tf, self._dt)
    def getXthresh(self):
        return self._xs[-1]
    def getXmin(self):
        return self._xs[0]
    def setXmin(self, x_min):
        self._xs, self._dx = self._space_discretize(x_min, self._dx)

    @classmethod
    def calculate_xmin(cls, alpha_bounds, params, num_std = 2.0):     
        XMIN_AT_LEAST = -.5;   
        mu, tc, beta = [x for x in params]
        alpha_min = alpha_bounds[0]
        xmin = tc*alpha_min - num_std*beta*sqrt(tc/2.0);
        return min([XMIN_AT_LEAST, xmin])
    @classmethod
    def calculate_dx(cls, alpha_bounds, params, xmin, factor = 1e-1, xthresh = 1.0):
        #PEclet number based calculation:
        mu, tc, beta = [x for x in params]
        max_speed = abs(mu) + max(alpha_bounds) + max([xmin, xthresh]) / tc;
        return factor * (beta / max_speed);
    @classmethod
    def calculate_dt(cls, alpha_bounds, params, dx, xmin, factor=2., xthresh = 1.0):
        mu, tc, beta = params[0], params[1], params[2]        
        max_speed = abs(mu) + max(alpha_bounds) + max([xmin, xthresh]) / tc;
        return factor * (dx / max_speed) 
        
    def _num_nodes(self):
        return len(self._xs)
    def _num_steps (self):
        return len(self._ts)
    
    
    def _getTCs(self, xs, alpha, tauchar, beta):
        from HJB_TerminalConditioner import calculateTs_Kolmogorov_BVP
        #WARNING! TODO: Arbitrary magic number in the lower limit of how to calculate the Kolmogorov Moments Equation. 
        XMIN_OFFSET = -10.0
        T2_xs, T1, T2 = calculateTs_Kolmogorov_BVP(alpha,beta, tauchar, xmin = xs[0] + XMIN_OFFSET)
        TCs = interp(xs, T2_xs, T2)
        return TCs
    
    def _getICs(self, xs, alpha0, beta):
        #WARNING! TODO: HOw do you choose 'a' correctly! 
        a = .1;
        pre_ICs = exp(-xs**2 / a**2) / (a * sqrt(pi))
        ICs = pre_ICs / (sum(pre_ICs)*self._dx) 
        return ICs

    ###########################################
    def solve(self, params, alphas, alpha_max, energy_eps, 
              visualize=False, save_fig=False):
        #Indirection method
        
        fs = self._fsolve( params, alphas, visualize, save_fig)
        
        xs, ts = self._xs, self._ts;
        
        J = self.calcCost(fs, energy_eps,
                          alphas, alpha_max, params)
        
        return xs, ts, fs, J
    
    ###########################################
    def calcCost(self,fs, energy_eps, alphas, alpha_max, params):
        xs, ts = self._xs, self._ts;
        T = ts[-1];
        dx, dt = self._dx, self._dt;
        mu, tc, beta = [x for x in params]
        
        Ttwo = self._getTCs(xs, alpha_max+mu, tc, beta)
        f_terminal = fs[:,-1];
        
        D = beta * beta / 2.; #the diffusion coeff
        outflow =  -D * (-fs[-2,:]) / dx 
        boundary_cost = (ts-T)*(ts-T)
        
        remaining_mass = sum(fs, axis = 0) * dx
                
        J  = sum(Ttwo * f_terminal)*dx \
             + sum(outflow*boundary_cost) * dt \
             + energy_eps*sum(alphas*alphas * remaining_mass)*dt
             
        return J
     
    def _fsolve(self, params, alphas, visualize=False, save_fig=False):
        mu, tauchar, beta = [x for x in params]
        
        dx, dt = self._dx, self._dt;
        xs, ts = self._xs, self._ts;
        
        if visualize:
            print 'tauchar = %.2f,  beta = %.2f,' %(tauchar, beta)
            print 'Tf = %.2f' %self.getTf()
            print 'xmin = %.f, dx = %f, dt = %f' %(self.getXmin(), dx,dt)
        
        #Allocate memory for solution:
        fs = zeros((self._num_nodes(),
                    self._num_steps() ));
        #Impose Dirichlet BCs: = Automatic 
        #Impose ICs: 
        fs[:,0] = self._getICs(xs, alphas[0], beta)
        
        if visualize:
            figure()
            subplot(311)
            plot(xs, fs[:,-1]); 
            title(r'$\alpha=%.2f, \tau=%.2f, \beta=%.2f$'%(alphas[0],tauchar, beta) + ':ICs', fontsize = 24);
            xlabel('x'); ylabel('f')
             
            subplot(312)
            plot(ts, fs[-1, :]);
            title('BCs at xth', fontsize = 24) ; xlabel('t'); ylabel('f')
            
            subplot(313)
            plot(ts, alphas);
            title('Control Input', fontsize = 24) ; xlabel('t'); ylabel(r'\alpha')
        
        
        #Solve it using C-N/C-D:
        D = beta * beta / 2.; #the diffusion coeff
        dx_sqrd = dx * dx;
        
        #Allocate mass mtx:    
        active_nodes = self._num_nodes() - 1
        M = lil_matrix((active_nodes, active_nodes));
        
        #Centre Diagonal:        
        e = ones(active_nodes);
        d_on = D * dt / dx_sqrd;
        
        centre_diag = e + d_on;
        M.setdiag(centre_diag)
        
        soln_fig = None;  
        if visualize:
            soln_fig = figure()
        
        for tk in xrange(1, self._num_steps()):
            #Rip the forward-in-time solution:
            f_prev = fs[:,tk-1];

            #Rip the control:
            alpha_prev = alphas[tk-1]
            alpha_next = alphas[tk]
            
            #Calculate the velocity field
            U_prev = (mu + alpha_prev - xs/ tauchar)
            U_next = (mu + alpha_next - xs/ tauchar)
            
            
            #Form the RHS:
            L_prev = -(U_prev[2:]*f_prev[2:] - U_prev[:-2]*f_prev[:-2]) / (2.* dx) + \
                      D * diff(f_prev, 2) / dx_sqrd;  
            
            #impose the x_min BCs: homogeneous Newmann: and assemble the RHS: 
            RHS = r_[0.,
                     f_prev[1:-1] + .5 * dt * L_prev];
            
            #Reset the Mass Matrix:
            #Lower Diagonal
            u =  U_next / (2*dx);
            d_off = D / dx_sqrd;
                    
            L_left = -.5*dt*(d_off + u[:-2]);
            M.setdiag(L_left, -1);
            
            #Upper Diagonal
            L_right = -.5*dt*(d_off - u[2:]);
            M.setdiag(r_[NaN,
                         L_right], 1);
            #Bottome BCs:
            M[0,0] = U_next[0] + D / dx;
            M[0,1] = -D / dx;
            
            #add the terms coming from the upper BC at the backward step to the end of the RHS
            #RHS[-1] += 0 #Here it is 0!!!
            
            #Convert mass matrix to CSR format:
            Mx = M.tocsr();            
            #and solve:
            f_next = spsolve(Mx, RHS);
            
            #Store solutions:
            fs[:-1, tk] = f_next;
                          
            if visualize:
                mod_steps = 4;  num_cols = 4;
                num_rows = ceil(double(self._num_steps())/num_cols / mod_steps)
                
                step_idx = tk;
                
                if 0 == mod(step_idx,mod_steps) or 1 == tk:
                    plt_idx = floor(tk / mod_steps) + 1
                    ax = soln_fig.add_subplot(num_rows, num_cols, plt_idx)
                    ax.plot(xs, fs[:,tk], label='k=%d'%tk); 
                    if 1 == tk:
                        ax.hold(True)
                        ax.plot(xs, fs[:,tk-1], 'r', label='ICs')
                    ax.legend(loc='upper left')
#                        ax.set_title('k = %d'%tk); 
                    if False : #(self._num_steps()-1 != tk):
                        ticks = ax.get_xticklabels()
                        for t in ticks:
                            t.set_visible(False)
                    else:
                        ax.set_xlabel('$x$'); ax.set_ylabel('$f$')
                        for t in ax.get_xticklabels():
                            t.set_visible(True)
                     
        #Return:
        if visualize:
            for fig in [soln_fig]:
                fig.canvas.manager.window.showMaximized()

            if save_fig:
                file_name = os.path.join(FIGS_DIR, 'f_t=%.0f_b=%.0f.png'%(10*tauchar, 10*beta))
                print 'saving to ', file_name
                soln_fig.savefig(file_name)
                
        return fs

   
########################
class FPSwitchSolution():
    def __init__(self,
                 params, xs, ts, fs,
                 cs_optimal, opt_val,
                 switch_time):
        self._ts  = ts;
        self._xs  = xs;
        self._fs = fs;
        self._cs_optimal = cs_optimal
        self._switch_t = switch_time
        
        self._J_opt = opt_val;
        
        self._mu = params[0]
        self._tau_char = params[1]
        self._beta = params[2]
        
        
    
    def getControls(self):
        return self._cs_optimal
                
    def save(self, file_name=None):
#        path_data = {'path' : self}
        if None == file_name:
            file_name = 'FPSwitchSoln_m=%.1f_b=%.1f_Tf=%.1f'%(self._mu,
                                                         self._beta,
                                                         self._ts[-1]);
        print 'saving path to ', file_name
        file_name = os.path.join(RESULTS_DIR, file_name + '.fps')
        import cPickle
        dump_file = open(file_name, 'wb')
        cPickle.dump(self, dump_file, 1) # 1: bin storage
        dump_file.close()
        
    @staticmethod
    def load(file_name=None, mu_beta_Tf=None):
        ''' not both args can be None!!!'''
        if None == file_name:
            mu,beta,Tf = [x for x in mu_beta_Tf]
            file_name = 'FPSwitchSoln_m=%.1f_b=%.1f_Tf=%.1f'%(mu,
                                                         beta,
                                                         Tf);

        file_name = os.path.join(RESULTS_DIR, file_name + '.fps') 
        print 'loading ', file_name
        import cPickle
        load_file = open(file_name, 'r')
        soln = cPickle.load(load_file)        
        return soln
########################
def calculateOptimalControl(params, Tf,
                              energy_eps,
                              alpha_bounds,
                              J_tol = 1e-2, K_max=100,
                              visualize=True):
    print 'Brent Optimizer:'
    xmin = FPSolver.calculate_xmin(alpha_bounds, params, num_std = 1.0)
    dx = FPSolver.calculate_dx(alpha_bounds, params, xmin)
    dt = FPSolver.calculate_dt(alpha_bounds, params, dx, xmin, factor = 4.)
    print 'Solver params: xmin, dx, dt', xmin,dx,dt

    #Set up solver
    #TODO: The way you pass params and the whole object-oriented approach is silly. Tf changes for each solve and atb don't, so maybe rething the architecture!!!
    S = FPSolver(dx, dt, Tf, xmin)
    ts = S._ts;

    alpha_min, alpha_max = alpha_bounds[0], alpha_bounds[1]
    def generateSwitchControl(switch_point):
        min_es = ones_like(ts); max_es = ones_like(ts)
        min_es[ts>=switch_point] = .0;
        max_es[ts<switch_point] = .0 
        
        return alpha_min*min_es + alpha_max * max_es
    
    def objective(switch_point):
        #Form the controls:
        alphas = generateSwitchControl(switch_point)
        
        #Find the solution
        xs, ts, fs, J =  S.solve(params,
                                alphas,
                                 alpha_bounds[1],
                                  energy_eps,
                                   visualize=False)
       
        #The associated cost
        return J;

    #TODO: Remove
    figure()
    switch_ts = linspace(-.1, Tf+.1, 12)
    Js = [objective(t) for t in switch_ts]
    plot(switch_ts, Js)
    title('REGIME=%s'%regime_tags[(params[0],params[2])])
    vlines(Tf, .0, 2.0)

    
#    from scipy.optimize import brent
#    opt_switch, J_opt, iterations, funcals = brent(objective,
#                                             brack=[0, Tf], tol=J_tol,
#                                             full_output=True, maxiter=K_max)
    
    from scipy.optimize import fminbound
    opt_switch, J_opt, ierrr, numfuncals = fminbound(objective,
                                                     0, Tf, xtol=5e-2,
                                                     maxfun=K_max, full_output=True)
    
    opt_controls = generateSwitchControl(opt_switch)
    xs, ts, fs, J =  S.solve(params,
                                opt_controls,
                                 alpha_bounds[1],
                                  energy_eps,
                                   visualize=False)
    
    
    opt_switch = min(Tf, max(opt_switch,.0))
    print 'Opt switch time = %.3f'%opt_switch
    
    return xs, ts, fs, opt_controls, J_opt, opt_switch
  


    
def SwitchDriver(params, Tf, save_soln = False,
              energy_eps = .001, alpha_bounds = (-2., 2.)): 
    xs, ts, fs, cs_optimal, opt_val, opt_switch = calculateOptimalControl(params, Tf,
                                                      energy_eps,
                                                      alpha_bounds,
                                                      visualize=True)
    
    #SAVE RESULS:
    (FPSwitchSolution(params, xs, ts,
                       fs,cs_optimal,
                        opt_val, opt_switch)).save()

def solveRegimes(regimeParams, Tf):
    for params in regimeParams:
        print 'm,tc,b =' , params
        SwitchDriver(params, Tf, save_soln=True)
    

def visualizeRegimes(regimeParams, Tf=  1.5, 
                     fig_name = None,
                     ):
    label_font_size = 24
    xlabel_font_size = 32
    
#    fig = figure(figsize = (17, 20))
#    subplots_adjust(hspace = .1,wspace = .1,
#                     left=.025, right=.975,
#                     top = .95, bottom = .05)
        
    inner_titles = {0:'A',
                    1:'B',
                    2:'C',
                    3:'D'}

    cs_fig = figure(figsize = (17, 24))
    subplots_adjust(hspace = .15,wspace = .2,
                     left=.1, right=.975,
                     top = .95, bottom = .05)
    N_regimes = len(regimeParams)
    for pidx, params in enumerate(regimeParams):
        fpSoln = FPSwitchSolution.load(mu_beta_Tf = params[::2]+[Tf])
        print 'mu,tc,b = %.2f,%.2f,%.2f'%(fpSoln._mu,fpSoln._tau_char, fpSoln._beta) 
        ts,xs,cs = fpSoln._ts, fpSoln._xs, fpSoln._cs_optimal
        axc = cs_fig.add_subplot(N_regimes, 1, 1+ pidx)
        axc.plot(ts, cs, linewidth = 3)
#        axc.legend()
#        axc.set_ylim(amin(cs)-.1, amax(cs)+.1);
        axc.set_xlabel('$t$', fontsize = xlabel_font_size);
        axc.set_ylabel(r'$\alpha(t)$', fontsize = xlabel_font_size);
        
        t = add_inner_title(axc, inner_titles[pidx], loc=3,
                              size=dict(size=ABCD_LABEL_SIZE))
        t.patch.set_ec("none"); t.patch.set_alpha(0.5)
        title('REGIME=%s'%regime_tags[(params[0],params[2])])
    
#    get_current_fig_manager().window.showMaximized()        
    if None!= fig_name:
        lfig_name = os.path.join(FIGS_DIR, fig_name + '_cs.pdf')
        print 'saving to ', lfig_name
        savefig(lfig_name)

def visualizeSwitchTimeAsFunctionOfHorizon(params, 
                                           Tfs, 
                                             fig_name = None,
                                             ):
    label_font_size = 24
    xlabel_font_size = 32
    
#    fig = figure(figsize = (17, 20))
#    subplots_adjust(hspace = .1,wspace = .1,
#                     left=.025, right=.975,
#                     top = .95, bottom = .05)
        
    inner_titles = {0:'A',
                    1:'B',
                    2:'C',
                    3:'D'}

    ts_fig = figure(figsize = (17, 24))
    subplots_adjust(hspace = .15,wspace = .2,
                     left=.1, right=.975,
                     top = .95, bottom = .05)
    
    switch_ts = empty_like(Tfs)
    for t_idx, Tf in enumerate(Tfs):
        fpSoln = FPSwitchSolution.load(mu_beta_Tf = params[::2]+[Tf])
        print 'mu,tc,b = %.2f,%.2f,%.2f'%(fpSoln._mu,fpSoln._tau_char, fpSoln._beta) 
#        ts,xs,cs = fpSoln._ts, fpSoln._xs, fpSoln._cs_optimal
        switch_ts[t_idx] = fpSoln._switch_t
       
    ax = ts_fig.add_subplot(111)
    ax.plot(Tfs, switch_ts, linewidth = 3)
#        axc.legend()
#        axc.set_ylim(amin(cs)-.1, amax(cs)+.1);
    ax.set_xlabel('$T^*$', fontsize = xlabel_font_size);
    ax.set_ylabel(r'$t_{sw}$', fontsize = xlabel_font_size);
    
    title('REGIME=%s'%regime_tags[(params[0],params[2])])
    
#    get_current_fig_manager().window.showMaximized()        
    if None!= fig_name:
        lfig_name = os.path.join(FIGS_DIR, fig_name + '_tswss.pdf')
        print 'saving to ', lfig_name
        savefig(lfig_name)
    
    
  
if __name__ == '__main__':
    from pylab import *
    Tf = 1.5; 
    energy_eps = .001; alpha_bounds = (-2., 2.);
    
    tau_char = .5;
    beta_high = 1.5
    beta_low = .3;
    mu_high = 1.5
    mu_low = .1
#    regimeParams = [ [mu_high/tau_char, tau_char, beta_high] ]
#    regimeParams =     [ [ mu_high/tau_char, tau_char, beta_high]] 
    regimeParams = [ [mu_high/tau_char, tau_char, beta_high],
                     [mu_high/tau_char, tau_char, beta_low],
                     [mu_low/tau_char, tau_char, beta_high],
                     [mu_low/tau_char, tau_char, beta_low]   ]
    
#    solveRegimes(regimeParams, Tf)
#    visualizeRegimes(regimeParams, Tf, fig_name = None)
 
    params = [ [ mu_low/tau_char, tau_char, beta_high]] 
    Tfs = arange(.2, 2.0, .1)
    for Tf in Tfs:
        solveRegimes(params, Tf)
    visualizeSwitchTimeAsFunctionOfHorizon(params[0], Tfs, 
                                           fig_name = 'switch_curve')

    show()
    