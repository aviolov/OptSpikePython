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
from PathSimulator import ABCD_LABEL_SIZE
from matplotlib.font_manager import FontProperties
from scipy import interpolate

RESULTS_DIR = '/home/alex/Workspaces/Python/OptSpike/Results/FP_Adjoint/'
FIGS_DIR = '/home/alex/Workspaces/Latex/OptSpike/Figs/FP_Adjoint'

import os
for D in [FIGS_DIR, RESULTS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)
import time


label_font_size = 32
xlabel_font_size = 40

def deterministicControlHarness(params,
                                 Tf = 1.5,    
                                   alpha_bounds = (-2., 2.)):
    mu, tau_char = params[0], params[1]
    xth = 1.0
#    \frac{\xth}{\tc(1 - e^{-\T/\tc})} - \m$$
    alpha_constant = xth / (tau_char * (1. - exp(-Tf/tau_char) )) - mu  
    
    return alpha_constant


#def deterministicControlHarness(params = [.1, .75, 1.25],
#                                 Tf = 1.5, energy_eps = .001, alpha_bounds = (-2., 2.), visualize=False, fig_name = None):
#    from scipy.integrate import odeint
#    xth = 1.0;
#    
#    mu, tauchar = params[0], params[1];
#    print r'\tc=', tauchar
#    a_max = alpha_bounds[1]
#    print r'\amax=', a_max
#    print r'\T=', Tf
#    print r'\e=', energy_eps
#    
#    def alpha(t, p_0):
#        return amin( [p_0 * exp(t/tauchar) / (2*energy_eps), a_max]  )
##        return amin(c_[p_0 * exp(t/tauchar) / (2*energy_eps), amax*ones(len(t))], axis=1)
#    
#    def dx(x, t, p_0):
#        return mu + alpha(t,p_0) - x /tauchar
#        
#    ts = linspace(.0, Tf, 100)
#    def p0_root(p0):
#        xs = odeint(dx, .0, ts, args=(p0,))
#        return xs[-1,0] - xth
#
##    if (p0_root(.0)*p0_root(a_max* energy_eps * 2) > 0):
##        '''same sign - return amax'''
##        return ts, a_max * ones_like(ts)
#                
#    p0 = brentq(p0_root, -.01, a_max* energy_eps * 2)
#    
#    xs = odeint(dx, .0, ts, args = (p0,))
#    alphas = [alpha(t,p0) for t in ts]
#    
#    if visualize:
#        figure()
#        subplot(211)
#        plot(ts,xs, linewidth=4); xlim((.0, Tf)) 
#        title(r'State Evolution: $\tau_c=%.2f,\alpha_{max}=%.2f$'%(tauchar, a_max), fontsize=24); ylabel('$x(t)$', fontsize=24)
#        subplot(212)
#        plot(ts,alphas, linewidth=4); xlim((.0, Tf))
#        title('Control Evolution')
#        xlabel('t', fontsize=24); ylabel(r'$\alpha(t)$', fontsize=24)
#        get_current_fig_manager().window.showMaximized()
#        
#        if None != fig_name:
#            file_name = os.path.join(FIGS_DIR, fig_name+ '.png')
#            print 'saving to ', file_name
#            savefig(file_name)
#    
#    return ts, alphas

#import ext_fpc
class FPAdjointSolver():    
    TAUCHAR_INDEX = 1;
    BETA_INDEX = 2
    
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
    
    def _getAdjointBCs(self, ts, T):
        return (ts-T)*(ts-T)

    ###########################################
    def solve(self, params, alphas, alpha_max, energy_eps, visualize=False, save_fig=False):
        #Indirection method
        
        fs = self._fsolve( params, alphas, visualize, save_fig)
        ps = self._psolve( params, alphas, alpha_max, visualize, save_fig)

        xs, ts = self._xs, self._ts;
        
        J = self.calcCost(energy_eps, alphas, params[2], fs,ps)
        
        #the Hamiltonian gradient:
        #NOTE THAT WE need to divide by dx in approximating the derivative and multiply by dx in approximating the integral so we just drop that. 
        dxps = diff(ps, axis=0);
        dxp_times_f = sum(fs[1:,:]*dxps, axis=0)
#        This term is zero: fs[-1,:], it's probably included here for comprehensiveness
#        This term should be close to zero: fs[0,:]
        minus_grad_H = -(2*energy_eps*alphas - ps[-1,:]*fs[-1,:] + 2*ps[0,:]*fs[0,:] + dxp_times_f)   
   
        return xs, ts, fs, ps, J, minus_grad_H
    
    ###########################################
    def calcCost(self,energy_eps, alphas, beta, fs,ps):
        xs, ts = self._xs, self._ts;
        dx, dt = self._dx, self._dt;
        
        Ttwo = ps[:, -1]
        f_terminal = fs[:,-1];
        
        D = beta * beta / 2.; #the diffusion coeff
        outflow =  -D * (-fs[-2,:]) / dx 
        boundary_cost = ps[-1, :];
        
        remaining_mass = sum(fs, axis = 0) * dx
                
        J  = sum(Ttwo * f_terminal)*dx \
             + sum(outflow*boundary_cost) * dt \
             + energy_eps*sum(alphas*alphas * remaining_mass)*dt
             
        return J
        
        
    
    def _psolve(self, params, alphas, alpha_max, visualize=False, save_fig=False):
        mu, tauchar, beta = [x for x in params]
        
        dx, dt = self._dx, self._dt;
        xs, ts = self._xs, self._ts;
        
        if visualize:
            print 'tauchar = %.2f,  beta = %.2f,' %(tauchar, beta)
            print 'amax = %.2f,'%alpha_max
            print 'Tf = %.2f' %self.getTf()
            print 'xmin = %.f, dx = %f, dt = %f' %(self.getXmin(), dx,dt)
        
        #Allocate memory for solution:
        ps = zeros((self._num_nodes(),
                    self._num_steps() ));
                    
        #Impose TCs: 
        ps[:,-1] = self._getTCs(xs, alpha_max+mu, tauchar, beta)
        
        #Impose BCs at upper end: 
        ps[-1,:] = self._getAdjointBCs(ts, self.getTf())
        
        if visualize:
            figure()
            subplot(311)
            plot(xs, ps[:,-1]); 
            title(r'$\alpha=%.2f, \tau=%.2f, \beta=%.2f$'%(alphas[-1], tauchar, beta) +
                   ':TCs', fontsize = 24);
            xlabel('x'); ylabel('p')
             
            subplot(312)
            plot(ts, ps[-1, :]);
            title('BCs at xth', fontsize = 24) ; xlabel('t'); ylabel('p')
            
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
        
        for tk in xrange(self._num_steps()-2,-1, -1):
            #Rip the forward-in-time solution:
            p_forward = ps[:,tk+1];

            #Rip the control:
            alpha_forward = alphas[tk+1]
            alpha_current = alphas[tk]
            
            #Calculate the velocity field
            U_forward = (mu + alpha_forward - xs[1:-1]/ tauchar)
            U_current = (mu + alpha_current - xs[1:-1]/ tauchar)
            
            #Form the RHS:
            L_forward = U_forward*(p_forward[2:] - p_forward[:-2]) / (2.* dx) + \
                        D        * diff(p_forward, 2) / dx_sqrd;  
            
            #Impose the x_min BCs: homogeneous Newmann: and assemble the RHS: 
            RHS = r_[0.,
                     p_forward[1:-1] + .5 * dt * L_forward];
            
            #Reset the Mass Matrix:
            #Lower Diagonal
            u =  U_current / (2*dx);
            d_off = D / dx_sqrd;
                    
            L_left = -.5*dt*(d_off - u[1:-1]);
            M.setdiag(L_left, -1);
            
            #Upper Diagonal
            L_right = -.5*dt*(d_off + u[1:]);
            M.setdiag(r_[NaN,
                         L_right], 1);
            #Bottom BCs:
            M[0,0] = -1.; M[0,1] = 1.;
            
            #add the terms coming from the upper BC at the backward step to the end of the RHS
            p_upper_boundary = ps[-1,tk];
            RHS[-1] += .5* dt*(D * p_upper_boundary / dx_sqrd + U_current[-1] *p_upper_boundary / (2*dx) )
            
            #Convert mass matrix to CSR format:
            Mx = M.tocsr();            
            #and solve:
            p_current = spsolve(Mx, RHS);
            
            #Store solutions:
            ps[:-1, tk] = p_current;
                          
            if visualize:
                mod_steps = 4;  num_cols = 4;
                num_rows = ceil(double(self._num_steps())/num_cols / mod_steps) + 1
                
                step_idx = self._num_steps() - 2 - tk;
                
                if 0 == mod(step_idx,mod_steps) or 0 == tk:
                    plt_idx = 1 + floor(tk / mod_steps) + int(0 < tk)
                    ax = soln_fig.add_subplot(num_rows, num_cols, plt_idx)
                    ax.plot(xs, ps[:,tk], label='k=%d'%tk); 
                    if self._num_steps() - 2 == tk:
                        ax.hold(True)
                        ax.plot(xs, ps[:,tk+1], 'r', label='TCs')
                    ax.legend(loc='upper left')
#                        ax.set_title('k = %d'%tk); 
                    if False : #(self._num_steps()-1 != tk):
                        ticks = ax.get_xticklabels()
                        for t in ticks:
                            t.set_visible(False)
                    else:
                        ax.set_xlabel('$x$'); ax.set_ylabel('$p$')
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
                
        return ps
    
    
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



def visualizeAdjointSolver(tb = [.6, 1.25], Tf = 1.5, energy_eps = .001, alpha_bounds = (-2., 2.),
                           fig_name = None):
    mpl.rcParams['figure.subplot.left'] = .1
    mpl.rcParams['figure.subplot.right'] = .95
    mpl.rcParams['figure.subplot.bottom'] = .1
    mpl.rcParams['figure.subplot.top'] = .9
        
    xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, tb, num_std = 1.0)
    dx = FPAdjointSolver.calculate_dx(alpha_bounds, tb, xmin)
    dt = FPAdjointSolver.calculate_dt(alpha_bounds, tb, dx, xmin, factor = 4.)
    
    deterministic_ts, deterministic_control = deterministicControlHarness(tb, Tf, energy_eps, alpha_bounds)

    #Set up solver
    #TODO: The way you pass params and the whole object-oriented approach is silly. Tf changes for each solve and atb don't, so maybe rething the architecture!!!
    S = FPAdjointSolver(dx, dt, Tf, xmin)
    
    ts = S._ts;
    
    alphas = interp(ts, deterministic_ts, deterministic_control)
    
    #the v solution:
    xs, ts, fs, ps =  S.solve(tb, alphas, alpha_bounds[1], visualize=True)
    
    Fth = sum(fs, axis = 0)*S._dx
    figure()
    plot(ts, Fth); xlabel('t'); ylabel('Fth')
    
    #Visualize:
    from mpl_toolkits.mplot3d import Axes3D
    for vs, tag in zip([fs, ps],
                       ['f', 'p']):
        az = -45; #    for az in arange(-65, -5, 20):
        fig = figure();
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.view_init(elev = None, azim= az)
        X, Y = np.meshgrid(ts, xs)
        ax.plot_surface(X, Y, vs, rstride=4, cstride=4, cmap=cm.jet,
                                   linewidth=0, antialiased=False)
        xlabel('t', fontsize = 18); ylabel('x',fontsize = 24)
        title('$'+tag +'(x,t)$', fontsize = 36);    
        get_current_fig_manager().window.showMaximized()




def stylizedVisualizeForwardAdjoint(tb = [.6, 1.25], Tf = 1.5, energy_eps = .001, alpha_bounds = (-2., 2.),
                                        fig_name = None):
    mpl.rcParams['figure.subplot.left'] = .1
    mpl.rcParams['figure.subplot.right'] = .95
    mpl.rcParams['figure.subplot.bottom'] = .1
    mpl.rcParams['figure.subplot.top'] = .9
        
    xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, tb, num_std = 1.0)
    dx = FPAdjointSolver.calculate_dx(alpha_bounds, tb, xmin)
    dt = FPAdjointSolver.calculate_dt(alpha_bounds, tb, dx, xmin, factor = 4.)
    
    deterministic_ts, deterministic_control = deterministicControlHarness(tb, Tf, energy_eps, alpha_bounds)

    #Set up solver
    S = FPAdjointSolver(dx, dt, Tf, xmin)
    ts = S._ts;
    alphas = interp(ts, deterministic_ts, deterministic_control)
    
    #the fs, ps :
    xs, ts, fs, ps, J, minus_grad_H =  S.solve(tb, alphas, alpha_bounds[1], energy_eps,
                                                visualize=False)
    
    figure()
    plot(xs, fs[:, 0], linewidth = 3); 
    xlabel('$x$', fontsize = 16); ylabel('$f$', fontsize = 16)
    get_current_fig_manager().window.showMaximized()
    file_name = os.path.join(FIGS_DIR, fig_name + '_f.png')
    print 'saving to', file_name
    savefig(file_name)
    
    
    figure()
    plot(xs, ps[:, -1], linewidth = 3); 
    xlabel('$x$', fontsize = 16); ylabel('$p$', fontsize = 16)
    get_current_fig_manager().window.showMaximized()
    file_name = os.path.join(FIGS_DIR, fig_name + '_p.png')
    print 'saving to', file_name
    savefig(file_name)
    
    
def compareControlTerm(tb = [.6, 1.25], Tf = 1.5, energy_eps = .001, alpha_bounds = (-2., 2.),
                           fig_name = None):
    mpl.rcParams['figure.subplot.left'] = .1
    mpl.rcParams['figure.subplot.right'] = .95
    mpl.rcParams['figure.subplot.bottom'] = .1
    mpl.rcParams['figure.subplot.top'] = .9
        
    xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, tb, num_std = 1.0)
    dx = FPAdjointSolver.calculate_dx(alpha_bounds, tb, xmin)
    dt = FPAdjointSolver.calculate_dt(alpha_bounds, tb, dx, xmin, factor = 4.)
    
    deterministic_ts, deterministic_control = deterministicControlHarness(tb, Tf, energy_eps, alpha_bounds)

    #Set up solver
    #TODO: The way you pass params and the whole object-oriented approach is silly. Tf changes for each solve and atb don't, so maybe rething the architecture!!!
    S = FPAdjointSolver(dx, dt, Tf, xmin)
    
    ts = S._ts;
    
    alphas = interp(ts, deterministic_ts, deterministic_control)
    
    #the v solution:
    xs, ts, fs, ps =  S.solve(tb, alphas, alpha_bounds[1], visualize=False)
    
    
    #the gradients
    dxfs = diff(fs, axis=0)/S._dx;
    dxps = diff(ps, axis=0)/S._dx;
    
    pdxf = sum(ps[1:,:]*dxfs, axis=0) 
    
    pf_minus_dxpf = (ps[-1,:]*fs[-1,:] - ps[0,:]*fs[0,:]) - sum(fs[1:,:]*dxps, axis=0)  
       
    figure(); hold(True)
    plot(ts, pdxf, 'b', label=r'$\int p \nabla_x f$')
    plot(ts, pf_minus_dxpf, 'r', label=r'$ pf|_{x-}^{x+} - \int f \nabla_x p$')
    legend(loc='upper left')
    
    figure(); hold(True)
    plot(xs[1:], dxfs[:, 1], 'b', label=r'$\nabla_x \, f$')
    plot(xs[1:], dxps[:, 1], 'g', label=r'$\nabla_x \, p$'); xlabel('x')
    legend(loc='upper left')
        

def calcGradH(tb = [.6, 1.25], Tf = 1.5, energy_eps = .001, alpha_bounds = (-2., 2.),
                           fig_name = None):

        
    xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, tb, num_std = 1.0)
    dx = FPAdjointSolver.calculate_dx(alpha_bounds, tb, xmin)
    dt = FPAdjointSolver.calculate_dt(alpha_bounds, tb, dx, xmin, factor = 4.)
    
    deterministic_ts, deterministic_control = deterministicControlHarness(tb, Tf, energy_eps, alpha_bounds)

    #Set up solver
    #TODO: The way you pass params and the whole object-oriented approach is silly. Tf changes for each solve and atb don't, so maybe rething the architecture!!!
    S = FPAdjointSolver(dx, dt, Tf, xmin)
    
    ts = S._ts;
    
    alphas = interp(ts, deterministic_ts, deterministic_control)
    
    #the f,p,J solution:
    xs, ts, fs, ps, J, minus_grad_H =  S.solve(tb, alphas, alpha_bounds[1], energy_eps, visualize=False)
    
    STEP_SIZE = .05;
    
    e = ones_like(alphas); alpha_min, alpha_max = alpha_bounds[0], alpha_bounds[1]
    alpha_next = alphas + minus_grad_H  * STEP_SIZE
    alpha_bounded_below = amax(c_[alpha_min*e, alpha_next], axis=1)
            
    alpha_next = amin(c_[alpha_max*e, alpha_bounded_below], axis=1)
    
    #VISUALIZE:
    mpl.rcParams['figure.subplot.left'] = .1
    mpl.rcParams['figure.subplot.right'] = .95
    mpl.rcParams['figure.subplot.bottom'] = .1
    mpl.rcParams['figure.subplot.top'] = .9
    figure(); hold(True)
    
#    plot(ts, grad_H, 'b', label=r'$\nabla_\alpha \, H$')
    plot(ts, minus_grad_H, 'g', label=r'$-\nabla_\alpha \, H$', linewidth = 4); 
    plot(ts, alphas, 'r--', label=r'$\alpha_0(t)$', linewidth = 4);
    plot(ts, alpha_next, 'b--', label=r'$\alpha_1(t)$', linewidth = 4);
    ylabel(r'$\alpha$', fontsize=24);xlabel('$t$', fontsize=24);    
    legend(loc='upper left')
    title('First Control Iteration', fontsize=36)
    
    if None != fig_name:
        get_current_fig_manager().window.showMaximized()
        file_name = os.path.join(FIGS_DIR, fig_name + '.png')
        print 'saving to ', file_name
        savefig(file_name);
        
    print 'J_0 = ', J
      
    
def calculateOutflow(tb = [.6, 1.25], Tf = 1.5, energy_eps = .001, alpha_bounds = (-2., 2.)):
    xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, tb, num_std = 1.0)
    dx = FPAdjointSolver.calculate_dx(alpha_bounds, tb, xmin)
    dt = FPAdjointSolver.calculate_dt(alpha_bounds, tb, dx, xmin, factor = 4.)
    
    deterministic_ts, deterministic_control = deterministicControlHarness(tb, Tf, energy_eps, alpha_bounds)

    #Set up solver
    #TODO: The way you pass params and the whole object-oriented approach is silly. Tf changes for each solve and atb don't, so maybe rething the architecture!!!
    S = FPAdjointSolver(dx, dt, Tf, xmin)
    
    ts = S._ts;
    
    alphas = interp(ts, deterministic_ts, deterministic_control)
    
    #the f,p,J solution:
    xs, ts, fs, ps, J, minus_gradH =  S.solve(tb, alphas, alpha_bounds[1], energy_eps, visualize=False)
    
    #the
    D = tb[1]**2/2.0
    upwinded_outflow = -D*(-fs[-2,:]) /S._dx
    central_outflow  = -D*(-fs[-2,:]) /(2*S._dx)
    
    upwinded_cumulative_outflow = cumsum(upwinded_outflow)*S._dt;
    central_cumulative_outflow = cumsum(central_outflow)*S._dt;
    
    remaining_mass =sum(fs, axis=0)*S._dx
    
    upwinded_conservation = remaining_mass +upwinded_cumulative_outflow
    central_conservation = remaining_mass   +central_cumulative_outflow
    
    figure(); hold(True)
    plot(ts, upwinded_conservation, 'b', label=r'mass + upwinded outflow')
    plot(ts, central_conservation, 'g', label='mass+central outflow'); 
    plot(ts, sum(fs, axis=0)*S._dx, 'r', label='mass');
    xlabel('t');    legend(loc='upper left')



def timeAdjointSolver(tb = [.5, 1.25], Tf = 1.5, energy_eps = .1, alpha_bounds = (-2., 2.)):
    import time
    
    xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, tb, num_std = 1.0)
    dx = FPAdjointSolver.calculate_dx(alpha_bounds, tb, xmin)
    dt = FPAdjointSolver.calculate_dt(alpha_bounds, tb, dx, xmin, factor = 4.)
    
    deterministic_ts, deterministic_control = deterministicControlHarness(tb, Tf, energy_eps, alpha_bounds)

    #Set up solver
    #TODO: The way you pass params and the whole object-oriented approach is silly. Tf changes for each solve and atb don't, so maybe rething the architecture!!!
    start = time.clock()
    S = FPAdjointSolver(dx, dt, Tf, xmin)
    
    ts = S._ts;
    alphas = interp(ts, deterministic_ts, deterministic_control)
    
    #the f,p solution:
    xs, ts, fs, ps =  S.solve(tb, alphas, alpha_bounds[1], visualize=False)
  
    end = time.clock()
    print 'compute time = ',end-start, 's'
    
def calculateOptimalControl(params, 
                             Tf,
                              energy_eps=.001,
                              alpha_bounds = (-2., 2.),
                              grad_norm_tol = 1e-5,
                              obj_diff_tol = 5e-3,
                              soln_diff_tol = 1e-3,
                              dt_factor =4.,
                              step_size_base = 10.,
                              initial_ts_cs = None,
                               visualize=False):
    #Interface for drivers:
    return gdOptimalControl_Aggressive(params,
                                        Tf,
                                         energy_eps=energy_eps,
                                          alpha_bounds=alpha_bounds,
                                          grad_norm_tol=grad_norm_tol,
                                          obj_diff_tol=obj_diff_tol,
                                          soln_diff_tol=soln_diff_tol,
                                          dt_factor=dt_factor,
                                          step_size_base = step_size_base,
                                          initial_ts_cs=initial_ts_cs,
                                          visualize=visualize)

def gdOptimalControl_Old(params, Tf,
                            energy_eps = .001, alpha_bounds = (-2., 2.),
                            J_tol = 1e-3, gradH_tol = 1e-2, K_max = 100,  
                            alpha_step = .05,
                            visualize=False,
                            initial_ts_cs = None,
                            dt_factor = 4.):
    print 'simple Gradient Descent'
    xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, params, num_std = 1.0)
    dx = FPAdjointSolver.calculate_dx(alpha_bounds, params, xmin)
    dt = FPAdjointSolver.calculate_dt(alpha_bounds, params,
                                       dx, xmin, factor = dt_factor)
    print 'Solver params: xmin, dx, dt', xmin,dx,dt

    #Set up solver
    #TODO: The way you pass params and the whole object-oriented approach is silly. Tf changes for each solve and atb don't, so maybe rething the architecture!!!
    S = FPAdjointSolver(dx, dt, Tf, xmin)
    ts = S._ts;


    min_es = ones_like(ts); max_es = ones_like(ts)
    switch_point = Tf/(1.01)
    min_es[ts>switch_point] = .0; max_es[ts<switch_point] = .0 
    alpha_min, alpha_max = alpha_bounds[0], alpha_bounds[1]
#    initial_control = zeros_like(ts)
#    initial_control = alpha_min*min_es + alpha_max *  max_es;
    initial_control = None;
    
    if (None == initial_ts_cs):
        initial_control = (alpha_max-alpha_min)*ts / Tf + alpha_min
    else:
        initial_control = interp(ts, initial_ts_cs[0], 
                                     initial_ts_cs[1])
#    deterministic_control = deterministicControlHarness(params,Tf, alpha_bounds)
#    initial_control = interp(ts, deterministic_ts, deterministic_control)
#    initial_control = deterministic_control*ones_like(ts)
    alphas = initial_control
    
    alpha_iterations = [alphas]
    J_iterations = []

    J_prev = Inf;
    
    def incrementAlpha(alpha_prev, direction):
        e = ones_like(alpha_prev); 
        alpha_next = alphas + direction * alpha_step
        alpha_bounded_below = amax(c_[alpha_min*e, alpha_next], axis=1)
        return amin(c_[alpha_max*e, alpha_bounded_below], axis=1)
    
    for k in xrange(K_max):
        #the f,p, J, gradJ solution:
        xs, ts, fs, ps, J, minus_grad_H =  S.solve(params, alphas, alpha_bounds[1], energy_eps, visualize=False)
        print k, J
        
        #Convergence check:
        if abs(J - J_prev) < J_tol:
            if visualize:
                print 'J-J_prev = ',  abs(J - J_prev) , ' ==> breaking!'
            break
        else:
            if visualize:
                print 'J-J_prev = ',  abs(J - J_prev)
            J_iterations.append(J);
            J_prev = J
        if amax(abs(minus_grad_H)) < gradH_tol:
            break
        
        alphas = incrementAlpha(alphas, minus_grad_H)
        alpha_iterations.append(alphas)
    
    
    if visualize:   
        mpl.rcParams['figure.subplot.left'] = .1
        mpl.rcParams['figure.subplot.right'] = .95
        mpl.rcParams['figure.subplot.bottom'] = .1
        mpl.rcParams['figure.subplot.top'] = .9     
        
        plot_every = int(floor(k/4));
        controls_fig = figure(); hold(True)
        for iter_idx in [0,1,-2,-1]:
            plot(ts, alpha_iterations[iter_idx],  linewidth = 3, label=str(iter_idx))
        title('Control Convergence', fontsize = 36)
        ylabel(r'$\alpha(t)$',fontsize= 24); xlabel('$t$', fontsize= 24);    legend(loc='upper left')
        
        J_fig = figure();
        plot(J_iterations,  linewidth = 3, label='J_k'); 
        title('Objective Convergence', fontsize = 36)
        ylabel(r'$J_k$',fontsize= 24); xlabel('$k$', fontsize= 24);    legend(loc='upper right')
       
#        if fig_name != None:
#            for fig,tag in zip([J_fig, controls_fig],
#                               ['_objective.png', '_control.png']):
#                fig.canvas.manager.window.showMaximized()
#                file_name = os.path.join(FIGS_DIR, fig_name + tag)
#                print 'saving to ' , file_name
#                fig.savefig(file_name)
            
    return xs, ts, fs, ps, alpha_iterations, J_iterations
  


def ncg4OptimalControl_NocedalWright(params, Tf,
                                    energy_eps = .001, alpha_bounds = (-2., 2.),
                                    J_tol = 1e-3, grad_tol = 1e-3, soln_norm_tol = 1e-3, K_max = 100,
                                    step_tol = .1, step_u_tol = .1, K_singlestep_max = 10,  
                                    orthogonality_tol = .1,
                                    alpha_hat = .5,
                                    visualize=False,
                                    initial_ts_cs = None,
                                    dt_factor = 4.):
    print 'simple Gradient Descent'
    xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, params, num_std = 1.0)
    dx = FPAdjointSolver.calculate_dx(alpha_bounds, params, xmin)
    dt = FPAdjointSolver.calculate_dt(alpha_bounds, params,
                                       dx, xmin, factor = dt_factor)
    print 'Solver params: xmin, dx, dt', xmin,dx,dt

    #Set up solver
    S = FPAdjointSolver(dx, dt, Tf, xmin)
    ts = S._ts;

    alpha_min, alpha_max = alpha_bounds[0], alpha_bounds[1]
    initial_control = None;
    if (None == initial_ts_cs):
        initial_control = (alpha_max-alpha_min)*ts / Tf + alpha_min
    else:
        initial_control = interp(ts, initial_ts_cs[0], 
                                     initial_ts_cs[1])
    alpha_current = initial_control
    
    
    #Initial eval:
    xs, ts, fs, ps, J_current, minus_grad_H = S.solve(params,
                                                       alpha_current,
                                                        alpha_bounds[1],
                                                         energy_eps)
    descent_d = minus_grad_H;
        
#    num_active_nodes= len(ts)-2
    delta_t = ts[1] - ts[0];
    e = ones_like(alpha_current)
    def incrementAlpha(a_k,
                       d_k):
        #Push alpha in direction up to constraints:
        alpha_next = alpha_current + a_k * d_k; 
        alpha_bounded_below = amax(c_[alpha_min*e, alpha_next], axis=1);
        return amin(c_[alpha_max*e, alpha_bounded_below], axis=1)
    
    #The return lists:
    alpha_iterations = [alpha_current]
    J_iterations = [J_current]
    k = 0; ##outer iteration counter
    active_nodes = (alpha_current>alpha_bounds[0]) &  (alpha_current<alpha_bounds[1])
    grad_norm = sqrt(dot(minus_grad_H[active_nodes],
                         minus_grad_H[active_nodes]));
                             
    while (k< K_max and grad_norm > grad_tol * len(active_nodes)):
        #the f,p, J, gradJ solution:
        
        
        minus_grad_H_prev = minus_grad_H;
        soln_norm = sqrt(dot(alpha_current,
                             alpha_current));
        descent_norm = sqrt(dot(descent_d,
                                descent_d));
        print 'k, J_k, ||g_k||, g_tol, ||d_k||, ||c_k||',\
              '=\n %d, %.4f, %.4f, %.4f, %.4f, %.4f'%(k,
                                                    J_current,
                                                    grad_norm,
                                                    grad_tol * len(active_nodes),
                                                    descent_norm,
                                                    soln_norm)
        #Single step search:
        k_ss = 0;
        step_size = 100.;         
        alpha_next, J_next = None, None
        wolfe_1_condition, wolfe_2_condition = False, False
        c1_wolfe = 1e-4; #c1, c2 from page 144 of N+W, wolfe conditions eq. 5.42a,b
        c2_wolfe = 0.1;
        while (k_ss < K_singlestep_max):
            #generate proposed control
            alpha_next = incrementAlpha(a_k=step_size,
                                        d_k=descent_d)
            print '\t|a_k+1 - a_k|= %.4f'%(sum(abs(alpha_next-alpha_current)))
            #evaluate proposed control
            xs, ts, fs, ps, J_next, minus_grad_H =  S.solve(params,
                                                        alpha_next,
                                                         alpha_bounds[1],
                                                          energy_eps)
#            #Sufficient decrease?
            print '\tstep search: k_ss=%d, step_size=%.4f, J=%.4f '%(k_ss, step_size, J_next)
            
            active_nodes = (alpha_next>alpha_bounds[0]) &\
                           (alpha_next<alpha_bounds[1])
            print '\t num active nodes %d / %d'%(len(alpha_current[active_nodes]),
                                                 len(alpha_current));

            cos_descent_dir = dot(minus_grad_H_prev,
                                  descent_d)
            wolfe_1_condition = (J_next <= J_current + c1_wolfe*step_size*cos_descent_dir);
            wolfe_2_condition = (abs(dot(minus_grad_H, descent_d)) <= c2_wolfe*abs(cos_descent_dir));
            print '\t w1:%.3f ? %.3f'%(J_next,
                                       J_current + c1_wolfe*step_size*cos_descent_dir)
            print '\t w2:%.3f ? %.3f'%(abs(dot(minus_grad_H, descent_d)),
                                       c2_wolfe*abs(cos_descent_dir)     ) 
#            if (wolfe_1_condition and wolfe_2_condition):
            if (wolfe_1_condition):
                print 'sufficient decreases for for wolfe{1,2} breaking'
                break;
            #reduce step_size
            step_size *=.8
            k_ss+=1
            
        if K_singlestep_max == k_ss:        
            print 'Single Step Failed::Too many iterations'
        
        alpha_current = alpha_next
        J_current = J_next
        #store latest iteration;
        alpha_iterations.append(alpha_current);
        J_iterations.append(J_current);
        
        #calculate grad_norm:
        grad_norm_squared = dot(minus_grad_H,
                                minus_grad_H)
        grad_norm = sqrt(grad_norm_squared);
                             
        delta_g = minus_grad_H_prev - minus_grad_H#Note that it is in reverse order since the minuses are already included
        
        beta_proposal = dot(minus_grad_H,
                            delta_g) / dot(minus_grad_H_prev,
                                           minus_grad_H_prev);
        beta_PR = max([.0,
                       beta_proposal]);
        print 'beta_PR+=%.3f' %beta_PR
        
        #Restart???
        
        #recompute descent dir:
        descent_d = minus_grad_H + beta_PR*descent_d;
                
        
        k+=1;

    
    if visualize:   
        plot_every = int(floor(k/4));
        controls_fig = figure(); hold(True)
        for iter_idx in [0,1,-2,-1]:
            plot(ts, alpha_iterations[iter_idx],  linewidth = 3, label=str(iter_idx))
        title('Control Convergence', fontsize = 36)
        ylabel(r'$\alpha(t)$',fontsize= 24); xlabel('$t$', fontsize= 24);    legend(loc='upper left')
        
        J_fig = figure();
        plot(J_iterations,  linewidth = 3, label='J_k'); 
        title('Objective Convergence', fontsize = 36)
        ylabel(r'$J_k$',fontsize= 24); xlabel('$k$', fontsize= 24);    legend(loc='upper right')
            
    return xs, ts, fs, ps,\
           alpha_iterations, J_iterations
  


def ncg4OptimalControl_BorziAnnunziato(params, Tf,
                        energy_eps = .001, alpha_bounds = (-2., 2.),
                        J_tol = 1e-3, grad_tol = 1e-3, soln_norm_tol = 1e-3, K_max = 100,
                        step_tol = .1, step_u_tol = .1, K_singlestep_max = 10,  
                        orthogonality_tol = .1,
                        alpha_hat = .5,
                        visualize=False,
                        initial_ts_cs = None,
                        dt_factor = 4.):
    print 'simple Gradient Descent'
    xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, params, num_std = 1.0)
    dx = FPAdjointSolver.calculate_dx(alpha_bounds, params, xmin)
    dt = FPAdjointSolver.calculate_dt(alpha_bounds, params,
                                       dx, xmin, factor = dt_factor)
    print 'Solver params: xmin, dx, dt', xmin,dx,dt

    #Set up solver
    S = FPAdjointSolver(dx, dt, Tf, xmin)
    ts = S._ts;

    alpha_min, alpha_max = alpha_bounds[0], alpha_bounds[1]
    initial_control = None;
    if (None == initial_ts_cs):
        initial_control = (alpha_max-alpha_min)*ts / Tf + alpha_min
    else:
        initial_control = interp(ts, initial_ts_cs[0], 
                                     initial_ts_cs[1])
    alpha_current = initial_control
    
    
    #Initial eval:
    xs, ts, fs, ps, J_current, minus_grad_H = S.solve(params,
                                                       alpha_current,
                                                        alpha_bounds[1],
                                                         energy_eps)
    descent_d = minus_grad_H;
        
#    num_active_nodes= len(ts)-2
    delta_t = ts[1] - ts[0];
        
    def incrementAlpha():
    #Push alpha in direction up to constraints:
        alpha_next = alpha_current + step_size * descent_d; 
        alpha_bounded_below = amax(c_[alpha_min*e,
                                      alpha_next], axis=1);
        return amin(c_[alpha_max*e,
                       alpha_bounded_below], axis=1)
    
    #The return lists:
    alpha_iterations = [alpha_current]
    J_iterations = [J_current]
    k = 0; ##outer iteration counter
    grad_norm = sqrt(dot(minus_grad_H,
                             minus_grad_H));
    while (k< K_max and grad_norm > grad_tol * Tf / delta_t):
        #the f,p, J, gradJ solution:
        
        minus_grad_H_prev = minus_grad_H;
        soln_norm = sqrt(dot(alpha_current,
                             alpha_current));
        descent_norm = sqrt(dot(descent_d,
                                descent_d));
        print 'k, J_k, ||g_k||, ||d_k||, ||c_k|| =%d, %.4f, %.4f, %.4f, %.4f'%(k,
                                                                    J_current,
                                                                    grad_norm,
                                                                    descent_norm,
                                                                    soln_norm)
        #Single step search:
        k_ss = 0;
        step_size_hat = .0;
        delta_decrease = .01; #sufficient decrease constant: put in top-function arg list:
        step_size_base = 1.;
        if 0 == k:
            step_size = step_size_base
        else:
            step_size = min([step_size_base,
                             3*sqrt(dot(alpha_current, alpha_iterations[-2]))/descent_norm]); #initial step size 
        delta_step_size = step_size;
        step_ratio_tol = .1; #used in 'delta_step_size > step_size_hat * step_ratio_tol' convergence criterion
        e = ones_like(alpha_current); 
        print 'step_size_init = %.3f'%step_size
        alpha_next, J_next = None, None
        while (delta_step_size > step_size_hat * step_ratio_tol and
               step_size*descent_norm > soln_norm * soln_norm_tol and
               k_ss < K_singlestep_max):
            
            #generate proposed control
            alpha_next = incrementAlpha()
            #evaluate proposed control
            xs, ts, fs, ps, J_next, minus_grad_H =  S.solve(params,
                                                        alpha_next,
                                                         alpha_bounds[1],
                                                          energy_eps,
                                                           visualize=False)
            #Sufficient decrease?
            print '\tstep search: k_ss=%d, step_size=%.4f, J=%.4f '%(k_ss, step_size, J_next)
            print '\t J_next= %.3f, J_required = %.3f'%(J_next,
                                                        J_current - delta_decrease* step_size*dot(descent_d,
                                                                                                  minus_grad_H))
            if (J_next < J_current - delta_decrease* step_size*dot(descent_d,
                                                                   minus_grad_H)):
                step_size_hat = step_size;            
            delta_step_size /= 2.;
            step_size = step_size_hat + delta_step_size
            k_ss+=1
            print 'Loop Conditions: %d, %d, %d'%(delta_step_size > step_size_hat * step_tol,
                                                 step_size*descent_norm > soln_norm * soln_norm_tol,
                                                 k_ss < K_singlestep_max)
        if K_singlestep_max == k_ss:        
            print 'Single Step Failed::Too many iterations'
        if (1e-8 > step_size_hat):
            print 'Single Step Failed::step_size_hat ~ .0'
        alpha_current = alpha_next
        J_current = J_next
        #store latest iteration;
        alpha_iterations.append(alpha_current);
        J_iterations.append(J_current);
        
        #calculate grad_norm:
        grad_norm_squared = dot(minus_grad_H,
                                minus_grad_H)
        grad_norm = sqrt(grad_norm_squared);
                             
        delta_g = minus_grad_H_prev - minus_grad_H#Note that it is in reverse order since the minuses are already included
        
        beta_DY = None;
        if abs(dot(minus_grad_H, minus_grad_H_prev)) / grad_norm_squared > orthogonality_tol:
            beta_DY = .0;
        else:
            beta_DY = grad_norm_squared / dot(descent_d,
                                              delta_g);  
        print 'beta_DY=%.3f' %beta_DY
        
        #recompute descent dir:
        descent_d = minus_grad_H + beta_DY*descent_d;
                
        
        k+=1;

    
    if visualize:   
        plot_every = int(floor(k/4));
        controls_fig = figure(); hold(True)
        for iter_idx in [0,1,-2,-1]:
            plot(ts, alpha_iterations[iter_idx],  linewidth = 3, label=str(iter_idx))
        title('Control Convergence', fontsize = 36)
        ylabel(r'$\alpha(t)$',fontsize= 24); xlabel('$t$', fontsize= 24);    legend(loc='upper left')
        
        J_fig = figure();
        plot(J_iterations,  linewidth = 3, label='J_k'); 
        title('Objective Convergence', fontsize = 36)
        ylabel(r'$J_k$',fontsize= 24); xlabel('$k$', fontsize= 24);    legend(loc='upper right')
            
    return xs, ts, fs, ps,\
           alpha_iterations, J_iterations
  


def gdOptimalControl_Aggressive(params, Tf,
                                energy_eps = .001,
                                alpha_bounds = (-2., 2.),
                                grad_norm_tol = 1e-5,
                                soln_diff_tol = 1e-3, #this should be related to alpha_min,max
                                obj_diff_tol =  5e-3, #we want three sig digids; 
                                K_max = 100,
                                K_singlestep_max = 10,
                                step_size_base = 10.,
                                step_size_reduce_factor = .5,
                                visualize=False,
                                initial_ts_cs = None,
                                dt_factor = 4.):
    print 'Aggresive Gradient Descent: TODO: Redefine active nodes to include those at the boundary but pointing inwards'
    xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, params, num_std = 1.0)
    dx = FPAdjointSolver.calculate_dx(alpha_bounds, params, xmin)
    dt = FPAdjointSolver.calculate_dt(alpha_bounds, params,
                                       dx, xmin, factor = dt_factor)
    print 'Solver params: xmin, dx, dt', xmin,dx,dt

    #Set up solver
    S = FPAdjointSolver(dx, dt, Tf, xmin)
    ts = S._ts;

    alpha_min, alpha_max = alpha_bounds[0], alpha_bounds[1]
    initial_control = None;
    if (None == initial_ts_cs):
        initial_control = (alpha_max-alpha_min)*ts / Tf + alpha_min
    else:
        initial_control = interp(ts, initial_ts_cs[0], 
                                     initial_ts_cs[1])
    alpha_current = initial_control
    
#    
#    #Initial eval:
#    xs, ts, fs, ps, J_current, minus_grad_H = S.solve(params,
#                                                       alpha_current,
#                                                        alpha_bounds[1],
#                                                         energy_eps)
#    descent_d = minus_grad_H;
        
#    num_active_nodes= len(ts)-2
#    delta_t = ts[1] - ts[0];
    e = ones_like(alpha_current)
    
    def incrementAlpha(a_k,
                       d_k):
        #Push alpha in direction up to constraints:
        alpha_next = alpha_current + a_k * d_k; 
        alpha_bounded_below = amax(c_[alpha_min*e, alpha_next], axis=1);
        return amin(c_[alpha_max*e, alpha_bounded_below], axis=1)
    
    #The return lists:
    alpha_iterations = []
    J_iterations = []
    
    xs, ts, fs, ps, J_current, minus_grad_H = S.solve(params,
                                               alpha_current,
                                                alpha_bounds[1],
                                                 energy_eps);
    step_size = step_size_base;
    for k in xrange(K_max):                         
#    while (k< K_max and grad_norm > grad_tol * len(active_nodes)):
        #the f,p, J, gradJ solution:
        #Calculate descent direction:
        alpha_iterations.append(alpha_current);
        J_iterations.append(J_current);
                                                         
        active_nodes = (alpha_current>alpha_bounds[0]) &  (alpha_current<alpha_bounds[1])
        print 'active_nodes = %d'%(len(alpha_current[active_nodes]))
        active_grad_norm = sqrt(dot(minus_grad_H[active_nodes],
                                    minus_grad_H[active_nodes]));
        effective_grad_tol = grad_norm_tol * len(alpha_current[active_nodes])
        if active_grad_norm <= effective_grad_tol:
            print 'active grad_norm = %.6f < %.6f, convergence!'%(active_grad_norm,
                                                                  effective_grad_tol);
            break
                         
        #Single line minimization: (step_size selection:
         
        print 'k=%d, J_k=%.4f, ||g_k||_active=%.4f, g_tol_effective=%.4f,'%(k,
                                                J_current,
                                                active_grad_norm,
                                                effective_grad_tol)
        #Single step search:
#        step_size /=step_size_reduce_factor;    #try to be a little more aggressive
        alpha_next, J_next = None, None
        single_step_failed = False;
        for k_ss in xrange(K_singlestep_max):
            #generate proposed control
            alpha_next = incrementAlpha(a_k=step_size,
                                        d_k=minus_grad_H);
#            print '\t|a_{k+1} - a_k|= %.4f'%(sum(abs(alpha_next-alpha_current)))
            #evaluate proposed control
            xs, ts, fs, ps, J_next, minus_grad_H =  S.solve(params,
                                                        alpha_next,
                                                         alpha_bounds[1],
                                                          energy_eps)
#            #Sufficient decrease?
            print '\tk_ss=%d, step_size=%.4f, J=%.4f '%(k_ss, step_size, J_next)
            
             
            
#            sufficent_decrease = J_current - c1_wolfe*step_size * active_grad_norm*active_grad_norm
            sufficent_decrease = J_current;
            wolfe_1_condition = (J_next <= sufficent_decrease);
            
            if (wolfe_1_condition):
                print '\t sufficient decrease: %.6f < %.6f breaking' %(J_next,
                                                                       sufficent_decrease);
                step_size /= step_size_reduce_factor;
                step_size = min([10*step_size_base,
                                 step_size]); #make sure it's not too big!                                    
                break;
#            if step_size < step_size_tol:
#                print 'too many '
                single_step_failed = True;
            if K_singlestep_max-1 == k_ss:        
                single_step_failed = True;
            #reduce step_size
            step_size *=step_size_reduce_factor
            ###Single step loop
            
        if single_step_failed:
            break;
            
#        #calculate grad_norm:
#        delta_soln = alpha_next - alpha_current;
#        delta_J = J_next - J_current;
#        active_soln_diff_norm = sqrt(dot(delta_soln[active_nodes],
#                                        delta_soln[active_nodes]));
#        print 'active_soln_diff = %.6f, J_diff = %.6f'%(active_soln_diff_norm,
#                                                        abs(delta_J))
#        print 'active_soln_diff_tol = %.6f, J_diff_rel_tol = %.6f'%(soln_diff_tol*len(alpha_current[active_nodes]),
#                                                                                       obj_diff_tol)
#        if (active_soln_diff_norm <= soln_diff_tol*len(alpha_current[active_nodes])) and \
#            (abs(delta_J)/J_current <= obj_diff_tol):
#            print 'convergence!'
#            break
#        else:
        #Update current control, objective
        alpha_current = alpha_next        
        J_current = J_next
        ###Main Loop
    
    if visualize:   
        plot_every = int(floor(k/4));
        controls_fig = figure(); hold(True)
        iter_ids = [0,-1]
        for iter_idx in iter_ids:
            plot(ts, alpha_iterations[iter_idx],  linewidth = 3, label=str(iter_idx))
        title('Control Convergence', fontsize = 36)
        ylabel(r'$\alpha(t)$',fontsize= 24); xlabel('$t$', fontsize= 24);    legend(loc='upper left')
        
        J_fig = figure();
        plot(J_iterations,  linewidth = 3, label='J_k'); 
        title('Objective Convergence', fontsize = 36)
        ylabel(r'$J_k$',fontsize= 24); xlabel('$k$', fontsize= 24);    legend(loc='upper right')
            
    return xs, ts, fs, ps, alpha_iterations, J_iterations
  


def exactStepOptimalControl(tb = [.6, 1.25], Tf = 1.5, energy_eps = .001, alpha_bounds = (-2., 2.),
                                    J_tol = .001, gradH_tol = .1, K_max = 100,  
                                    visualize=False, fig_name = None):
    print 'Congugate-Gradient Descent'
    
    xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, tb, num_std = 1.0)
    dx = FPAdjointSolver.calculate_dx(alpha_bounds, tb, xmin)
    dt = FPAdjointSolver.calculate_dt(alpha_bounds, tb, dx, xmin, factor = 4.)
    
    deterministic_ts, deterministic_control = deterministicControlHarness(tb, Tf, energy_eps, alpha_bounds)

    #Set up solver
    #TODO: The way you pass params and the whole object-oriented approach is silly. Tf changes for each solve and atb don't, so maybe rething the architecture!!!
    S = FPAdjointSolver(dx, dt, Tf, xmin)
    ts = S._ts;
    alphas = interp(ts, deterministic_ts, deterministic_control)
    
    alpha_iterations = [alphas]
    J_iterations = []

    J_prev = Inf;
    
    def incrementAlpha(alpha_prev, direction, step):
        alpha_min, alpha_max = alpha_bounds[0], alpha_bounds[1]
        e = ones_like(alpha_prev); 
        alpha_next = alphas + direction  * step
        alpha_bounded_below = amax(c_[alpha_min*e, alpha_next], axis=1)
        return amin(c_[alpha_max*e, alpha_bounded_below], axis=1)
    
    def exactLineSearch(alpha_prev, direction):
        alphas = None
        def line_objective(step):
            alphas = incrementAlpha(alpha_prev, direction, step)
            xs, ts, fs, ps,J, minus_grad_H =  S.solve(tb, alphas, alpha_bounds[1], energy_eps, visualize=False)
#            print 'inner J = ', J
            return J
        best_step = fminbound(line_objective, .0, 1.0, xtol = 1e-2, maxfun = 16, disp=3)
        alphas = incrementAlpha(alpha_prev, direction, best_step)
        return alphas
        
         
    #THE MONEY LOOP:
    for k in xrange(K_max):
        #the f,p solution:
        xs, ts, fs, ps,J, minus_grad_H =  S.solve(tb, alphas, alpha_bounds[1], energy_eps, visualize=False)
        print k, J
        
        #Convergence check:
        if abs(J - J_prev) < J_tol:
            break
        else:
            if visualize:
                print 'J-J_prev = ',  abs(J - J_prev)
            J_iterations.append(J);
            J_prev = J
        if amax(abs(minus_grad_H)) < gradH_tol:
            break
        
        alphas = exactLineSearch(alphas, minus_grad_H)
        alpha_iterations.append(alphas)
    
    
    if visualize:   
        mpl.rcParams['figure.subplot.left'] = .1
        mpl.rcParams['figure.subplot.right'] = .95
        mpl.rcParams['figure.subplot.bottom'] = .1
        mpl.rcParams['figure.subplot.top'] = .9     
        
        plot_every = int(ceil(k/4));
        controls_fig = figure(); hold(True)
        for iter_idx in xrange(0,k, plot_every):
            plot(ts, alpha_iterations[iter_idx],  linewidth = 3, label=str(iter_idx))
        title('Control Convergence', fontsize = 36)
        ylabel(r'$\alpha(t)$',fontsize= 24); xlabel('$t$', fontsize= 24);    legend(loc='upper left')
        
        J_fig = figure();
        plot(J_iterations,  linewidth = 3, label='J_k'); 
        title('Objective Convergence', fontsize = 36)
        ylabel(r'$J_k$',fontsize= 24); xlabel('$k$', fontsize= 24);    legend(loc='upper right')
       
        if fig_name != None:
            for fig,tag in zip([J_fig, controls_fig],
                               ['_objective.png', '_control.png']):
                fig.canvas.manager.window.showMaximized()
                file_name = os.path.join(FIGS_DIR, fig_name + tag)
                print 'saving to ' , file_name
                fig.savefig(file_name)
            
            
    return alphas, S._ts, J_iterations[-1], k  

########################
class FBKSolution():
    def __init__(self,params, xs, ts, fs, ps,  
                 cs_iterates, J_iterates,
                 energy_eps):
        self._ts  = ts;
        self._xs  = xs;
        self._fs = fs;
        self._ps = ps;
        self._cs_iterates = cs_iterates
        self._J_iterates = J_iterates;
        
        self._mu = params[0]
        self._tau_char = params[1]
        self._beta = params[2]
        
        self._energy_eps = energy_eps
    
    def getControls(self):
        return self._cs_iterates[-1]
                
    def save(self, file_name=None):
#        path_data = {'path' : self}
        if None == file_name:
            file_name = 'FBKSoln_m=%.1f_b=%.1f_Tf=%.1f_eps=%.3f'%(self._mu,
                                                         self._beta,
                                                         self._ts[-1],
                                                         self._energy_eps);
        print 'saving path to ', file_name
        file_name = os.path.join(RESULTS_DIR, file_name + '.fbk')
        import cPickle
        dump_file = open(file_name, 'wb')
        cPickle.dump(self, dump_file, 1) # 1: bin storage
        dump_file.close()
        
    @staticmethod
    def load(file_name=None, mu_beta_Tf=None, energy_eps=.001):
        ''' not both args can be None!!!'''
        if None == file_name:
            mu,beta,Tf = [x for x in mu_beta_Tf]
            file_name = 'FBKSoln_m=%.1f_b=%.1f_Tf=%.1f_eps=%.3f'%(mu,
                                                         beta,
                                                         Tf,
                                                         energy_eps);

        file_name = os.path.join(RESULTS_DIR, file_name + '.fbk') 
        print 'loading ', file_name
        import cPickle
        load_file = open(file_name, 'r')
        soln = cPickle.load(load_file)        
        return soln
########################
    
def FBKDriver(params,
              Tf,
              energy_eps = .001,
              alpha_bounds = (-2., 2.),
              initial_ts_cs = None,
              save_soln = False):
     
    xs, ts, fs, ps,\
     cs_iterates, J_iterates = calculateOptimalControl(params,
                                                        Tf,
                                                        energy_eps,
                                                        alpha_bounds,
                                                        step_size_base = 1.,
                                                        initial_ts_cs=initial_ts_cs,
                                                        visualize=True)

    (FBKSolution(params, xs, ts, fs, ps,
                  cs_iterates, J_iterates,
                   energy_eps)).save()

def solveRegimes(regimeParams, Tf, energy_eps = .001,
                 initial_ts_cs = None):
    from multiprocessing import Process
    procs = [];
    for params in regimeParams:
        print 'm,tc,b =' , params
        #Simulate:
#        procs.append( Process(target=FBKDriver,
#                                 args=(params,Tf,energy_eps),
#                                 kwargs = {'save_soln':True}))
#        procs[-1].start()

        FBKDriver(params, Tf,
                  energy_eps,
                  initial_ts_cs=initial_ts_cs,
                  save_soln=True)
        
#    for proc in procs:
#        proc.join()
    

def visualizeRegimes(regimeParams,
                     regimeTitles, 
                     Tf=  1.5, 
                     energy_eps = .001,
                     fig_name = None):
    label_font_size = 24
    xlabel_font_size = 32
    
#    fig = figure(figsize = (17, 20))
#    subplots_adjust(hspace = .1,wspace = .1,
#                     left=.025, right=.975,
#                     top = .95, bottom = .05)
#        
#    inner_titles = {0:'A',
#                    1:'B',
#                    2:'C',
#                    3:'D'}

    cs_fig = figure(figsize = (17, 14))
    subplots_adjust(hspace = .2,wspace = .2,
                     left=.1, right=.975,
                     top = .95, bottom = .05)
    N_regimes = len(regimeParams)
    c_max = 2.0; c_min = -2.0;
    for pidx, params in enumerate(regimeParams):
        fbkSoln = FBKSolution.load(mu_beta_Tf = params[::2]+[Tf])
        print 'mu,tc,b = %.2f,%.2f,%.2f'%(fbkSoln._mu,
                                          fbkSoln._tau_char,
                                          fbkSoln._beta) 
        ts,xs,cs = fbkSoln._ts, fbkSoln._xs, fbkSoln._cs_iterates[-1]
        axc = cs_fig.add_subplot(N_regimes, 1, 1+ pidx)
        axc.plot(ts, cs, linewidth = 3)
        
        axc.hlines(0, ts[0], ts[-1], linestyles='--')
        
        axc.set_xlim(ts[0], ts[-1])
        axc.set_ylim(c_min ,c_max)
        ticks = [ts[-1] /3. , 2.*ts[-1] /3. ,ts[-1] ]
        axc.set_xticks(ticks)
        ticks = [c_min, c_min/2., .0, c_max/2. ,c_max]
        axc.set_yticks(ticks)
       
#        axc.legend()
#        axc.set_ylim(amin(cs)-.1, amax(cs)+.1);
        if N_regimes -1  == pidx:
            axc.set_xlabel('$t$', fontsize = xlabel_font_size);
        axc.set_ylabel(r'$\alpha(t)$', fontsize = xlabel_font_size);
        axc.set_title(regimeTitles[(fbkSoln._mu,
                                  fbkSoln._beta) ])
        
#        for ticklabels in [axc.xaxis.get_majorticklabels(),
#                           axc.yaxis.get_majorticklabels()]:
#            for label in ticklabels:
#                label.set_fontsize(label_font_size )
#           
#        t = add_inner_title(axc, inner_titles[pidx], loc=3,
#                              size=dict(size=ABCD_LABEL_SIZE))
#        t.patch.set_ec("none"); t.patch.set_alpha(0.5)

        axc.set_yticks((alpha_bounds[0], 0,alpha_bounds[1]))
        axc.set_yticklabels(('$%.1f$'%alpha_bounds[0], '$0$','$%.1f$'%alpha_bounds[1]),
                             fontsize = label_font_size)
        
        axc.set_xticks((.5, 1., 1.5, ))
        axc.set_xticklabels(('$.5$' ,'$1.0$' ,'$1.5$'), fontsize = label_font_size)
        
#        for ax in [axc, axv]:
#            for label in ax.xaxis.get_majorticklabels():
#                label.set_fontsize(label_font_size )
#            for label in ax.yaxis.get_majorticklabels():
#                label.set_fontsize(label_font_size )
#        inner_tag = chr(65+pidx)
#        t = add_inner_title(axv, inner_tag, loc=3,
#                              size=dict(size=ABCD_LABEL_SIZE))
#        t.patch.set_ec("none"); t.patch.set_alpha(0.5)
#        t = add_inner_title(axc,chr(65+pidx+N_regimes), loc=3,
#                              size=dict(size=ABCD_LABEL_SIZE))
#        t.patch.set_ec("none"); t.patch.set_alpha(0.5)
        axc.text(-.15, 1.0, '(%s)'%chr(65+pidx),
                horizontalalignment='center', verticalalignment='center',
                transform=axc.transAxes,
                fontsize = ABCD_LABEL_SIZE)
        
        
    get_current_fig_manager().window.showMaximized()        
    if None!= fig_name:
        lfig_name = os.path.join(FIGS_DIR, fig_name + '_cs.pdf')
        print 'saving to ', lfig_name
        savefig(lfig_name, dpi=300)


def visualizeRegimesSinglePlot(regimeParams, 
                                regimeLabels,
                                 Tf=  1.5, 
                                  energy_eps = .001,
                                   fig_name = None):
        

    cs_fig = figure(figsize = (17, 8))
    subplots_adjust(hspace = .2,wspace = .2,
                     left=.1, right=.975,
                     top = .95, bottom = .15)
    N_regimes = len(regimeParams)
    c_max = 2.0; c_min = -2.0;
    axc = cs_fig.add_subplot(111)
    axc.hold(True)
    for pidx, params in enumerate(regimeParams):
        fbkSoln = FBKSolution.load(mu_beta_Tf = params[::2]+[Tf],
                                   energy_eps = energy_eps)
        print 'mu,tc,b = %.2f,%.2f,%.2f'%(fbkSoln._mu,fbkSoln._tau_char, fbkSoln._beta) 
        ts,xs,cs = fbkSoln._ts, fbkSoln._xs, fbkSoln._cs_iterates[-1]
        
        axc.hlines(0, ts[0], ts[-1], linestyles='--')
        axc.plot(ts, cs, linewidth = 3,
                  label = regimeLabels[(params[0],
                                        params[2])]);
        
        axc.set_xlim(ts[0], ts[-1])
        axc.set_ylim(c_min ,c_max)
        

    axc.set_xlabel('$t$', fontsize = xlabel_font_size);
    axc.set_ylabel(r'$\alpha(t)$', fontsize = xlabel_font_size);
    fontP = FontProperties()
    fontP.set_size(24)
    axc.legend(loc = 'upper left', prop = fontP)
    axc.set_yticks((alpha_bounds[0], 0,alpha_bounds[1]))
    axc.set_yticklabels(('$%.1f$'%alpha_bounds[0], '$0$','$%.1f$'%alpha_bounds[1]),
                         fontsize = label_font_size)
    
    axc.set_xticks((0.75, 1.5, ))
    axc.set_xticklabels(('$t^*/2$' ,'$t^*$'),
                         fontsize = label_font_size)
        
        
    get_current_fig_manager().window.showMaximized()        
    if None!= fig_name:
        lfig_name = os.path.join(FIGS_DIR, fig_name + '_cs_singleplot.pdf')
        print 'saving to ', lfig_name
        savefig(lfig_name, dpi=300)

def visualizeRegimesPresentation(regimeParams, Tf=  1.5, 
                     energy_eps = .001,
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
    plot_titles = {0:'Low-Noise',
                    1:'High-Noise',
                    2:'Low-Noise',
                    3:'High-Noise'}

    cs_fig = figure(figsize = (17, 6))
    subplots_adjust(hspace = .2,wspace = .2,
                     left=.1, right=.975,
                     top = .9, bottom = .15)
    N_regimes = len(regimeParams)
    c_max = 2.0; c_min = -2.0;
    for pidx, params in enumerate(regimeParams):
        fbkSoln = FBKSolution.load(mu_beta_Tf = params[::2]+[Tf])
        print 'mu,tc,b = %.2f,%.2f,%.2f'%(fbkSoln._mu,fbkSoln._tau_char, fbkSoln._beta) 
        ts,xs,cs = fbkSoln._ts, fbkSoln._xs, fbkSoln._cs_iterates[-1]
       
        axc = cs_fig.add_subplot(1, N_regimes, 1+ pidx)
        axc.plot(ts, cs, linewidth = 3)
        
        axc.hlines(0, ts[0], ts[-1], linestyles='--')
        
        axc.set_xlim(ts[0], ts[-1])
        axc.set_ylim(c_min ,c_max)
        ticks = [ts[-1] /3. , 2.*ts[-1] /3. ,ts[-1] ]
        axc.set_xticks(ticks)
        ticks = [c_min, c_min/2., .0, c_max/2. ,c_max]
        axc.set_yticks(ticks)
       
#        axc.legend()
#        axc.set_ylim(amin(cs)-.1, amax(cs)+.1);
        axc.set_xlabel('$t$', fontsize = xlabel_font_size);
        axc.set_ylabel(r'$\alpha(t)$', fontsize = xlabel_font_size);
        
        for ticklabels in [axc.xaxis.get_majorticklabels(),
                           axc.yaxis.get_majorticklabels()]:
            for label in ticklabels:
                label.set_fontsize(label_font_size )
           
        axc.set_title(plot_titles[pidx], fontsize = xlabel_font_size)
#        t = add_inner_title(axc, inner_titles[pidx], loc=3,
#                              size=dict(size=ABCD_LABEL_SIZE))
#        t.patch.set_ec("none"); t.patch.set_alpha(0.5)
    
    get_current_fig_manager().window.showMaximized()        
    if None!= fig_name:
        lfig_name = os.path.join(FIGS_DIR, fig_name + '_cs_presentation.pdf')
        print 'saving to ', lfig_name
        savefig(lfig_name, dpi=300)

def compareEffectOfEnergyEps(regimeParams, Tf, values_of_eps = [.001, .1],
                             alpha_bounds = [-2., 2.],
                             fig_name = None):

    c_min ,c_max = -2., 2
    ####CONTROLS SNAPSHOTS:
    cuts_fig = figure(figsize = (17, 20))
    subplots_adjust(hspace = .2, wspace = .4,
                     left=.15, right=.975,
                     top = .95, bottom = .05)
  
    N_regimes = len(regimeParams)
    N_eps = len(values_of_eps)
    for pidx, params in enumerate(regimeParams):
        for eidx, energy_eps in enumerate(values_of_eps):
            fbkSoln = FBKSolution.load(mu_beta_Tf = params[::2]+[Tf],
                                       energy_eps = energy_eps)
            print 'mu,tc,b = %.2f,%.2f,%.2f'%(fbkSoln._mu,fbkSoln._tau_char, fbkSoln._beta) 
            print 'Tf, energy_eps   = %.3f,%.3f '%(fbkSoln._ts[-1],
                                                   fbkSoln._energy_eps)
            ts,xs,cs = fbkSoln._ts, fbkSoln._xs, fbkSoln._cs_iterates[-1]
            ax = cuts_fig.add_subplot(N_regimes,N_eps,1 + N_eps*pidx + eidx)
            
            if  0 == pidx:
                ax.set_title(r'$ \epsilon = %.3f $'%energy_eps,
                         fontsize = xlabel_font_size)
            ax.set_ylim(alpha_bounds[0]-.1,
                        alpha_bounds[1]+.1);
            ax.plot(ts, cs, linewidth = 3)
            if N_regimes -1  == pidx:
                ax.set_xlabel('$t$', fontsize = xlabel_font_size);
            if 0 == eidx:
                ax.set_ylabel(r'$\alpha(t)$', fontsize = xlabel_font_size);
        
            axc = ax
            axc.hlines(0, ts[0], ts[-1], linestyles='--')
        
            axc.set_xlim(ts[0], ts[-1])
            axc.set_ylim(c_min ,c_max)
            ticks = [ts[-1] /3. , 2.*ts[-1] /3. ,ts[-1] ]
            axc.set_xticks(ticks)
            ticks = [c_min, .0, c_max]
            axc.set_yticks(ticks)
            axc.set_yticklabels(('$%.1f$'%alpha_bounds[0], '$0$','$%.1f$'%alpha_bounds[1]),
                                 fontsize = label_font_size)
            axc.set_xticks((.5, 1.0, 1.5))
            axc.set_xticklabels(('$.5$', '$1$','$1.5$'), fontsize = label_font_size)
            
            inner_tag ='(%s)'%chr(65+pidx*2 + eidx)
            axc.text(-.2, 1.0, inner_tag,
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes,
                fontsize = ABCD_LABEL_SIZE)
          
    #        axc.legend()
    
            
#            for ticklabels in [axc.xaxis.get_majorticklabels(),
#                               axc.yaxis.get_majorticklabels()]:
#                for label in ticklabels:
#                    label.set_fontsize(label_font_size )
                    
#            t = add_inner_title(ax, chr(65+pidx*N_eps + eidx), loc=3,
#                                size=dict(size=ABCD_LABEL_SIZE))
#            t.patch.set_ec("none"); t.patch.set_alpha(0.5)
    
    get_current_fig_manager().window.showMaximized()        
    if None != fig_name:
        lfig_name = os.path.join(FIGS_DIR, fig_name + '_eps_comparison.pdf')
        print 'saving to ', lfig_name
        savefig(lfig_name)
    
def compareEffectOfEnergyEpsJoined(regimeParams, Tf, values_of_eps = [.001, .1],
                             alpha_bounds = [-2., 2.],
                             fig_name = None):
    '''This is the same as compareEffectOfEnergyEps, except that it plots them 
    both on the same graph (ref reply)'''

    c_min ,c_max = -2., 2
    ####CONTROLS SNAPSHOTS:
    cuts_fig = figure(figsize = (17, 20))
    subplots_adjust(hspace = .2, wspace = .4,
                     left=.15, right=.975,
                     top = .95, bottom = .05)
  
    N_regimes = len(regimeParams)
    N_eps = len(values_of_eps)
    for pidx, params in enumerate(regimeParams):
        for eidx, energy_eps in enumerate(values_of_eps):
            fbkSoln = FBKSolution.load(mu_beta_Tf = params[::2]+[Tf],
                                       energy_eps = energy_eps)
            print 'mu,tc,b = %.2f,%.2f,%.2f'%(fbkSoln._mu,fbkSoln._tau_char, fbkSoln._beta) 
            print 'Tf, energy_eps   = %.3f,%.3f '%(fbkSoln._ts[-1],
                                                   fbkSoln._energy_eps)
            ts,xs,cs = fbkSoln._ts, fbkSoln._xs, fbkSoln._cs_iterates[-1]
            ax = cuts_fig.add_subplot(N_regimes, 1,1 + pidx)
            ax.hold(True)
            
            if  0 == pidx:
                ax.set_title(r'Effect of $\epsilon$',
                         fontsize = xlabel_font_size)
            ax.plot(ts, cs, linewidth = 3, label=r'$\epsilon=%.3f$'%energy_eps)
            if N_regimes -1  == pidx:
                ax.set_xlabel('$t$', fontsize = xlabel_font_size);
            ax.set_ylabel(r'$\alpha(t)$', fontsize = xlabel_font_size);
        
            axc = ax
            axc.hlines(0, ts[0], ts[-1], linestyles='--')
        
            axc.set_xlim(ts[0], ts[-1])
            ax.set_ylim(alpha_bounds[0]-.2,
                        alpha_bounds[1]+.2);
#            axc.set_ylim(c_min ,c_max)
            ticks = [ts[-1] /3. , 2.*ts[-1] /3. ,ts[-1] ]
            axc.set_xticks(ticks)
            ticks = [c_min, .0, c_max]
            axc.set_yticks(ticks)
            axc.set_yticklabels(('$%.1f$'%alpha_bounds[0], '$0$','$%.1f$'%alpha_bounds[1]),
                                 fontsize = label_font_size)
            axc.set_xticks((.5, 1.0, 1.5))
            axc.set_xticklabels(('$.5$', '$1$','$1.5$'), fontsize = label_font_size)
            
            inner_tag ='(%s)'%chr(65+pidx)
            axc.text(-.1, 1.0, inner_tag,
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize = ABCD_LABEL_SIZE)
            if 3 == pidx:
                ax.legend(loc='lower right',
                      prop={'size':label_font_size})
                
#            for ticklabels in [axc.xaxis.get_majorticklabels(),
#                               axc.yaxis.get_majorticklabels()]:
#                for label in ticklabels:
#                    label.set_fontsize(label_font_size )
                    
#            t = add_inner_title(ax, chr(65+pidx*N_eps + eidx), loc=3,
#                                size=dict(size=ABCD_LABEL_SIZE))
#            t.patch.set_ec("none"); t.patch.set_alpha(0.5)
    
    get_current_fig_manager().window.showMaximized()        
    if None != fig_name:
        lfig_name = os.path.join(FIGS_DIR, fig_name + '_eps_comparison_joined.pdf')
        print 'saving to ', lfig_name
        savefig(lfig_name)
def crossCompare(regimeParams, regimeTitles,  Tf):
    for opt_params in regimeParams:
        fbkSoln = FBKSolution.load(mu_beta_Tf = opt_params[::2]+[Tf])
        print 'mu,tc,b = %.2f,%.2f,%.2f'%(fbkSoln._mu,
                                          fbkSoln._tau_char,
                                          fbkSoln._beta)
        ts_opt,cs_opt = fbkSoln._ts, fbkSoln._cs_iterates[-1];
        print 'Optimal Control for ', regimeTitles[(fbkSoln._mu, fbkSoln._beta) ], 'J=%.3f'%fbkSoln._J_iterates[-1]
        for params in regimeParams:
            mu,beta = params[::2]
            xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, params, num_std = 1.0)
            dx = FPAdjointSolver.calculate_dx(alpha_bounds, params, xmin)
            dt = FPAdjointSolver.calculate_dt(alpha_bounds, params, dx, xmin, factor = 4.)
            S = FPAdjointSolver(dx, dt, Tf, xmin)
            ts = S._ts;
            
            cs_interped  = interp(ts, ts_opt, cs_opt); 
            alphas = cs_interped
    
            xs, ts, fs, ps, J, minus_grad_H =  S.solve(params,
                                                        alphas,
                                                         alpha_bounds[1],
                                                          energy_eps=fbkSoln._energy_eps,
                                                           visualize=False)
            
        
            print '\t:%s'%regimeTitles[(mu,beta)],'\t:J=%.3f'%J
        

def SuperThreshLowNoiseHarness(low_noise_params,
                                Tf = 1.5, energy_eps = .001):
    
    fbkSoln = FBKSolution.load(mu_beta_Tf = low_noise_params[::2]+[Tf])
    print 'mu,tc,b = %.2f,%.2f,%.2f'%(fbkSoln._mu,
                                      fbkSoln._tau_char,
                                      fbkSoln._beta)
    ts_opt,cs_opt = fbkSoln._ts, fbkSoln._cs_iterates[-1];
    print fbkSoln._J_iterates[-1]
    
    params = low_noise_params;
    xs, ts, fs, ps, cs_iterates, J_iterates = calculateOptimalControl(params, Tf,
                                                    energy_eps,
                                                     alpha_bounds,
                                                      alpha_step = .9,
                                                       visualize=True,
                                                       J_tol = 1e-4,
                                                       initial_ts_cs = (ts_opt,cs_opt))
    print J_iterates[-3:]
                                            #    initial_ts_cs = (ts_opt,cs_opt),

def SuperThreshHighNoiseHarness(high_noise_params,
                                low_noise_params,
                                alpha_bounds = [-2,2],
                                Tf = 1.5, energy_eps = .001):
    params = high_noise_params
    print params
    
#    fbkSoln_ln = FBKSolution.load(mu_beta_Tf = low_noise_params[::2]+[Tf])
#    print 'mu,tc,b = %.2f,%.2f,%.2f'%(fbkSoln_ln._mu,
#                                      fbkSoln_ln._tau_char,
#                                      fbkSoln_ln._beta)
#    ts_ln,cs_ln = fbkSoln_ln._ts, fbkSoln_ln._cs_iterates[-1];
    
    
#    params = high_noise_params;
#    xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, params, num_std = 1.0)
#    dx = FPAdjointSolver.calculate_dx(alpha_bounds, params, xmin)
#    dt = FPAdjointSolver.calculate_dt(alpha_bounds, params, dx, xmin, factor = 4.)
#    S = FPAdjointSolver(dx, dt, Tf, xmin)
#    ts = S._ts;
#    alphas = interp(ts, ts_ln, cs_ln); 
#    xs, ts, fs, ps, J_hn_with_ln_soln, minus_grad_H =  S.solve(params,
#                                            alphas,
#                                                 alpha_bounds[1],
#                                                  energy_eps= energy_eps,
#                                                   visualize=False)
    
    
    fbkSoln_hn = FBKSolution.load(mu_beta_Tf = high_noise_params[::2]+[Tf])
    cs_linint_init, J_linint_init = fbkSoln_hn._cs_iterates[0],\
                                    fbkSoln_hn._J_iterates[0];

 ################################################   
    HN_high_prec_file_name = 'SuperTHighNoise_to_highprec'                              
    #REload high prec solution:
    fbkSoln_init = FBKSolution.load(HN_high_prec_file_name)
    ts_init,cs_init = fbkSoln_init._ts, fbkSoln_init._cs_iterates[-1];
    xs, ts, fs, ps, cs_iterates, J_iterates = calculateOptimalControl(params, Tf,
                                                    energy_eps,
                                                     alpha_bounds,
                                                      alpha_step = 2.,
                                                       visualize=True,
                                                       J_tol = 1e-6,
                                                       K_max = 200,
                                                       dt_factor=4.,
                                                       initial_ts_cs = (ts_init,cs_init))
    (FBKSolution(params, xs, ts, fs, ps,
                  cs_iterates, J_iterates,
                   energy_eps)).save(HN_high_prec_file_name)
#    
    fbkSoln_hn_highprec = FBKSolution.load(HN_high_prec_file_name)
    ts_hn_high_pres, cs_hn_high_pres, J_hn_high_prec= fbkSoln_hn_highprec._ts,\
                                                        fbkSoln_hn_highprec._cs_iterates[-1],\
                                                        fbkSoln_hn_highprec._J_iterates[-1];
    
################################################
#    HN_from_ln_ics_file_name = 'SuperTHighNoise_from_LowNoiseInits'                              
#    ts_init,cs_init = ts_ln,cs_ln;
#    xs, ts, fs, ps, cs_iterates, J_iterates = calculateOptimalControl(params, Tf,
#                                                    energy_eps,
#                                                     alpha_bounds,
#                                                      alpha_step = .9,
#                                                       visualize=True,
#                                                       J_tol = 1e-6,
#                                                       K_max = 200,
#                                                       dt_factor=4.,
#                                                       initial_ts_cs = (ts_init,cs_init))
#    (FBKSolution(params, xs, ts, fs, ps,
#                  cs_iterates, J_iterates,
#                   energy_eps)).save(HN_from_ln_ics_file_name)
#    fbkSoln_hn_from_ln_ics = FBKSolution.load(HN_from_ln_ics_file_name)
#    
#    ts_hn_from_ln_ics, cs_hn_from_ln_ics, J_hn_from_ln_ics= fbkSoln_hn_from_ln_ics._ts,\
#                                                            fbkSoln_hn_from_ln_ics._cs_iterates[-1],\
#                                                            fbkSoln_hn_from_ln_ics._J_iterates[-1];


################################################    
    HN_from_amin_ics_file_name = 'SuperTHighNoise_amin_ICs'                              
#    ts_init,cs_init = ts_hn_high_pres,alpha_bounds[0]*ones_like(ts_hn_high_pres);
#    xs, ts, fs, ps, cs_iterates, J_iterates = calculateOptimalControl(params, Tf,
#                                                    energy_eps,
#                                                     alpha_bounds,
#                                                      alpha_step = 2.,
#                                                       visualize=True,
#                                                      J_tol = 1e-6,
#                                                       K_max = 400,
#                                                       dt_factor=4.,
#                                                       initial_ts_cs = (ts_init,cs_init))
#    (FBKSolution(params, xs, ts, fs, ps,
#                  cs_iterates, J_iterates,
#                   energy_eps)).save(HN_from_amin_ics_file_name)
    fbkSoln_hn_from_amin_ics = FBKSolution.load(HN_from_amin_ics_file_name)
    
    ts_hn_from_amin_ics, cs_amin_init, cs_hn_from_amin_ics,\
     J_amin, J_hn_from_amin_ics = fbkSoln_hn_from_amin_ics._ts,\
                                    fbkSoln_hn_from_amin_ics._cs_iterates[0],\
                                    fbkSoln_hn_from_amin_ics._cs_iterates[-1],\
                                    fbkSoln_hn_from_amin_ics._J_iterates[0],\
                                    fbkSoln_hn_from_amin_ics._J_iterates[-1];
#    print 'J_from_ln = %.4f, J_low_prec = %.4f,\
#            J_high_prec=%.4f, J_hn_from_ln_ics = %.4f' %(J_hn_with_ln_soln,
#                                                        J_hn_low_prec,
#                                                        J_hn_high_prec,
#                                                        J_hn_from_ln_ics)

    #Visualize it:
    ts = ts_hn_from_amin_ics
    figure(figsize = (17, 20)); hold(True)
    subplot(211)
    sub_stride = 2;
    plot(ts, cs_amin_init, 'b-.',
                 label='alpha min', linewidth = 3)
    plot(ts, cs_hn_from_amin_ics, 'b-',
                     label='alpha_opt starting from amin', linewidth = 3)
    plot(ts, cs_linint_init, 'r-.',
         label='alpha_0 linear interp amin->amax', linewidth = 3)
    plot(ts, cs_hn_high_pres,'r-', 
         label='alpha_opt starting from alin interp', linewidth = 3)
    xlabel('$t$')
    xlim((.0, Tf))
    legend(loc='upper left')
    
    
    ax = subplot(212)
    Js = [J_linint_init,
          J_hn_high_prec,
          J_hn_from_amin_ics,
          J_amin]
    ax.bar( arange(4), Js, 0.3)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(arange(4))
    ax.set_xlim([-0.5, 3.5])
    ax.set_ylim([0.85, .89])
    ax.set_xticklabels(['alpha linint',
                        'alpha_opt starting \n from alpha linint',
                        'alpha_opt starting \n from  alpha min',
                        'alpha_min'])
    for x, J in enumerate(Js):
        ax.text(x, J+.0001, r'$J = %.4f$'%J)
    ax.set_title('Objective values for different control ($J$)')
    
    file_name = os.path.join(FIGS_DIR, 'is_it_local_min.pdf')
    print 'saving to ', file_name
    savefig(file_name)
    

def algosHarness(params,
               Tf = 1.5, energy_eps = .001):
    
#    ncg_file_name = 'NCG_Solution'
#    xs, ts, fs, ps,\
#       cs_iterates, J_iterates = ncg4OptimalControl_NocedalWright(params, Tf),
##       cs_iterates, J_iterates = ncg4OptimalControl(params, Tf)
##    TODO: Create a FBK NCG Solution file??? 
#    (FBKSolution(params, xs, ts, fs, ps,
#                  cs_iterates, J_iterates,
#                   energy_eps)).save(ncg_file_name)
#    
#    ncgSoln = FBKSolution.load(ncg_file_name)
    
    gd_file_name = 'GD_Solution_%.2f_%.2f'%(params[0], 
                                            params[2])
#    gdSoln = FBKSolution.load(gd_file_name)
#    initial_ts_cs = (gdSoln._ts, gdSoln._cs_iterates[-1])
    
    xs, ts, fs, ps,\
       cs_iterates, J_iterates = gdOptimalControl_Aggressive(params, Tf)
    (FBKSolution(params, xs, ts, fs, ps,
                  cs_iterates, J_iterates,
                   energy_eps)).save(gd_file_name)
    
    fbkSoln_new = FBKSolution.load(gd_file_name)
    
    fbkSoln_old = FBKSolution.load(mu_beta_Tf = params[::2]+[Tf])
        
    figure(figsize= (16,16))
    for idx, (fbkSoln, soln_label) in enumerate(zip([fbkSoln_old,fbkSoln_new],
                                                    ['Old Soln', 'New soln'])):
        #Left vs. Right panels:
        ts,cs ,J_iterates = fbkSoln._ts, fbkSoln._cs_iterates[-1], fbkSoln._J_iterates;
        cs_init = fbkSoln._cs_iterates[0]
        subplot(2,1, 1); hold(True)
        plot(ts, cs, label=soln_label)
        xlim((.0, Tf))
        title('mu, beta = %.2f_%.2f'%(params[0], 
                                      params[2]))
        legend(loc='upper left')

        ax = subplot(2,1,2); hold(True)
        plot(J_iterates, label=soln_label + ':J=%.5f'%J_iterates[-1])
#        ax.text(len(J_iterates)/3.,J_iterates[0], 'num iterations = %d' %len(J_iterates))
        ax.legend(loc = 'upper right')
    
    figure()
    title('Control Iterates:: mu, beta = %.2f_%.2f'%(params[0], 
                                                     params[2]))
    ts = fbkSoln_new._ts
    for idx, cs in enumerate(fbkSoln_new._cs_iterates):
        plot(ts,cs, label='%d'%(idx+1)); hold(True)
    legend(loc='lower right')
    
    
#    file_name = os.path.join(FIGS_DIR, 'is_it_local_min.pdf')
#    print 'saving to ', file_name
#    savefig(file_name)

def PyVsCDriver(regimeParams):
    pass

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
    regimeParams = [[mu_high/tau_char, tau_char, beta_low],
                    [mu_high/tau_char, tau_char, beta_high],
                     [mu_low/tau_char, tau_char, beta_low],
                     [mu_low/tau_char, tau_char, beta_high]]
#    regimeTitles = {(mu_high/tau_char, beta_low) :'SuperThresh, Low-Noise',
#                     (mu_high/tau_char, beta_high):'SuperThresh,High-Noise', 
#                     (mu_low/tau_char, beta_low)  :'SubThresh, Low-Noise', 
#                     (mu_low/tau_char, beta_high) :'SubThresh, High-Noise'}
    regimeTitles = {(mu_high/tau_char, beta_low) :'SupraT, low-noise',
                      (mu_high/tau_char, beta_high):'SupraT, high-noise', 
                     (mu_low/tau_char, beta_low)  :'SubT, low-noise', 
                     (mu_low/tau_char, beta_high) :'SubT, high-noise'}
        
#    PyVsCDriver(regimeParams)
#    solveRegimes(regimeParams[0:],
#                  Tf, energy_eps = .001)

#    visualizeRegimes(regimeParams[:],
#                     regimeTitles,
#                     Tf= 1.5,
#                     fig_name = 'Regimes')
    
#    crossCompare(regimeParams[0:2],
#                 regimeTitles, 
#                 Tf= 1.5 )
#    for params in regimeParams:
#        algosHarness(params);
#    algosHarness(regimeParams[1]);
    
#    SuperThreshHighNoiseHarness(regimeParams[1],
#                                regimeParams[0]);
#    SuperThreshLowNoiseHarness(regimeParams[0])
#
#    visualizeRegimesSinglePlot(regimeParams,
#                               regimeTitles,
#                               Tf= 1.5,
#                               energy_eps=.001,
#                               fig_name = 'Regimes')
    visualizeRegimesSinglePlot(regimeParams,
                               regimeTitles,
                               Tf= 1.5,
                               energy_eps=.1,
                               fig_name = 'RegimesHighEps')
    
#    visualizeRegimes(regimeParams[2:],
#                      Tf= 1.5,
#                       fig_name = 'SubT_Regimes')
#       
#    visualizeRegimesPresentation(regimeParams[2:],
#                      Tf= 1.5,
#                       fig_name = 'SubT_Regimes')
#    initial_ts_cs_high_eps = [[.0, Tf/2., Tf],
#                              zeros_like([.0, Tf/2., Tf])]
#    solveRegimes(regimeParams[:],
#                  Tf,
#                   energy_eps = .1,
#                   initial_ts_cs=initial_ts_cs_high_eps)
#    compareEffectOfEnergyEps(regimeParams, 
#                             Tf,
#                             values_of_eps = [.001, .1],
#                             fig_name = 'Regimes')
#    compareEffectOfEnergyEpsJoined(regimeParams, 
#                                   Tf,
#                                   values_of_eps = [.001, .1],
#                                   fig_name = 'Regimes')


###########################


#    visualizeAdjointSolver(tb, Tf, energy_eps, alpha_bounds)
#    compareControlTerm(tb, Tf, energy_eps, alpha_bounds)
#    calcGradH(tb, Tf, energy_eps, alpha_bounds, fig_name = 'First_control_iteration')
#    calculateOutflow(tb, Tf, energy_eps, alpha_bounds)
    
    #calculateOptimalControl(tb, Tf, 
#                            energy_eps, alpha_bounds, visualize=True, fig_name = 'ExampleControlConvergence')
    
#    start = time.clock()
#    alphas, ts, J, k = exactStepOptimalControl(tb, Tf, energy_eps, alpha_bounds,  visualize=True)
#    print 'Calc time = ', time.clock( )- start
    
#    start = time.clock()
#    alphas, ts, J, k = calculateOptimalControl(tb, Tf, energy_eps, alpha_bounds, alpha_step = .75, visualize=True)
#    print 'Calc time = ', time.clock( )- start

#    alphas_large_step, ts, k = calculateOptimalControl(tb, Tf, energy_eps, alpha_bounds, alpha_step = 1.)
#    
#    figure()
#    plot(ts, alphas, ts, alphas_large_step); 
    
#    timeAdjointSolver(tb, Tf, energy_eps, alpha_bounds)
 
#    stylizedVisualizeForwardAdjoint(  fig_name='Stylized_')


    show()
    