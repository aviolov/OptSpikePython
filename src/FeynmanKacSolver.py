# -*- coding:utf-8 -*-
"""
@author: alex
"""
from __future__ import division

from numpy import linspace, float, arange, sum, exp, NaN
from numpy import sin, sqrt, ones_like, zeros_like, where, diff, pi, log, max , sign, amin, amax
from numpy import zeros, ones, array, c_, r_, float64, matrix, bmat, Inf, ceil, arange, empty, interp, dot, sqrt
from scipy.sparse import spdiags, lil_matrix
from scipy.sparse.linalg.dsolve.linsolve import spsolve
from copy import deepcopy
from ControlSimulator import deterministicControlHarness
from scipy.optimize.optimize import fminbound
from scipy import interpolate

RESULTS_DIR = '/home/alex/Workspaces/Python/OptSpike/Results/FeynmanKac/'
FIGS_DIR = '/home/alex/Workspaces/Latex/OptSpike/Figs/FeynmanKac'

import os
for D in [FIGS_DIR, RESULTS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)
import time

from AdjointSolver import FPAdjointSolver

#import ext_fpc

class FeynmanKacSolver():    
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
    def calculate_xmin(cls, alpha_bounds, tb, num_std = 2.0):     
        XMIN_AT_LEAST = -.5;   
        tc, beta = tb[0], tb[1]
        alpha_min = alpha_bounds[0]
        xmin = tc*alpha_min - num_std*beta*sqrt(tc/2.0);
        return min([XMIN_AT_LEAST, xmin])
    @classmethod
    def calculate_dx(cls, alpha_bounds, tb, xmin, factor = 1e-1, xthresh = 1.0):
        #PEclet number based calculation:
        tauchar = tb[0]; beta = tb[1]        
        alpha_min  = alpha_bounds[0]; alpha_max = alpha_bounds[1];
        
        max_speed = max([ alpha_max - xmin / tauchar,
                          -alpha_min + xthresh / tauchar ]);
        return factor * (beta / max_speed);
    @classmethod
    def calculate_dt(cls, alpha_bounds, tb, dx, xmin, factor=2., xthresh = 1.0):
        tauchar = tb[0]; beta = tb[1]        
        alpha_min  = alpha_bounds[0]; alpha_max = alpha_bounds[1];
        max_speed = max([ alpha_max - xmin / tauchar,
                          -alpha_min + xthresh / tauchar ]);
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
    def solve(self, tb, alphas, alpha_max, energy_eps, visualize=False, save_fig=False):
        #Indirection method
        
        fs = self._fsolve( tb, alphas, visualize, save_fig)
        ps = self._psolve( tb, alphas, alpha_max, visualize, save_fig)

        xs, ts = self._xs, self._ts;
        
        J = self.calcCost(energy_eps, alphas, tb[1], fs,ps)
        
        #the Hamiltonian gradient:
        dxps = diff(ps, axis=0)/self._dx;
        dxpf = sum(fs[1:,:]*dxps, axis=0)
    
        minus_grad_H = -(2*energy_eps*alphas - ps[-1,:]*fs[-1,:] + 2*ps[0,:]*fs[0,:] + dxpf)   
   
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
        
        
    
    def _psolve(self, tb, alphas, alpha_max, visualize=False, save_fig=False):
        tauchar, beta = tb[0], tb[1]
        
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
        ps[:,-1] = self._getTCs(xs, alpha_max, tauchar, beta)
        
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
        #Bottome BCs:
        M[0,0] = -1.;
        
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
            U_forward = (alpha_forward - xs[1:-1]/ tauchar)
            U_current = (alpha_current - xs[1:-1]/ tauchar)
            
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
            #Bottome BCs:
            M[0,1] = 1.0;
            
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
    
    
    def _fsolve(self, tb, alphas, visualize=False, save_fig=False):
        tauchar, beta = tb[0], tb[1]
        
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
            U_prev = (alpha_prev - xs/ tauchar)
            U_next = (alpha_next - xs/ tauchar)
            
            
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
            M[0,1] = - D / dx;
            
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


def simulateXs(alphas, ts, tb, Tf, num_samples = 100, dt = 1e-3):
    sqrt_dt = sqrt(dt)
    tauchar, beta = tb[0],tb[1];
    num_steps = len(ts)
    num_sub_steps = int( floor((ts[1] - ts[0]) / dt) );
    x_thresh = 1.0;
    
    Xs = x_thresh*ones( (num_steps, num_samples) )
    
    BrownianIncrements = randn(num_sub_steps*num_steps, num_samples)*sqrt_dt
    
    Xs[0,:] = .0;
    active_samples = range(num_samples);
    
    for idx in xrange(num_steps-1):
        alpha = alphas[idx];
        
        #for each trajectory increment one time-step:
        spiked_samples = []
        for sample_idx in active_samples:
            X = Xs[idx, sample_idx];
            #EVOLVE trajectory, until spike:
            for inner_idx in xrange(num_sub_steps):
                
                dB = BrownianIncrements[idx*num_sub_steps + inner_idx, sample_idx]
                
                dX = (alpha - X/tauchar)*dt +  beta*dB

                X += dX;            
                
                if X >= x_thresh:
                    spiked_samples.append(sample_idx)
                    break
                
            Xs[idx+1, sample_idx]  =  X
            
        for spiked_idx in spiked_samples:
            active_samples.remove(spiked_idx)
            
    return Xs;
#
#def simulateXs(alphas, ts, tb, Tf, num_samples = 100, dt = 1e-3):
#    tauchar, beta = tb[0],tb[1];
#    sqrt_dt = sqrt(dt)
#    lts = arange(ts[0], ts[-1]+dt, dt); 
#    
#    print tauchar
#    alphas = interp(lts, ts, alphas);
#    
#    x_thresh = 1.0;
#    Xs = empty(( len(ts), num_samples) );
#    
#    BrownianIncrements = randn(len(lts), num_samples)*sqrt_dt
#    
#    lXs = x_thresh*ones_like(lts);
#    lXs[0]  = .0
#    
#    for sample_idx in xrange(num_samples):
#        for idx in xrange(1,len(lts)):
#            alpha = alphas[idx]
#            
#            dB = BrownianIncrements[idx, sample_idx]
#            
#            dX = (alpha - lXs[idx-1]/tauchar)*dt +  beta*dB
#
#            lXs[idx] = lXs[idx-1] + dX;            
#            
#            if lXs[idx] >= x_thresh:
#                break
#            
#        Xs[:, sample_idx] = interp(ts, lts, lXs)
#            
#    return Xs;
 

#def simulateXs(alphas, ts, tb, Tf, num_samples = 100, dt = 1e-3):
#    tauchar, beta = tb[0],tb[1];
#    dt = ts[1]-ts[0]
#    sqrt_dt = sqrt(dt)
#    
#    print tauchar
#    
#    x_thresh = 1.0;
#    Xs = x_thresh * ones(( len(ts), num_samples) );
#    
#    BrownianIncrements = randn(len(ts), num_samples)*sqrt_dt
#    
#    
#    Xs[0,:]  = .0
#    
#    for sample_idx in xrange(num_samples):
#        for idx in xrange(1,len(ts)):
#            alpha = alphas[idx]
#            
#            dB = BrownianIncrements[idx, sample_idx]
#            
#            dX = (alpha - Xs[idx-1, sample_idx] / tauchar)*dt +  beta*dB
#
#            Xs[idx, sample_idx] = Xs[idx-1, sample_idx] + dX;            
#            
#            if Xs[idx, sample_idx] >= x_thresh:
#                break
#            
#    return Xs;



def visualizeForwardSample(tb = [.6, 1.25], Tf = 1.5, energy_eps = .001, alpha_bounds = (-2., 2.),
                           fig_name = None):
    mpl.rcParams['figure.subplot.left'] = .1
    mpl.rcParams['figure.subplot.right'] = .95
    mpl.rcParams['figure.subplot.bottom'] = .1
    mpl.rcParams['figure.subplot.top'] = .9
        
    xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, tb, num_std = 1.0)
    dx = FPAdjointSolver.calculate_dx(alpha_bounds, tb, xmin)
    dt = FPAdjointSolver.calculate_dt(alpha_bounds, tb, dx, xmin, factor = 4.)
    
    S = FPAdjointSolver(dx, dt, Tf, xmin)
    
    ts = S._ts;
    
    alphas = interp(ts, ts, 1.0*ones_like(ts))
    
    #the v solution:
    xs, ts, fs, ps, J, minus_grad_H =  S.solve(tb, alphas, alpha_bounds[1], energy_eps,visualize=False)
        
    for num_samples in [128]:
#    num_samples = 16
        Xs = simulateXs( alphas, ts, tb, Tf, num_samples )
        figure()
        plot(ts, Xs)
        
        fig = figure();
        
        slices = [10, int(len(ts)/2), int(len(ts)/1.5), int(len(ts)/1.25),len(ts)-2]
        for idx, tidx in enumerate(slices):
            lXs = Xs[tidx,:]
            subplot(len(slices),2,1 + 2*idx); hold(True);
            plot(xs, fs[:,tidx], linewidth=3);
            hist(lXs, bins = 20, normed= True);
            
            print 'analytical:', sum(fs[:,tidx]* S._dx), ' ; sampled: ', len(lXs[lXs<1.0])/num_samples
            #plot only active samples:
            subplot(len(slices),2,2 + 2*idx); hold(True);
            plot(xs, fs[:,tidx] / sum(fs[:,tidx]* S._dx), linewidth=3);
            if len(lXs[lXs<1.0]) > 0:
                   hist(lXs[lXs<1.0], bins = 20, normed= True) 
            title('t = %.2f, N=%d'%(ts[tidx], num_samples))
    
        fig.canvas.manager.window.showMaximized()

def sampleObjective(tb = [.6, 1.25], Tf = 1.5, energy_eps = .001, alpha_bounds = (-2., 2.),
                           fig_name = None):
    mpl.rcParams['figure.subplot.left'] = .1
    mpl.rcParams['figure.subplot.right'] = .95
    mpl.rcParams['figure.subplot.bottom'] = .1
    mpl.rcParams['figure.subplot.top'] = .9
        
    xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, tb, num_std = 1.0)
    dx = FPAdjointSolver.calculate_dx(alpha_bounds, tb, xmin)
    dt = FPAdjointSolver.calculate_dt(alpha_bounds, tb, dx, xmin, factor = 4.)
    
    S = FPAdjointSolver(dx, dt, Tf, xmin)
    
    ts = S._ts;
    
    alphas = interp(ts, ts, 1.0*ones_like(ts))
    
    #the v solution:
    xs, ts, fs, ps, J, minus_grad_H =  S.solve(tb, alphas, alpha_bounds[1], energy_eps,visualize=False)
        
    Ttwo = ps[:, -1];
    
    sample_counts = [8, 16,32, 64, 128, 512, 1024, 2048]
    Jsampled = zeros(len(sample_counts))
    for jidx, num_samples in enumerate(sample_counts):
        Xs = simulateXs( alphas, ts, tb, Tf, num_samples )

        for sample_idx in xrange(num_samples):
            lXs= Xs[:, sample_idx];
            
            spike_ids= find(lXs>=1.0)
            if 0 == len(spike_ids):
                Jsampled[jidx] += interp(lXs[-1], xs, Ttwo) / num_samples;
            else:
                spike_idx = spike_ids[0]
                Jsampled[jidx] += (ts[spike_idx] - Tf)**2 / num_samples;
        
    plot(sample_counts, Jsampled)
    hlines(J,sample_counts[0], sample_counts[-1])
    

def visualizeForwardSampleAdjointSolve(tb = [.6, 1.25], Tf = 1.5, energy_eps = .001, alpha_bounds = (-2., 2.),
                           fig_name = None):
    mpl.rcParams['figure.subplot.left'] = .1
    mpl.rcParams['figure.subplot.right'] = .95
    mpl.rcParams['figure.subplot.bottom'] = .1
    mpl.rcParams['figure.subplot.top'] = .9
        
    xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, tb, num_std = 1.0)
    dx = FPAdjointSolver.calculate_dx(alpha_bounds, tb, xmin)
    dt = FPAdjointSolver.calculate_dt(alpha_bounds, tb, dx, xmin, factor = 4.)
    
    S = FPAdjointSolver(dx, dt, Tf, xmin)
    
    ts = S._ts;
    
    alphas = interp(ts, ts, 1.0*ones_like(ts))
    
    #the v solution:
    start = time.clock()
    xs, ts, fs, ps, J, minus_grad_H =  S.solve(tb, alphas, alpha_bounds[1], energy_eps,visualize=False)
    end = time.clock()
    print 'solve time = ', end-start
                
    dxps = diff(ps, axis=0)/S._dx;
    soln_fig = figure(); hold(True)
    plot(ts, minus_grad_H,  linewidth = 4,    label='Forward Solve')
    
    for num_samples in [32, 64, 128]:
        start = time.clock()
        Xs = simulateXs( alphas, ts, tb, Tf, num_samples )
        end = time.clock()
        print 'sim time = ', end-start
        
        sampledMinusGradH = zeros_like(minus_grad_H)
        
        for idx in xrange(len(ts)):
            pspline = interpolate.splrep(xs, ps[:, idx], s=0)
            lXs = Xs[idx,:]
            if 0 != len(lXs[lXs<1.0]):
                dxp_sampled = interpolate.splev( lXs[lXs<1.0],pspline ,der=1 )
                    
                sampledMinusGradH[idx] = -sum(dxp_sampled) / num_samples;
        
        plot(ts, sampledMinusGradH, label='Sampled %d'%num_samples)
        
#        averaged = sampledMinusGradH
#        averaged[2:-1] = (sampledMinusGradH[0:-3]+  sampledMinusGradH[1:-2]+  sampledMinusGradH[2:-1]+  sampledMinusGradH[3:])/4.0  
#        smoothInterpolant = interpolate.splrep(ts, sampledMinusGradH)
#        plot(ts, interpolate.splev(ts,smoothInterpolant, der= 0), label='averaged')
        
        legend(loc = 'lower right')
        ylim((2*amin(minus_grad_H), 2*amax(minus_grad_H) ))
        
        
    if None != fig_name:
        file_name = os.path.join(FIGS_DIR, fig_name + '_t=%.0f_b=%.0f.png'%(10*tb[0], 10*tb[1]))
        print 'saving to ', file_name
        soln_fig.savefig(file_name)
    
    
if __name__ == '__main__':
    from pylab import *
    
    tb = [.75, 1.25]; Tf = 1.5; energy_eps = .001; alpha_bounds = (-2., 2.);
#    visualizeForwardSample(tb, Tf, energy_eps, alpha_bounds)
#    sampleObjective(tb, Tf, energy_eps, alpha_bounds)
    visualizeForwardSampleAdjointSolve(tb, Tf, energy_eps, alpha_bounds, fig_name = 'fsampled_psolved')
    
 
    show()
    