# -*- coding:utf-8 -*-
"""
@author: alex
"""
from __future__ import division

from numpy import linspace, float, arange, sum
from numpy import sin, sqrt, ones_like, zeros_like, where, diff, pi, log, max , sign, amin, amax
from numpy import zeros, ones, array, c_, r_, float64, matrix, bmat, Inf, ceil, arange, empty, interp, dot, sqrt
from scipy.sparse import spdiags, lil_matrix
from scipy.sparse.linalg.dsolve.linsolve import spsolve
from copy import deepcopy
from PathSimulator import ABCD_LABEL_SIZE
from matplotlib.font_manager import FontProperties

RESULTS_DIR = '/home/alex/Workspaces/Python/OptSpike/Results/HJB/'
FIGS_DIR    = '/home/alex/Workspaces/Latex/OptSpike/Figs/HJB'

import os
for D in [FIGS_DIR, RESULTS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)
import time

#import ext_fpc

label_font_size = 32
xlabel_font_size = 40

import ext_fb_hjb

class HJBSolver():
    ALPHA_INDEX = 0;
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
        mu, tc, beta = params[0], params[1], params[2]
        alpha_min = alpha_bounds[0]
        xmin = tc*(mu + alpha_min) - num_std*beta*sqrt(tc/2.0);
        return min([-.5, xmin])
    @classmethod
    def calculate_dx(cls, alpha_bounds, params, xmin,
                     factor = 1e-1, xthresh = 1.0):
        #PEclet number based calculation:
        mu, tc, beta = params[0], params[1], params[2]
        max_speed = abs(mu) + max(alpha_bounds) + max([abs(xmin), xthresh]) / tc;
        return factor * (beta / max_speed);
    @classmethod
    def calculate_dt(cls, alpha_bounds, params, dx, xmin,
                      factor=2., xthresh = 1.0):
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
         
    
    def _getBCs(self, ts, T):
        return (ts-T)*(ts-T)

    ###########################################
    def solve(self, params,
               alpha_bounds = (-1, 1.),
                energy_eps = 1.0,
                 visualize=False, save_fig=False):
        mu, tauchar, beta = params[0], params[1], params[2]
        alpha_min, alpha_max = alpha_bounds[0],alpha_bounds[1]
        
        dx, dt = self._dx, self._dt;
        xs, ts = self._xs, self._ts;
        
        if visualize:
            print 'tauchar = %.2f,  beta = %.2f,' %(tauchar, beta)
            print 'amin = %.2f, amax = %.2f, energy_eps=%.3f,'%(alpha_min, alpha_max,energy_eps)
            print 'Tf = %.2f' %self.getTf()
            print 'xmin = %.1f, dx = %f, dt = %f' %(self.getXmin(), dx,dt)
#
        #Allocate memory for solution:
        vs = empty((self._num_nodes(),
                    self._num_steps() ));
        cs = empty((self._num_nodes()-1,
                    self._num_steps()))            

        #Impose TCs: 
        vs[:,-1] = self._getTCs(xs, alpha_max+mu, tauchar, beta)
        
        #Impose Dirichlet BCs: 
        vs[-1,:] = self._getBCs(ts, self.getTf() )
        
        if visualize:
            figure()
            subplot(211)
            plot(xs, vs[:,-1]); 
            title(r'$\alpha=%.2f, \tau=%.2f, \beta=%.2f$'%(alpha_max,tauchar, beta) + ':TCs', fontsize = 24); xlabel('x'); ylabel('v')
             
            subplot(212)
            plot(ts, vs[-1, :]);
            title('BCs at xth', fontsize = 24) ; xlabel('t'); ylabel('v')

        
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
        
        controls_fig, soln_fig = None,None;  
        if visualize:
            soln_fig = figure()
            controls_fig = figure()
            
        def calc_control(di_x_v):
            alpha_reg = -di_x_v / (2.*energy_eps)
            e = ones_like(alpha_reg)
            alpha_bounded_below = amax(c_[alpha_min*e, alpha_reg], axis=1)
            
            return amin(c_[alpha_max*e, alpha_bounded_below], axis=1)
        
            
        for tk in xrange(self._num_steps()-2,-1, -1):
            #Rip the forward-in-time solution:
            v_forward = vs[:,tk+1];
            di_x_v_forward = (v_forward[2:] - v_forward[:-2]) / (2*dx)
            di2_x_v_forward = (v_forward[2:] - 2*v_forward[1:-1] + v_forward[:-2]) / (dx_sqrd)
            
            #Calculate the control:
            alpha = calc_control(di_x_v_forward)
            
            #Form the velocity field:
            U = (mu + alpha - xs[1:-1]/tauchar) 
                             
            #Form the right hand side:
            L_prev =  D * di2_x_v_forward + \
                      U * di_x_v_forward + \
                      2*energy_eps * alpha*alpha

            #impose the x_min BCs: homogeneous Neumann: and assemble the RHS: 
            RHS = r_[(.0,
                      v_forward[1:-1] + .5 * dt * L_prev)];
            
            #Reset the Mass Matrix:
            #Lower Diagonal
            u =  U / (2*dx);
            d_off = D / dx_sqrd;
                    
            L_left = -.5*dt*(d_off - u);
            M.setdiag(L_left, -1);
            
            #Upper Diagonal
            L_right = -.5*dt*(d_off + u);
            M.setdiag(r_[(1.0, L_right)], 1);
            
            #add the terms coming from the upper BC at the backward step to the end of the RHS
            v_upper_boundary = vs[-1,tk]
            RHS[-1] += .5* dt*(D * v_upper_boundary / dx_sqrd + U[-1] *v_upper_boundary / (2*dx) )
            
            #Convert mass matrix to CSR format:
            Mx = M.tocsr();            
            #and solve:
            v_backward = spsolve(Mx, RHS);
            
            
            #Store solutions:
            vs[:-1, tk] = v_backward;
#            cs[:, tk+1] = r_[0.0, alpha]
            cs[:, tk+1] = r_[ alpha[0], alpha]
                          
            if visualize:
                mod_steps = 8;
                num_cols = 2;
                num_rows = ceil(double(self._num_steps())/num_cols / mod_steps) + 1
                
                step_idx = self._num_steps() - 2 - tk;
                
                if 0 == mod(step_idx,mod_steps) or 0 == tk:
                    plt_idx = 1 + floor(tk / mod_steps) + int(0 < tk)
                    ax = soln_fig.add_subplot(num_rows, num_cols, plt_idx)
                    ax.plot(xs, vs[:,tk], label='k=%d'%tk); 
                    if self._num_steps() - 2 == tk:
                        ax.hold(True)
                        ax.plot(xs, vs[:,tk+1], 'r', label='TCs')
                    ax.legend()
#                        ax.set_title('k = %d'%tk); 
                    if (0 != tk):
                        ticks = ax.get_xticklabels()
                        for t in ticks:
                            t.set_visible(False)
                    else:
                        ax.set_xlabel('$x$'); ax.set_ylabel('$v$')
                        for t in ax.get_xticklabels():
                            t.set_visible(True)
                    
                    ax = controls_fig.add_subplot(num_rows, num_cols, plt_idx)
                    ax.plot(xs[1:-1], alpha, label='k = %d'%tk)
                    ax.set_ylim(alpha_min-.1, alpha_max+.1)
                    ax.legend(); #ax.set_title('k = %d'%tk); 
                    if (0 != tk):
                        for t in ax.get_xticklabels():
                            t.set_visible(False)
                    else:
                        ax.set_xlabel('$x$'); ax.set_ylabel(r'$\alpha$')
                        for t in ax.get_xticklabels():
                            t.set_visible(True)
                
        #//end time loop
        #Now store the last control:
        v_init = vs[:,0];
        cs[1:, 0] = calc_control( (v_init[2:] - v_init[:-2]) / (2*dx) );
        cs[0,0] = cs[1,0]
                        
        #Return:
        if visualize:
            for fig in [soln_fig, controls_fig]:
                fig.canvas.manager.window.showMaximized()

            if save_fig:
                file_name = os.path.join(FIGS_DIR, 'soln_al=%.0f_au=%.0f__t=%.0f_b=%.0f.png'%(10*alpha_min, 10*alpha_max,10*tauchar, 10*beta))
                print 'saving to ', file_name
                soln_fig.savefig(file_name)
                
                file_name = os.path.join(FIGS_DIR, 'control_al=%.0f_au=%.0f__t=%.0f_b=%.0f.png'%(10*alpha_min, 10*alpha_max,10*tauchar, 10*beta))
                print 'saving to ', file_name
                controls_fig.savefig(file_name)
            
        return  xs, ts, vs, cs

    def c_solve(self, params,
                alpha_bounds,
                energy_eps = 1.0):
        
        mu, tauchar, beta = params[0], params[1], params[2]
#        alpha_bounds= (-2,2)
        alpha_min, alpha_max = alpha_bounds[0], alpha_bounds[1]
        
        dx, dt = self._dx, self._dt;
        xs, ts = self._xs, self._ts;
        
        TCs = self._getTCs(xs, alpha_max+mu, tauchar, beta)
        
        #WARNING: for some reason giving the raw xs, without calling array on it, caused all kinds of trouble on the C end... (The C driver reads garbage essentiall...)
        xs = array(xs); 
        
        alpha_bounds = array([alpha_min, alpha_max], dtype=float);
        params = array(params, dtype=float)
        vs_cs = ext_fb_hjb.solveHJB(params, 
                                 alpha_bounds,
                                 energy_eps,
                                 TCs,
                                 xs,
                                 ts)
        
        vs= vs_cs[0,:,:];
        cs = vs_cs[1,:-1,:];
        return  xs, ts, vs, cs
        
        


def add_inner_title(ax, title, loc, size=None, **kwargs):
            from matplotlib.offsetbox import AnchoredText
            from matplotlib.patheffects import withStroke
            if size is None:
                size = dict(size=plt.rcParams['legend.fontsize'])
            at = AnchoredText(title, loc=loc, prop=size,
                              pad=0., borderpad=0.5,
                              frameon=False, **kwargs)
            ax.add_artist(at)
            at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
            return at


def visualizeControlSurfaces(regimeParams,
                     Tf=1.5, energy_eps = .001, 
                     fig_name = None):
    line_width = 3
    
    plot_titles = {0:'Low-Noise',
                   1:'High-Noise',
                   2:'Low-Noise',
                   3:'High-Noise'}
    
    fig = figure(figsize = (17, 6))
    fig.hold(True)
    subplots_adjust(hspace = .1,wspace = .1,
                     left=.025, right=.975,
                     top = .95, bottom = .05)
  
    #Visualize:
    from mpl_toolkits.mplot3d.axes3d import Axes3D,proj3d
    ####NOW PLOT CONTROL SURFACE:
    for pidx, params in enumerate(regimeParams):
        hjbSoln = HJBSolution.load(mu_beta_Tf = params[::2]+[Tf], energy_eps = energy_eps)
        print 'mu,tc,b = %.2f,%.2f,%.2f'%(hjbSoln._mu,hjbSoln._tau_char, hjbSoln._beta) 
        ts,xs,vs, cs = hjbSoln._ts, hjbSoln._xs, hjbSoln._vs, hjbSoln._cs
        
        ax = fig.add_subplot(ceil(len(regimeParams)/2), 2, 1+pidx, projection='3d')
        el = 30; az = -25;
        ax.view_init(elev = el, azim= az)
        x_cut = 10;
        X, Y = np.meshgrid(ts, xs[x_cut:])
        r_stride = int(floor(.1 / (ts[1]-ts[0])))
        c_stride = int(floor(.1 / (xs[1]-xs[0])))
        ax.plot_surface(X, Y, cs[x_cut-1:], 
                         rstride=r_stride, cstride=c_stride, alpha=.25)
#        x_an, y_an, _ = proj3d.proj_transform(xs[-1],
#                                              ts[-1],
#                                              mean(vs.mean()), ax.get_proj())
#       cset = ax.contour(X, Y, vs, zdir='x', offset=-xoffset, cmap=cm.jet)
        yoffset = .0; xoffset = .0; zoffset = 0;
#        contour_times = [.0, ts[-1] / 2., ts[-2]]
#        contour_xs = [xs[1], .0, .5, xs[-2]]
#        cset = ax.contour(X, Y, vs, contour_times,
#                           zdir='x',
#                           offset=ts[0]-xoffset,  
#                           colors = ('b', 'g', 'r'))
#        cset = ax.contour(X, Y, vs, contour_xs, 
#                          zdir='y',
#                          offset=xs[-1]+yoffset,
#                          colors = ('m', 'b', 'g', 'r'))
#        tk_mid = where(ts == ts[-1]/2.)[0][0];
#        for tk,slice_color  in zip([0, tk_mid, -1],
#                                   ['b', 'g', 'r']):
#            slice_vs = vs[:, tk];
#            slice_xs = xs;
#            slice_ts = ones_like(slice_vs)*ts[tk];
#            ax.plot(slice_ts, slice_xs, slice_vs,
#                    color = slice_color,
#                    linewidth = 2*line_width)

            
        
#        cset = ax.contour(X, Y, vs, zdir='z', offset=-zoffset, cmap=cm.jet)
        ax.set_xlim((ts[0]-xoffset,ts[-1]))
        ax.set_ylim((xs[0],xs[-1]+yoffset))
        
        ax.set_xlabel('$t$', fontsize = xlabel_font_size); 
        ax.set_ylabel('$x$',fontsize = xlabel_font_size)
        ax.set_zlabel(r'$\alpha(x,t)$', fontsize = xlabel_font_size);

        
        ticks = [ts[0], ts[-1]]
        ax.set_xticks(ticks)
        ax.set_xticklabels([r'$%.1f$'%tick for tick in ticks])
        ticks = [.0, xs[-1]]
        ax.set_yticks(ticks)
        ax.set_yticklabels([r'$%.0f$'%tick for tick in ticks])
        max_v = 2.0 #amax(vs)
#        yticks(ticks, [r'$%d$'%tick for tick in ticks])
        ticks = [ -2., .0, 2.0]
        ax.set_zticks(ticks)
        ax.set_zticklabels([r'$%.1f$'%tick for tick in ticks])

        ax.set_title(plot_titles[pidx], fontsize = xlabel_font_size)
        for label in ax.xaxis.get_majorticklabels():
            label.set_fontsize(label_font_size)
        for label in ax.yaxis.get_majorticklabels():
            label.set_fontsize(label_font_size  )
        for label in ax.zaxis.get_majorticklabels():
            label.set_fontsize(label_font_size  )
            
#        t = add_inner_title(ax, chr(65+pidx), loc=3,
#                            size=dict(size=ABCD_LABEL_SIZE))
#        t.patch.set_ec("none")
#        t.patch.set_alpha(0.5)
    
    get_current_fig_manager().window.showMaximized()        
    if None!= fig_name:
        lfig_name = os.path.join(FIGS_DIR, fig_name + '_control_surf.pdf')
        print 'saving to ', lfig_name
        savefig(lfig_name)


def visualizeRegimes(regimeParams,
                     Tf=1.5, energy_eps = .001, 
                     alpha_bounds = [-2,2],
                     fig_name = None):
    #Visualize:
    from mpl_toolkits.mplot3d.axes3d import Axes3D,proj3d
    line_width = 3
    ################################
    ### CONTROL SURFACES:
    ####################################
    fig = figure(figsize = (17, 18))
    fig.hold(True)
    subplots_adjust(hspace = .15,wspace = .25,
                     left=.15, right=.975,
                     top = .95, bottom = .05)
  
    for pidx, params in enumerate(regimeParams):
        hjbSoln = HJBSolution.load(mu_beta_Tf = params[::2]+[Tf],
                                    energy_eps = energy_eps)
        print energy_eps
        print 'mu,tc,b = %.2f,%.2f,%.2f'%(hjbSoln._mu,hjbSoln._tau_char, hjbSoln._beta) 
        ts,xs,vs,cs = hjbSoln._ts, hjbSoln._xs, hjbSoln._vs, hjbSoln._cs
        xs = xs[1:-1];
        cs = cs[1:, :]
        print 'delta t = %.4f, delta x = %.4f'%(ts[2]-ts[1], xs[2]-xs[1])
        ax = fig.add_subplot(ceil(len(regimeParams)/2), 2, 1+pidx,
                              projection='3d')
#    els = [40, 40, 40, 40]
#    azs = [70, 70, 70,70]
#        el = els[pidx]; az = azs[pidx];
        el = 40; az =70;
        ax.view_init(elev = el, azim= az)
        X, Y = np.meshgrid(ts, xs)
        r_stride = int(floor(.1 / (ts[1]-ts[0])))
        c_stride = int(floor(.1 / (xs[1]-xs[0])))
        ax.plot_surface(X, Y, cs, 
                         rstride=r_stride, cstride=c_stride, alpha=.25)
        yoffset = .2; xoffset = .1; zoffset = 0;
        tk_mid = where(ts == ts[-1]/2.)[0][0];
        for tk,slice_color  in zip([0, tk_mid, -1],
                                   ['b', 'g', 'r']):
            slice_cs = cs[:, tk];
            slice_xs = xs;
            slice_ts = ones_like(slice_cs)*ts[tk];
            ax.plot(slice_ts, slice_xs, slice_cs,
                    color = slice_color,
                    linewidth = 2*line_width)
        ax.set_xlim((ts[0]-xoffset,ts[-1]+xoffset))
        ax.set_ylim((xs[0]-yoffset,xs[-1]+yoffset))       
        ax.set_xlabel('$t$', fontsize = xlabel_font_size); 
#        ax.set_zlabel(r'$\alpha(x,t)$', fontsize = xlabel_font_size);
#        ax.set_ylabel(r'$x$', fontsize = xlabel_font_size);
#        yx = [.2,.2,.25,.25]
        ax.text2D(.125, .2, '$x$',
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes,
            fontsize = xlabel_font_size)
        ax.text2D(.05, .6, r'$\alpha(x,t)$',
                         fontsize = xlabel_font_size,
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform=ax.transAxes)
        
        ticks = [ts[0], ts[-1]]
        ax.set_xticks(ticks)
        ax.set_xticklabels([r'$%.1f$'%tick for tick in ticks])
        ticks = [.0, xs[-1]]
        ax.set_yticks(ticks)
        ax.set_yticklabels([r'$%.0f$'%tick for tick in ticks])
        max_a = 2.0 #amax(vs)
        min_a = -2.0 #amax(vs)
        ax.set_zlim((min_a, max_a))
        ticks = [max_a]
        ax.set_zticks(ticks)
        ax.set_zticklabels([r'$%.0f$'%tick for tick in ticks])
        ax.text2D(.10, .4, r'$%.0f$' %min_a,
                         fontsize = label_font_size,
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform=ax.transAxes)

        for label in ax.xaxis.get_majorticklabels():
            label.set_fontsize(label_font_size)
        for label in ax.yaxis.get_majorticklabels():
            label.set_fontsize(label_font_size  )
        for label in ax.zaxis.get_majorticklabels():
            label.set_fontsize(label_font_size  )
            
        t = add_inner_title(ax, '(%s)'%chr(65+pidx), loc=3,
                            size=dict(size=ABCD_LABEL_SIZE))
        t.patch.set_ec("none")
        t.patch.set_alpha(0.5)
        
    get_current_fig_manager().window.showMaximized()        
    if None!= fig_name:
        lfig_name = os.path.join(FIGS_DIR, fig_name + '_controlsurf.pdf')
        print 'saving to ', lfig_name
        savefig(lfig_name, dpi = 300)
        
    print 'exiting early!!!'
    return

    ################################
    ### VALUE FUNC SURFACES:
    ####################################
    fig = figure(figsize = (17, 18))
    fig.hold(True)
    subplots_adjust(hspace = .15,wspace = .25,
                     left=.15, right=.975,
                     top = .95, bottom = .05)
  
    for pidx, params in enumerate(regimeParams):
        hjbSoln = HJBSolution.load(mu_beta_Tf = params[::2]+[Tf], energy_eps = energy_eps)
        print 'mu,tc,b = %.2f,%.2f,%.2f'%(hjbSoln._mu,hjbSoln._tau_char, hjbSoln._beta) 
        ts,xs,vs,cs = hjbSoln._ts, hjbSoln._xs, hjbSoln._vs, hjbSoln._cs
        
        print 'delta t = %.4f, delta x = %.4f'%(ts[2]-ts[1], xs[2]-xs[1])
        ax = fig.add_subplot(ceil(len(regimeParams)/2), 2, 1+pidx,
                              projection='3d')
        el = 30; az = -25;
        ax.view_init(elev = el, azim= az)
        X, Y = np.meshgrid(ts, xs)
        r_stride = int(floor(.1 / (ts[1]-ts[0])))
        c_stride = int(floor(.1 / (xs[1]-xs[0])))
        ax.plot_surface(X, Y, vs, 
                         rstride=r_stride, cstride=c_stride, alpha=.25)
#        x_an, y_an, _ = proj3d.proj_transform(xs[-1],
#                                              ts[-1],
#                                              mean(vs.mean()), ax.get_proj())
#       cset = ax.contour(X, Y, vs, zdir='x', offset=-xoffset, cmap=cm.jet)
        yoffset = .2; xoffset = .1; zoffset = 0;
#        contour_times = [.0, ts[-1] / 2., ts[-2]]
#        contour_xs = [xs[1], .0, .5, xs[-2]]
#        cset = ax.contour(X, Y, vs, contour_times,
#                           zdir='x',
#                           offset=ts[0]-xoffset,  
#                           colors = ('b', 'g', 'r'))
#        cset = ax.contour(X, Y, vs, contour_xs, 
#                          zdir='y',
#                          offset=xs[-1]+yoffset,
#                          colors = ('m', 'b', 'g', 'r'))
        tk_mid = where(ts == ts[-1]/2.)[0][0];
        for tk,slice_color  in zip([0, tk_mid, -1],
                                   ['b', 'g', 'r']):
            slice_vs = vs[:, tk];
            slice_xs = xs;
            slice_ts = ones_like(slice_vs)*ts[tk];
            ax.plot(slice_ts, slice_xs, slice_vs,
                    color = slice_color,
                    linewidth = 2*line_width)

            
        
#        cset = ax.contour(X, Y, vs, zdir='z', offset=-zoffset, cmap=cm.jet)
        ax.set_xlim((ts[0]-xoffset,ts[-1]+xoffset))
        ax.set_ylim((xs[0]-yoffset,xs[-1]+yoffset))
        
        ax.set_xlabel('$t$', fontsize = xlabel_font_size); 
#        ax.set_ylabel('$x$', fontsize = xlabel_font_size)
        ax.set_zlabel('$w(x,t)$', fontsize = xlabel_font_size);
        ax.text2D(.75, .025, '$x$',
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes,
            fontsize = xlabel_font_size)
        
        ticks = [ts[0], ts[-1]]
        ax.set_xticks(ticks)
        ax.set_xticklabels([r'$%.1f$'%tick for tick in ticks])
        ticks = [.0, xs[-1]]
        ax.set_yticks(ticks)
        ax.set_yticklabels([r'$%.1f$'%tick for tick in ticks])
        max_v = 2.0 #amax(vs)
        ticks = [.0, max_v]
        ax.set_zticks(ticks)
        ax.set_zticklabels([r'$%.1f$'%tick for tick in ticks])

        for label in ax.xaxis.get_majorticklabels():
            label.set_fontsize(label_font_size)
        for label in ax.yaxis.get_majorticklabels():
            label.set_fontsize(label_font_size  )
        for label in ax.zaxis.get_majorticklabels():
            label.set_fontsize(label_font_size  )
            
        t = add_inner_title(ax, '(%s)'%chr(65+pidx), loc=3,
                            size=dict(size=ABCD_LABEL_SIZE))
        t.patch.set_ec("none")
        t.patch.set_alpha(0.5)
        

#        text(-.1, 1.0, '(%s)'%chr(65+pidx),
#                horizontalalignment='center', verticalalignment='center',
#                transform=ax.transAxes,
#                fontsize = ABCD_LABEL_SIZE)
    
    get_current_fig_manager().window.showMaximized()        
    if None!= fig_name:
        lfig_name = os.path.join(FIGS_DIR, fig_name + '_valuesurf.pdf')
        print 'saving to ', lfig_name
        savefig(lfig_name)
        
#    print 'exiting early!!!'
#    return
    
################################
####NOW PLOT SNAPSHOTS:
################################
    cuts_fig = figure(figsize = (17, 18))
    subplots_adjust(hspace = .25, wspace = .4,
                     left=.15, right=.975,
                     top = .95, bottom = .05)
    N_regimes = len(regimeParams)
    for pidx, params in enumerate(regimeParams):
        hjbSoln = HJBSolution.load(mu_beta_Tf = params[::2]+[Tf], energy_eps = energy_eps)
        print 'mu,tc,b = %.2f,%.2f,%.2f'%(hjbSoln._mu,hjbSoln._tau_char, hjbSoln._beta) 
        ts,xs,vs,cs = hjbSoln._ts, hjbSoln._xs, hjbSoln._vs, hjbSoln._cs
        axv = cuts_fig.add_subplot(4,2,1+ 2*pidx)
        axc = cuts_fig.add_subplot(4,2,1+ 2*pidx+1)
        
        tk_mid = where(ts == ts[-1]/2.)[0][0];
#        print tk_mid, ts[tk_mid]
        
        labels = {0:r'$t=0$',
                  tk_mid:r'$t=t^*/2$',
                  -1:r'$t=t^*$'}
        for tk in [0, tk_mid, -1]:
            vk = vs[:, tk];
            ck = cs[:, tk];
            
            axv.plot(xs[3:], vk[3:], label=labels[tk], linewidth = line_width)
            axc.plot(xs[4:], ck[3:], label=labels[tk], linewidth = line_width)
        if 0 == pidx:
            fontP = FontProperties()
            fontP.set_size(label_font_size)
            axv.legend(loc = 'upper left', prop = fontP, fancybox = True)
            #axc.legend()
        axc.set_ylim(amin(cs)-.1, amax(cs)+.1);
        axv.set_xlabel('$x$', fontsize = xlabel_font_size);
        axc.set_xlabel('$x$', fontsize = xlabel_font_size);
        axv.set_ylabel('$w(x,t)$', fontsize = xlabel_font_size);
        axc.set_ylabel(r'$\alpha(x,t)$', fontsize = xlabel_font_size);
        
        x_min = xs[0] * .8;
        axv.set_xlim(x_min, 1.)
        axc.set_xlim(x_min, 1.)
        #Value ticks:
        
        axv.set_yticks((0, 1., 2.0))
        axv.set_yticklabels(('$0$', '$1$', '$2$'), fontsize = label_font_size)
        axc.set_yticks((alpha_bounds[0], 0,alpha_bounds[1]))
        axc.set_yticklabels(('$%.0f$'%alpha_bounds[0], '$0$','$%.0f$'%alpha_bounds[1]),
                             fontsize = label_font_size)
        
        for ax in [axv, axc]:
            ax.set_xticks((x_min/2., 0, 1.0, ))
            ax.set_xticklabels(('$%.1f$'%(x_min/2.), '$0$','$1$'), fontsize = label_font_size)       
#        for ax in [axc, axv]:
#            for label in ax.xaxis.get_majorticklabels() + ax.yaxis.get_majorticklabels():
#                label.set_fontsize(label_font_size )
#        inner_tag = chr(65+pidx)
#        t = add_inner_title(axv, inner_tag, loc=3,
#                              size=dict(size=ABCD_LABEL_SIZE))
#        t.patch.set_ec("none"); t.patch.set_alpha(0.5)
#        t = add_inner_title(axc,chr(65+pidx+N_regimes), loc=3,
#                              size=dict(size=ABCD_LABEL_SIZE))
#        t.patch.set_ec("none"); t.patch.set_alpha(0.5)
        axv.text(-.15, 1.0, '(%s)'%chr(65+2*pidx),
                horizontalalignment='center', verticalalignment='center',
                transform=axv.transAxes,
                fontsize = ABCD_LABEL_SIZE)
        axc.text(-.15, 1.0, '(%s)'%chr(65+2*pidx+1),
                horizontalalignment='center', verticalalignment='center',
                transform=axc.transAxes,
                fontsize = ABCD_LABEL_SIZE)
        
    get_current_fig_manager().window.showMaximized()        
    if None != fig_name:
        lfig_name = os.path.join(FIGS_DIR, fig_name + '_vc_cuts.pdf')
        print 'saving to ', lfig_name
        savefig(lfig_name)
        
        
def compareEffectOfEnergyEps(regimeParams, Tf, values_of_eps = [.001, .1],
                             alpha_bounds = [-2., 2.],
                             fig_name = None):
    
    ####CONTROLS SNAPSHOTS:
    cuts_fig = figure(figsize = (17, 20))
    subplots_adjust(hspace = .25, wspace = .4,
                     left=.15, right=.975,
                     top = .95, bottom = .05)
    N_regimes = len(regimeParams)
    N_eps = len(values_of_eps)
    for pidx, params in enumerate(regimeParams):
        for eidx, energy_eps in enumerate(values_of_eps):
            hjbSoln = HJBSolution.load(mu_beta_Tf = params[::2]+[Tf], energy_eps = energy_eps)
            print 'mu,tc,b = %.2f,%.2f,%.2f'%(hjbSoln._mu,hjbSoln._tau_char, hjbSoln._beta) 
            print 'Tf, energy_eps   = %.3f,%.3f '%(hjbSoln._ts[-1],hjbSoln.energy_eps)
            ts,xs,vs,cs = hjbSoln._ts, hjbSoln._xs, hjbSoln._vs, hjbSoln._cs
            ax = cuts_fig.add_subplot(N_regimes,N_eps,1 + N_eps*pidx + eidx)
            
            tk_mid = where(ts == ts[-1]/2.)[0][0];
    #        print tk_mid, ts[tk_mid]
            
            labels = {0:r'$t=0$',
                      tk_mid:r'$t=t^*/2$',
                      -1:r'$t=t^*$'}
            for tk in [0, tk_mid, -1]:
                ck = cs[:, tk];
                ax.plot(xs[4:], ck[3:], linewidth = 2, label=labels[tk])
            if 0 == pidx and 0 == eidx:
                fontP = FontProperties()
                fontP.set_size(label_font_size)
                ax.legend(loc = 'best', prop = fontP, fancybox = True)
            if  0 == pidx:
                ax.set_title(r'$ \epsilon = %.3f $'%energy_eps,
                         fontsize = xlabel_font_size)
#            ax.title(r' eps = %.3f '%energy_eps)
            
            #axc.legend()
            ax.set_ylim(alpha_bounds[0]-.1,
                        alpha_bounds[1]+.1);
            if (N_regimes-1 == pidx):
                ax.set_xlabel('$x$', fontsize = xlabel_font_size);
            if (0 == eidx):
                ax.set_ylabel(r'$\alpha(x,t)$', fontsize = xlabel_font_size);
            
            inner_tag ='(%s)'%chr(65+pidx*2 + eidx)
            ax.text(-.2, 1.0, inner_tag,
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes,
                fontsize = ABCD_LABEL_SIZE)
            
            x_min = xs[0] * .8;
            ax.set_xlim(x_min, 1.)
            ax.set_xlim(x_min, 1.)
            
            ax.set_yticks((alpha_bounds[0], 0,alpha_bounds[1]))
            ax.set_yticklabels(('$%.1f$'%alpha_bounds[0], '$0$','$%.1f$'%alpha_bounds[1]),
                                 fontsize = label_font_size)
            ax.set_xticks((x_min/2., 0, 1.0, ))
            ax.set_xticklabels(('$%.1f$'%(x_min/2.), '$0$','$1$'), fontsize = label_font_size)
            
        
           
    get_current_fig_manager().window.showMaximized()        
    if None != fig_name:
        lfig_name = os.path.join(FIGS_DIR, fig_name + '_eps_comparison.pdf')
        print 'saving to ', lfig_name
        savefig(lfig_name)
        
    
def timeHJB(tb = [.5, 1.25],
            Tf = 1.5, energy_eps = .1, alpha_bounds = (-2., 2.), fig_name = None):
    import time
    
#    xmin = -1.5; #HJBSolver.calculate_xmin(alpha_bounds, tb, num_std = 1.0)
    xmin = -.5;
    dx = HJBSolver.calculate_dx(alpha_bounds, tb, xmin)
    dt = HJBSolver.calculate_dt(alpha_bounds, tb, dx, xmin, factor = 4.)
    
    #Set up solver
    #TODO: The way you pass params and the whole object-oriented approach is silly. Tf changes for each solve and atb don't, so maybe rething the architecture!!!
    start = time.clock()
    S = HJBSolver(dx, dt, Tf, xmin)
    #the v solution:
    xs, ts, vs =  S.solve(tb, alpha_bounds=alpha_bounds, energy_eps=energy_eps)
    end = time.clock()
    print 'compute time = ',end-start, 's'
    
    
def solveRegimes(regimeParams, Tf = 1.5, energy_eps = .001):
    for params in regimeParams:
        print 'm,tc,b =' , params
        HJBDriver(params, Tf, energy_eps =energy_eps, save_soln=True)
    

########################
class HJBSolution():
    def __init__(self,params, xs, ts, vs, cs, energy_eps ):
        self._cs = cs
        self._vs = vs;
        self._ts  = ts;
        self._xs  = xs;
        
        self._mu = params[0]
        self._tau_char = params[1]
        self._beta = params[2]
        
        self.energy_eps = energy_eps
                
    def save(self, file_name=None):
        if None == file_name:
            file_name = 'HJBSoln_m=%.1f_b=%.1f_Tf=%.1f_eps=%.3f'%(self._mu,
                                                         self._beta,
                                                         self._ts[-1],
                                                         self.energy_eps);
        print 'saving path to ', file_name
        file_name = os.path.join(RESULTS_DIR, file_name + '.hjb')
        import cPickle
        dump_file = open(file_name, 'wb')
        cPickle.dump(self, dump_file, 1) # 1: bin storage
        dump_file.close()
        
    @staticmethod
    def load(file_name=None, mu_beta_Tf=None, energy_eps=.001):
        ''' not both can be none!!!'''
        if None == file_name:
            mu,beta,Tf = [x for x in mu_beta_Tf]
            file_name = 'HJBSoln_m=%.1f_b=%.1f_Tf=%.1f_eps=%.3f'%(mu,
                                                         beta,
                                                         Tf,
                                                         energy_eps);
        file_name = os.path.join(RESULTS_DIR, file_name + '.hjb') 
        print 'loading ', file_name
        import cPickle
        load_file = open(file_name, 'r')
        hjbSoln = cPickle.load(load_file)        
        return hjbSoln
########################
def HJBDriver(params, Tf,
              energy_eps = .001, alpha_bounds = (-2., 2.), 
              save_soln=True,
              save_soln_file_name = None,
              xmin = None,
              dx = None,
              dt = None):

    if None == xmin:
        xmin = HJBSolver.calculate_xmin(alpha_bounds, params, num_std = 2.0)
    if None == dx:  
        dx = HJBSolver.calculate_dx(alpha_bounds, params, xmin)
    if None == dt:
        dt = HJBSolver.calculate_dt(alpha_bounds, params, dx, xmin)
    
    #Set up solver
    S = HJBSolver(dx, dt, Tf, xmin)
    
    #the v solution:
    xs, ts, vs, cs =  S.solve(params,
                               alpha_bounds=alpha_bounds,
                                energy_eps=energy_eps,
                                 visualize=False)
    
    (HJBSolution(params, xs, ts, vs, cs, energy_eps )).save(file_name = save_soln_file_name)



def investigateMaxSpped(regimeParams,
                         regimeTitles,
                         alpha_bounds = (-2., 2.)):
    
    for params in regimeParams:
        print regimeTitles[(params[0],params[2])], ':'
        mu, tc, beta = params[0], params[1], params[2]
        
        xmin = HJBSolver.calculate_xmin(alpha_bounds, params, num_std = 2.0);
        speed_at_min = (mu + alpha_bounds[1] - xmin / tc);
        speed_at_th = -(mu + alpha_bounds[0] - 1.0 / tc);
        
        print 'xmin=%.2f, speed_at_min = %.2f, speed_at_th=%.2f'%(xmin,
                                                                  speed_at_min,
                                                                  speed_at_th)
        
        speed_labels = {speed_at_min:'speed_at_min',
                        speed_at_th:'speed_at_th'}
        print 'max_speed at ', speed_labels[max(speed_labels.keys())]
            

def investigateIntegrationParams(regimeParams,
                                 regimeTitles,
                                 integrate_new = True,
                                 Tf = 1.5,
                                 energy_eps = .001,
                                 alpha_bounds = (-2., 2.)):
    
    for params in regimeParams:
        regime_name = regimeTitles[(params[0],params[2])]
        print regimeTitles[(params[0],params[2])]
        xmin = HJBSolver.calculate_xmin(alpha_bounds, params, num_std = 2.0)
        dx_base = HJBSolver.calculate_dx(alpha_bounds, params, xmin)
        dt_base = HJBSolver.calculate_dt(alpha_bounds, params, dx_base, xmin)
        
        factors = array([5, 2, 1, .5])
#        factors = array([5])
        
        if integrate_new:
            #dx refinement
            for factor in factors:
                dx = dx_base * factor
                dt = dt_base
                soln_name = 'HJB_refine_%s_dt_base_dxX%.2f'%(regime_name,factor)
                HJBDriver(params, Tf, energy_eps =energy_eps,
                          save_soln=True,
                          save_soln_file_name=soln_name,
                          dx= dx, dt = dt)
#            dt refinement
            for factor in factors:
                dx = dx_base 
                dt = dt_base * factor
                soln_name = 'HJB_refine_%s_dx_base_dtX%.2f'%(regime_name,factor) 
                HJBDriver(params, Tf, energy_eps =energy_eps,
                          save_soln=True,
                          save_soln_file_name=soln_name,
                          dx= dx, dt = dt)
        
        #VISUALIZE:
        figure()
        #dx refinement
        v00s_dx_refine = empty_like(factors) 
        for fidx,factor in enumerate(factors):
            dx = dx_base * factor
            dt = dt_base
            #load soln:
            soln_name = 'HJB_refine_%s_dt_base_dxX%.2f'%(regime_name, factor)
            hjbSoln = HJBSolution.load(file_name=soln_name)
            ts,xs,vs,cs = hjbSoln._ts, hjbSoln._xs, hjbSoln._vs, hjbSoln._cs
            #Rips V[0][0]
            value_at_zero = vs[abs(xs).argmin(), 0]
            v00s_dx_refine[fidx] = value_at_zero
            print "dx= %.5f, dt= %.5f , v00 = %.5f"%(dx, dt, value_at_zero)
        dxs = dx_base * factors
        ax = subplot(211)
        plot(dxs, v00s_dx_refine, 'ok', markersize=10);
        title(regime_name);
        xlabel(r'$\Delta x$')
        vlines(dx_base, amin(v00s_dx_refine), amax(v00s_dx_refine))
        ylim((amin(v00s_dx_refine), amax(v00s_dx_refine)))
        ticks = dxs
        ax.set_xticks(ticks)
        ax.set_xticklabels([r'$ %.1f \Delta x$'%factor for factor in factors])
        ticks = [amin(v00s_dx_refine), amax(v00s_dx_refine)]
        ax.set_yticks(ticks)
        ax.set_yticklabels([r'$%.6f$'%tick for tick in ticks])
        
        
        #dt refinement
        v00s_dt_refine = empty_like(factors)
        for fidx, factor in enumerate( factors):
            dx = dx_base 
            dt = dt_base * factor
            soln_name = 'HJB_refine_%s_dx_base_dtX%.2f'%(regime_name, factor)
            hjbSoln = HJBSolution.load(file_name=soln_name)
            ts,xs,vs,cs = hjbSoln._ts, hjbSoln._xs, hjbSoln._vs, hjbSoln._cs
            #Rips V[0][0]
            value_at_zero = vs[abs(xs).argmin(), 0]
            v00s_dt_refine[fidx] = value_at_zero
            print "dx= %.5f, dt= %.5f , v00 = %.5f"%(dx, dt, value_at_zero)
        dts = dt_base *factors
        ax = subplot(212)
        plot(dts, v00s_dt_refine,'ok', markersize=10);
        vlines(dt_base, amin(v00s_dt_refine), amax(v00s_dt_refine))
        xlabel(r'$\Delta t$')
        ylim((amin(v00s_dt_refine), amax(v00s_dt_refine)))
        ticks = [amin(v00s_dt_refine), amax(v00s_dt_refine)]
        ax.set_yticks(ticks)
        ax.set_yticklabels([r'$%.6f$'%tick for tick in ticks])
        ticks = dts
        ax.set_xticks(ticks)
        ax.set_xticklabels([r'$ %.1f \Delta t$'%factor for factor in factors])
      
        
        
        print v00s_dx_refine
        print v00s_dt_refine
        
        lfig_name = os.path.join(FIGS_DIR,
                                 'Investigate_dxdt_v00_%s_.pdf'%regime_name)
        print 'saving to ', lfig_name
        savefig(lfig_name)
            
             

def PyVsCTinyDriver():
    energy_eps=.1
    params = array([1.0, .5,   1.314]);
    S = HJBSolver(.5, .099, .2, -.5)

    print 'const int num_nodes = %d; \
            const int num_time_slices = %d;'%(len(S._xs),len(S._ts))
    
    for var_name, arr in zip(['xs', 'ts', 'TCs'],
                       [S._xs, S._ts, 
                        S._getTCs(S._xs,
                                alpha_bounds[1]+params[0],
                                params[1],
                                 params[2])]):
        print_str = 'double %s[%d] = {'%(var_name,len(arr));
        for v in arr:
            print_str += '%.3f,'%v 
        print_str = print_str[:-1]
        print_str += '};'
        print print_str
    
    #the Py solution:
    xs, ts, py_vs, py_cs =  S.solve(params,
                               alpha_bounds=alpha_bounds,
                                energy_eps=energy_eps,
                                 visualize=False)
    print 'vs'
    for tk in xrange(3):
        print tk, ['%.3f' %py_vs[xk,tk] for  xk in xrange(len(xs))]
    print 'cs'
    for tk in xrange(3):
        print tk, ['%.3f' %py_cs[xk,tk] for  xk in xrange(len(xs)-1)]

def PyVsCDriver(regimeParams, Tf = 1.5, energy_eps=.1, alpha_bounds=(-2,2)):
    for params in regimeParams:
        print params
        xmin = HJBSolver.calculate_xmin(alpha_bounds, params, num_std = 2.0)
        dx = HJBSolver.calculate_dx(alpha_bounds, params, xmin)
        dt = HJBSolver.calculate_dt(alpha_bounds, params, dx, xmin)
    
        #Set up solver
        S = HJBSolver(dx, dt, Tf, xmin)
#        #generate parameters into C code::
#        print 'const int num_nodes = %d; \
#                const int num_time_slices = %d;'%(len(S._xs),len(S._ts))
#        
#        for var_name, arr in zip(['xs', 'ts', 'TCs'],
#                           [S._xs, S._ts, 
#                            S._getTCs(S._xs,
#                                    alpha_bounds[1]+params[0],
#                                    params[1],
#                                     params[2])]):
#            print_str = 'double %s[%d] = {'%(var_name,len(arr));
#            for v in arr:
#                print_str += '%.3f,'%v 
#            print_str = print_str[:-1]
#            print_str += '};'
#            print print_str
#        return
        #the C solution:
        c_start = time.clock();
        print alpha_bounds
        xs, ts, C_vs, C_cs =  S.c_solve(params,
                                   alpha_bounds=alpha_bounds,
                                    energy_eps=energy_eps)
        c_end = time.clock();
        (HJBSolution(params, xs, ts,
                      C_vs, C_cs, energy_eps )).save(file_name = 'PyVsC_C_soln')
#        print ['%.3f' %C_cs[xk,0] for  xk in xrange(len(xs)-1)]
#        print ['%.3f' %C_cs[xk,-1] for  xk in xrange(len(xs)-1)]
        
        #the Py solution:
        py_start = time.clock();
        xs, ts, py_vs, py_cs =  S.solve(params,
                                   alpha_bounds=alpha_bounds,
                                    energy_eps=energy_eps,
                                     visualize=False)
        py_end  = time.clock();
        (HJBSolution(params, xs, ts,
                      py_vs, py_cs, energy_eps )).save(file_name = 'PyVsC_Py_soln')
#        print '00', ['%.3f' %py_cs[xk,0] for  xk in xrange(len(xs)-1)]
#        print len(S._ts)-1, ['%.3f' %py_cs[xk,-1] for  xk in xrange(len(xs)-1)]
        
        #Analyze:
        print 'PyVsC L1 Error: '
        print 'vs error:', amax(abs(py_vs - C_vs))
        print 'cs error:', amax(abs(py_cs - C_cs))
        print 'times: C=%.3f, Py=%.3f secs'%(c_end-c_start,
                                             py_end-py_start)
    
    print 'PyVsC Complete - check results'
    
def SingleSolutionForAllT(params, Tf_long = 3.,
                           Tf_short = 1., energy_eps=.1, 
                           resimulate=False):
        xmin = HJBSolver.calculate_xmin(alpha_bounds, params, num_std = 2.0)
        dx = HJBSolver.calculate_dx(alpha_bounds, params, xmin)
        dt = HJBSolver.calculate_dt(alpha_bounds, params, dx, xmin)
    
        if resimulate:
        #Set up solver
            S = HJBSolver(dx, dt, Tf_long, xmin)
    
            #the C solution:
            
            long_xs, long_ts, long_vs, long_cs =  S.c_solve(params,
                                       alpha_bounds=alpha_bounds,
                                        energy_eps=energy_eps)
            
            (HJBSolution(params, long_xs, long_ts,
                          long_vs, long_cs , energy_eps )).save(file_name = 'SingleSoln_long_T=%.1f'%Tf_long)
    
            #Set up solver
            S = HJBSolver(dx, dt, Tf_short, xmin)
    
            #the C solution:
            
            short_xs, short_ts, short_vs, short_cs =  S.c_solve(params,
                                       alpha_bounds=alpha_bounds,
                                        energy_eps=energy_eps)
            
            (HJBSolution(params, short_xs, short_ts,
                          short_vs, short_cs , energy_eps )).save(file_name = 'SingleSoln_short_T=%.1f'%Tf_short)
            
        longSoln  = HJBSolution.load(file_name = 'SingleSoln_long_T=%.1f'%Tf_long)
        shortSoln = HJBSolution.load(file_name = 'SingleSoln_short_T=%.1f'%Tf_short)
        
        long_ts = longSoln._ts;
        
        indxs = where(long_ts>=Tf_long-Tf_short)
        long_cropped_vs = squeeze(longSoln._vs[:, indxs]);
#        long_cropped_vs = longSoln._vs[:, 644:];
        short_vs = shortSoln._vs
        
        #Analyze:
        print 'LongVsShort L1 Error: '
        print 'ts error:', amax(abs(long_ts[indxs] - shortSoln._ts-2.))
        print 'vs error:', amax(abs(long_cropped_vs - short_vs))
        print 'vs relative error:', amax(abs(long_cropped_vs - short_vs) / (short_vs+1e-8))
            

if __name__ == '__main__':
    from pylab import *
    
    Tf = 1.5; 
    energy_eps = .001; alpha_bounds = (-2., 2.);
    
    tau_char = .5;
    beta_high = 1.5
    beta_low = .3;
    mu_high = 1.5
    mu_low = .1
#    regimeParams = [[mu_high/tau_char, tau_char, beta_high] ]
    regimeParams = [[mu_high/tau_char, tau_char, beta_low], 
                    [mu_high/tau_char, tau_char, beta_high],
                    [mu_low/tau_char, tau_char, beta_low], 
                    [mu_low/tau_char, tau_char, beta_high] ]
    regimeTitles = {(mu_high/tau_char, beta_low):'SuperThresh-LowNoise',
                    (mu_high/tau_char, beta_high):'SuperThresh-HighNoise', 
                    (mu_low/tau_char, beta_low):'SubThresh-LowNoise', 
                    (mu_low/tau_char, beta_high):'SubrThresh-highNoise'}

    
#    highEpsParams  = [mu_low/tau_char, tau_char, beta_high];
    
#    solveRegimes(regimeParams, Tf, energy_eps=.001)
#    visualizeRegimes(regimeParams[:], Tf, 
#                    fig_name='Regimes')
#    visualizeRegimes(regimeParams[:], Tf, 
#                     energy_eps = .1,
#                    fig_name='RegimesHighEps')

#    solveRegimes(regimeParams, Tf, energy_eps=.1)
#    visualizeRegimes(regimeParams, Tf, energy_eps=.1,
#                     fig_name='Regimes')
#    visualizeRegimes(regimeParams[2:],
#                      Tf, fig_name='Regimes_presentation')
#    compareEffectOfEnergyEps(regimeParams, Tf, values_of_eps = [.001, .1],
#                             fig_name = 'Regimes')



#    timeHJB()
#    for atb in [[2.0, .5, .75],
#                [1., .5, 1.0],
#                [.5, 1.0, 1.0]]:
#        
#        visualizeHJB(atb)

#    investigateMaxSpped(regimeParams, regimeTitles,alpha_bounds)
#    investigateIntegrationParams(regimeParams[0:],
#                                 regimeTitles=regimeTitles,
#                                 integrate_new = False)
    

#C driver;
#    PyVsCTinyDriver()
#    PyVsCDriver([[0.20000000000000001, 0.5, 1.5]],
#                 Tf = 3.0,
#                 energy_eps = .1,
#                 alpha_bounds = alpha_bounds);
#    PyVsCDriver(regimeParams, Tf = 1.5);


#    SingleSolutionForAllT(regimeParams[2],
#                          resimulate= False)

    
    show()
    