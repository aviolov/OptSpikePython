# -*- coding:utf-8 -*-
"""
Created on Mar 13, 2012
@author: alex
"""
from __future__ import division
import numpy as np
from numpy import *
from copy import deepcopy
from numpy.random import randn, rand
from matplotlib.pyplot import savefig
RESULTS_DIR = '/home/alex/Workspaces/Python/OptSpike/Results/PathSimulator/'
FIGS_DIR = '/home/alex/Workspaces/Latex/OptSpike/Figs/PathSimulator'

import os
for D in [FIGS_DIR, RESULTS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)

ABCD_LABEL_SIZE = 30
        
def simulatePath(mu, tau_char, beta, 
                 Tf, dt = None, 
                 save_path=False,
                 v_thresh = 1.0):
    #Set default dt:        
    if (None == dt):
        dt = 1e-3

    #Set the (fixed) integration  times:
    ts = arange(0., Tf, dt);

    #Preallocate space for the solution and the spike times:
    spike_ts = [];
    vs = zeros_like(ts);
    
    Bs = randn(len(ts));
    sqrt_dt = sqrt(dt);
    
    #THE MAIN INTEGRATION LOOP:
    for t, idx in zip(ts[1:], xrange(1, len(ts))):            
        v_prev = vs[idx-1];
        
        dv = (mu - v_prev / tau_char)*dt + beta*Bs[idx]*sqrt_dt;
         
        v_new = vs[idx-1] + dv
        
        if v_new >= v_thresh:        
            spike_ts.append(t)
            v_new = .0;
        
        vs[idx] = v_new;
    
    if save_path:
        savePath(mu, beta, tau_char, 
                 ts, vs, spike_ts)
    
    #Return:
    return vs, spike_ts

def savePath(mu, beta, tau_char,
             ts, vs, spike_ts):
    savedPath = Path(ts, vs, spike_ts, mu, beta, tau_char) 
    savedPath.save()

########################
class Path():
    def __init__(self, ts, vs, spike_ts, mu, beta, tau_char):
        self._vs = vs;
        self._ts  = ts;
        self._spike_ts = spike_ts;
        self._mu = mu
        self._tau_char = tau_char
        self._beta = beta
                
    def save(self, file_name=None):
#        path_data = {'path' : self}
        if None == file_name:
            file_name = 'Path_m=%d_b=%d'%(int(10*self._mu), int(10*self._beta));
        print 'saving path to ', file_name
        file_name = os.path.join(RESULTS_DIR, file_name + '.path')
        import cPickle
        dump_file = open(file_name, 'wb')
        cPickle.dump(self, dump_file, 1) # 1: bin storage
        dump_file.close()
        
    @staticmethod
    def load(file_name=None, mu_beta=None):
        ''' not both can be none!!!'''
        if None == file_name: 
            file_name = 'Path_m=%d_b=%d'%(int(10*mu_beta[0]),
                                          int(10*mu_beta[1]));
            
        import cPickle
        print file_name
        file_name = os.path.join(RESULTS_DIR, file_name + '.path')
        load_file = open(file_name, 'r')
        path = cPickle.load(load_file)        
        return path
 
    
def generateRegimePaths( regimeParams,
                         Tf = 30.,
                         dt = 1e-2):
    
    for param in regimeParams:
        mu,tau_char, beta = param[0], param[1], param[2]
        print 'm,tc,b = %.2f,%.2f,%.2f' %(mu,tau_char, beta)
        
        simulatePath(mu, tau_char, beta,
                     Tf, dt, save_path=True)
        

def visualizeRegimePaths(regimeParams, inner_titles,
                         save_fig=False, fig_tag  = ''):
    #Parametrization:
    mpl.rcParams['figure.figsize'] = 17, 6*2
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['figure.subplot.left'] = .15
    mpl.rcParams['figure.subplot.right'] =.975
    mpl.rcParams['figure.subplot.bottom'] = .075
    label_font_size = 32
    xlabel_font_size = 40
    
    fig = figure(figsize = (17, 20))
    subplots_adjust(hspace = .15, wspace = .2)
    spike_height = 1.5
    sub_sample = 20;
    plot_terminal_time = 8.0;
        
    for idx, params in enumerate(regimeParams):
        
        P = Path.load(mu_beta = params[0::2])
        Tf = P._ts[-1];
        
        #Post-processing:
        N = len(P._spike_ts)
        
        ax = subplot(2,2,idx+1)
        hold(True)
        plot (P._ts[::sub_sample], P._vs[::sub_sample], linewidth=.5)
#        title(r'$\alpha,\beta,\gamma = (%.3g,%.3g,%.3g) $' %(alpha,beta,gamma), fontsize = 32)
        if (0 == mod(idx,2)):
            ylabel('$X_t$', fontsize = xlabel_font_size)
        if 0<N:
            vlines(P._spike_ts, .0, spike_height*ones_like(P._spike_ts), linewidth=2);
        hlines(0, 0, P._ts[-1], linestyles='dashed',  linewidth=2);
        hlines(1.0, 0, P._ts[-1], 'r', linestyles='dashed',  linewidth=2);
        
#        ylim( (amin(P._vs), spike_height) );
        ylim( (-1.5, spike_height) );
#        xlim( (P._ts[0], P._ts[-1]))
        
        xlim( (P._ts[0], plot_terminal_time))
        ax.set_xticks((0, 8.0))
        ax.set_xticklabels(('$0$', '$8$'))
        ax.set_yticks((0, 1.0))
        ax.set_yticklabels(('$0$', '$1$'))

        if (idx > 1):
            xlabel('$t$', fontsize = xlabel_font_size)
        tick_params(labelsize = label_font_size)
        
#        def add_inner_title(ax, title, loc, size=None, **kwargs):
#            from matplotlib.offsetbox import AnchoredText
#            from matplotlib.patheffects import withStroke
#            if size is None:
#                size = dict(size=plt.rcParams['legend.fontsize'])
#            at = AnchoredText(title, loc=loc, prop=size,
#                              pad=0., borderpad=0.5,
#                              frameon=False, **kwargs)
#            ax.add_artist(at)
#            at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
#            return at
#        t = add_inner_title(ax, inner_titles[idx], loc=2,
#                             size=dict(size=ABCD_LABEL_SIZE))
#        t.patch.set_ec("none")
#        t.patch.set_alpha(0.5)
        
        text(-.1, 1.0, '(%s)'%inner_titles[idx],
              horizontalalignment='center',
              verticalalignment='center',
              transform=ax.transAxes,
              fontsize = ABCD_LABEL_SIZE)
    
    if save_fig:
        get_current_fig_manager().window.showMaximized() 
        filename = os.path.join(FIGS_DIR, 'path_T=%d_%s.pdf'%(Tf, 'combined'))
        print 'Saving fig to: ', filename
        savefig(filename, dpi=(300))
 
        
def batchSimulator():
#    for idx in range(1,9):
#    for idx in range(9,17):
#        sinusoidal_spike_train(5000.0, save_path=True, path_tag = 'crit_%d'%idx, 
#                           params = [.55, .5, .55, 2.0])
#
#        sinusoidal_spike_train(5000.0, save_path=True, path_tag = 'superT_%d'%idx, 
#                           params = [1.5, .3, 1.0, 2.0])
#
#        sinusoidal_spike_train(20000.0, save_path=True, path_tag = 'subT_%d'%idx, 
#                           params = [.4, .3, .4, 2.0])
#
#        sinusoidal_spike_train(5000.0, save_path=True, path_tag = 'superSin_%d'%idx, 
#                           params = [.1, .3, 2.0, 2.0])
    for idx in range(1,17):
#        sinusoidal_spike_train(5000.0, save_path=True, path_tag = 'crit_%d'%idx, 
#                           params = [.55, .5, .55, 2.0])
#
#        sinusoidal_spike_train(5000.0, save_path=True, path_tag = 'superT_%d'%idx, 
#                           params = [1.5, .3, 1.0, 2.0])

        sinusoidal_spike_train(20000.0, save_path=True, path_tag = 'subT_%d'%idx, 
                           params = [.4, .3, .4, 2.0])

#        sinusoidal_spike_train(5000.0, save_path=True, path_tag = 'superSin_%d'%idx, 
#                           params = [.1, .3, 2.0, 2.0])

def generateSDF(Is):
    N = len(Is)
    unique_Is = unique(Is)
    SDF = zeros_like(unique_Is)
    
    for (Ik, idx) in zip(unique_Is, arange(len(SDF))):
        SDF[idx] = sum(Is> Ik) / N;
        
    return SDF, unique_Is

def visualizeDistributions(file_name, fig_name):
    file_name = os.path.join(RESULTS_DIR, file_name)
    P = Path.load(file_name)
    Is = r_[(P._spike_ts[0], diff(array(P._spike_ts)))];
    
    SDF, unique_Is = generateSDF(Is)
    
    mpl.rcParams['figure.subplot.left'] = .15
    mpl.rcParams['figure.subplot.right'] =.95
    mpl.rcParams['figure.subplot.bottom'] = .15
    mpl.rcParams['figure.subplot.hspace'] = .4
    
    figure()
    ax = subplot(211)
    hist(Is, 100)
    title(r'$\alpha,\beta,\gamma = (%.3g,%.3g,%.3g) $' %(P._params._alpha, P._params._beta, P._params._gamma), fontsize = 24)
    xlabel('$I_n$', fontsize = 22);
    ylabel('$g(t)$', fontsize = 22);
    
    for label in ax.xaxis.get_majorticklabels():
                label.set_fontsize(20)
    for label in ax.yaxis.get_majorticklabels():
        label.set_fontsize(20)
    
    
    ax = subplot(212)
    plot(unique_Is, SDF, 'rx', markersize = 10)
    ylim((.0,1.))
    xlabel('$t$', fontsize = 22)
    ylabel('$1 - G(t)$', fontsize = 22)
    for label in ax.xaxis.get_majorticklabels():
                label.set_fontsize(20)
    for label in ax.yaxis.get_majorticklabels():
        label.set_fontsize(20)
            
    
    fig_name = os.path.join(FIGS_DIR, fig_name)
    print 'saving to ', fig_name
    savefig(fig_name)

def analyzeSuperTHighN():
    mu,tau_char, beta = 3.0 - 2.0, .5,  1.5
    
    P = None;
    try:
        P = Path.load(mu_beta = [mu, beta])
    except:
        vs, spike_ts = simulatePath(mu, tau_char, beta,
                                    1000., save_path=True)
        P = Path(None, vs, spike_ts, 
                   mu,tau_char, beta)
    
    ISIs = diff(r_[.0, P._spike_ts]);
    
    T_target = 1.5
    errors = ISIs - T_target;
    title('%.2f, %.2f '%(mu, beta))
    subplot(211);
    hist(errors, range = [-T_target, T_target]);
    subplot(212)
    hist(errors**2, range = [0, T_target**2]);
    
    print 'num spikes = ', len(ISIs)
    print 'mean ISIs = %.2f'%mean(ISIs)
    print 'mean squared error = %.2f'%(dot(errors, errors) /  len(ISIs))
    
    
    
 
if __name__ == '__main__':
    from pylab import *
    
    tau_char = .5;
    beta_high = 1.5
    beta_low = .3;
    mu_high = 1.5
    mu_low = .1
    regimeParams = [ [mu_high/tau_char, tau_char, beta_low],
                     [mu_high/tau_char, tau_char, beta_high],
                     [mu_low/tau_char, tau_char, beta_low],
                     [mu_low/tau_char, tau_char, beta_high]   ]
    
    inner_titles = {0:'A',
                    1:'B',
                    2:'C',
                    3:'D'}
    
#    generateRegimePaths(regimeParams,
#                        Tf = 15, dt= 5e-4)
    
#    visualizeRegimePaths(regimeParams, inner_titles,
#                            save_fig=True)

    analyzeSuperTHighN()
    
    show()
        