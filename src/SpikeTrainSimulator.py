# -*- coding:utf-8 -*-
"""
Created on Mar 13, 2012

@author: alex
"""
from __future__ import division
import numpy as np
from numpy import zeros, ones, array, r_, float64, matrix, bmat, Inf, ceil, arange
from numpy import zeros_like, diff, unique, ones_like
from copy import deepcopy
from numpy import sin, sqrt
from numpy.random import randn, rand
from matplotlib.pyplot import savefig
from scipy.integrate.odepack import odeint
from scipy.interpolate.interpolate import interp1d

RESULTS_DIR = '/home/alex/Workspaces/Python/OptSpike/SpikeTrains/'
FIGS_DIR = '/home/alex/Workspaces/Latex/OptSpike/Figs/'
import os

for D in [RESULTS_DIR, FIGS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)

class OUSimulator():
    def __init__(self, alpha, tauchar, beta, x0 = .0, x_thresh = 1.0):  
        self._path_params = OUParams(alpha, tauchar, beta,x0, x_thresh)
                
    def simulate(self, spikes_requested, dt = 1e-4):
        #Set the (fixed) integration  times:
        #        ts = arange(0., spikes_requested, dt);
        spike_ts = [];
        
        sqrt_dt = sqrt(dt);
        alpha, tauchar, beta, x0, x_thresh = self._path_params.getParams();

        #THE MAIN INTEGRATION LOOP:
        X = x0
        t = .0;
        recorded_spikes = 0;
        xs = [x0];
        while recorded_spikes < spikes_requested:
            xi = randn()
            dB = xi*sqrt_dt
            dX = (alpha - X/tauchar)*dt +  beta*dB

            X += dX;
            t += dt;
            if X >= x_thresh:
                spike_ts.append(t)
                X = x0;
                recorded_spikes += 1
            xs.append(X)
        
        #Return:
        path_params = deepcopy(self._path_params)
        simulatedSpikeTrain = SpikeTrain(spike_ts, path_params)
        
        return simulatedSpikeTrain, xs;

########################
class SpikeTrain():
    FILE_EXTENSION = '.st'
    def __init__(self, spike_ts, params):
        self._params = params; # this is an instance of OUParams.
        self._spike_ts = spike_ts;
        self._intervals = diff(r_[(.0,spike_ts)]);
        self._uIs, self._SDF = self._generateSDF();
        
    def getNumSamples(self):
        return len(self._spike_ts)
    
    def getTf(self):
        return self._spike_ts[-1]

    def getMaxInterval(self):
        return max(self._intervals)
    
    def getEmpiricalMoment(self, n=1):
        return sum(self._intervals**n) / len(self._intervals)
    
    def getSDF(self):
        return self._uIs, self._SDF
    
    def _generateSDF(self):
        Is = self._intervals; N = len(Is)
        unique_Is = r_[(.00, unique(Is))]
        
        #populate the empirical survivor function
        SDF = ones_like(unique_Is)
        for (Ik, idx) in zip(unique_Is, arange(len(SDF))):
            SDF[idx] = sum(Is > Ik) / N;
            
        assert (1.0 == SDF[0])
        
        return unique_Is, SDF
        
    
        
    def save(self, file_name):
        import cPickle
        dump_file = open(file_name + SpikeTrain.FILE_EXTENSION, 'wb')
        cPickle.dump(self, dump_file, 1) # 1: bin storage
        dump_file.close()
        
    @staticmethod
    def load(file_name, dir_name = RESULTS_DIR):
        file_name = os.path.join(dir_name, file_name)
        import cPickle
        load_file = open(file_name+ SpikeTrain.FILE_EXTENSION, 'r')
        path = cPickle.load(load_file)        
        return path

########################
class OUParams():
    def __init__(self, alpha, tauchar, beta, x0=.0, x_thresh=1.0):
            self._alpha = alpha;
            self._tauchar = tauchar;
            self._beta = beta;
            self._x0 = x0
            self._x_thresh = x_thresh
            
    def getParams(self):
        return self._alpha, self._tauchar, self._beta, self._x0, self._x_thresh
    def getATcB(self):
        return self._alpha, self._tauchar, self._beta,
    
    def _generatePathTag(self):
        alpha, tauchar, beta, x0, x_thresh = self.getParams()
        return OUParams.generatePathTag(alpha, tauchar, beta, x0);
    
    @staticmethod
    def generatePathTag(alpha, tauchar, beta, x0=.0, x_thresh = 1.0):
        path_tag = '_a=%.2f_t=%.1f_b=%.2f' %(alpha,tauchar, beta)
        if .0 != x0:
            path_tag += '_x0=%.2f'%x0;
        if 1.0 != x_thresh:
            path_tag += '_xth=%.2f'%x_thresh;
        
        return path_tag
                               
              
def simulate_spike_train(N_spikes = 128, save_path=False, path_tag = '',
                                        save_fig=False, fig_tag  = '',
                                        params = [1.,2.0,.5], x0 = .0, #x_thresh = 1.0,
                                        dt = 1e-4):
    
    alpha,tauchar, beta = params[0],params[1], params[2] ;
    print 'Parameterizing with: ', alpha, tauchar, beta 

    # The actual work:
    S = OUSimulator(alpha, tauchar, beta, x0);
    T, xs = S.simulate(N_spikes, dt);
                  
    print 'Spike count = ', N_spikes
    print 'Simulation time = ' , T.getTf()
    print 'Max Interval = ', T.getMaxInterval()
    
    #Post-processing:
    if save_fig or '' != fig_tag:
        figure()
        ts = arange(len(xs))*dt;
        subplot(211); hold(True)
        plot(ts, xs, 'g', linewidth=1)
        stem(T._spike_ts, ones_like(T._spike_ts), 'k')
        xlabel('$t$', fontsize = 24)
        ylabel('$X_t$', fontsize = 24)
        
        subplot(212)
        stem(T._spike_ts, ones_like(T._spike_ts), 'k')
        xlim( (0., T.getTf()))
        xlabel('$t_{sp}$', fontsize = 24)
        
        if save_fig:
            get_current_fig_manager().window.showMaximized()
            filename = os.path.join(FIGS_DIR, 'sinusoidal_train_N=%d_%s.png'%(N_spikes, fig_tag))
            print 'Saving fig to: ', filename
            savefig(filename) 
        
    if save_path:
        path_tag = T._params._generatePathTag() + '_' +path_tag;
        filename = os.path.join(RESULTS_DIR, 'spike_train_N=%d%s'%(N_spikes, path_tag))  
        print 'Saving path to ' , filename    
        T.save(filename)

def batch_simulator_x0s():
    N_spikes = 512

    x0s = linspace(-.5, .99, 7)
    alphas = [1., 1.5]
    betas = [.1, 1.0]; 
    
    for x0 in x0s:
        for alpha in alphas:
            for beta in betas:
                simulate_spike_train(N_spikes, save_path=True, path_tag='', params=[alpha, 1.0 ,beta], x0= x0, dt=1e-4)
                
    
    
if __name__ == '__main__':
    from pylab import *
    
    simulate_spike_train(N_spikes = 10, save_path=False,
                         save_fig = True, fig_tag = 'example',                                        
                                        params = [1.9,.4,.4], dt = 1e-2)
   
    
#    simulate_spike_train(N_spikes = 1000, save_path=True, path_tag = '1',                                        
#                                        params = [2.,.5,.5])
#    batch_simulator_x0s()    
    
    show()