# -*- coding:utf-8 -*-
"""
@author: alex
"""
from __future__ import division

from numpy import *
from numpy.random import randn, seed
from scipy.sparse import spdiags, lil_matrix
from scipy.sparse.linalg.dsolve.linsolve import spsolve
from copy import deepcopy
from scipy.optimize.zeros import brentq
from scipy.interpolate.interpolate import interp2d
from scipy.interpolate.fitpack2 import RectBivariateSpline

from matplotlib.pyplot import figure, subplot, ylabel, vlines, ylim, xlim, xlabel, title, plot,\
subplots_adjust, Axes, setp, legend, hlines

from pylab import hold
#My solvers:
from HJB_TerminalConditioner import calculateTs_Kolmogorov_BVP
from AdjointSolver import FBKSolution, deterministicControlHarness,\
    calculateOptimalControl, FPAdjointSolver
from HJBSolver import HJBSolver, HJBSolution 
energy_eps = .001;

RESULTS_DIR = '/home/alex/Workspaces/Python/OptSpike/Results/MLTrainController/'
FIGS_DIR    = '/home/alex/Workspaces/Latex/OptSpike/Figs/MLTrainController'

from matplotlib.patches import FancyArrowPatch, FancyArrow, ArrowStyle
from matplotlib.font_manager import FontProperties
label_font_size = 32
xlabel_font_size = 40

import os
for D in [FIGS_DIR, RESULTS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)
import time

label_font_size = 24
xlabel_font_size = 32
###########################################

from TrainController import SpikeTrain
from MLBox import MLAnnotatedSolution, MLSolution, MLParameters,\
    generateOUSpikesOnlyFromMLAnnSoln

def generateTargetTrainFromMLTrajectory(path_tag = 'ML',
                                        regime_tag='1', Tf = 2000.0,
                                        visualize = False):
    
    file_name = 'Basic_Example_Type%s_%.1f'%(regime_tag,Tf)
    
    MLAnnSoln = MLAnnotatedSolution.load(file_name)
    
    spikes = MLAnnSoln.spike_times;

    if visualize:
        figure()
        plot(spikes, zeros_like(spikes), 'b.'); 
        ylim(-.25,.25); xlim(-.25, spikes[-1] + .25)

    N_spikes = len(spikes)
    save_file_name = 'target_%s%s_%d'%(path_tag,
                                              regime_tag,
                                              N_spikes)
    SpikeTrain(spikes).save(save_file_name);
    return save_file_name;

class MLControlledSimulation():
    _FILE_EXTENSION = '.mlcs'
    def __init__(self, 
                 MLparams,
                 OUparams,
                 targetSpikeTrain,
                 controlGenerator,
                 refractory_time = 40.,
                 V_thresh = -14.0,
                 X_0 =  array([-26.0, 0.2]), 
                 dt = 1e-1):
        self.MLParams = MLparams;
        self.OUParams = OUparams;
        self.targetSpikeTrain = targetSpikeTrain;
        self.controlGenerator = controlGenerator;
        from MLBox import MLSimulator
        '''Borrow from existing ML Simulator;;: '''
        self._InnerSimulator = MLSimulator(MLparams);
        
        #needed to determine spikes:
        self.refractory_time = refractory_time
        self.V_thresh = V_thresh
        
        #Simulation parameters:
        self.dt = dt;
        self.X_0 = X_0;
        ###The Solutions List:
        self.SolnsList = []
         
    #TODO: Add analysis / visualize methods:
    def addSolution(self, MLSoln):
        self.SolnsList.append(MLSoln);
                 
    def save(self, file_name):
        print 'saving MLControlledStudy to ', file_name
        file_name = os.path.join(RESULTS_DIR,
                                 file_name + self._FILE_EXTENSION)
        import cPickle
        dump_file = open(file_name, 'wb')
        cPickle.dump(self, dump_file, 1) # 1: bin storage
        dump_file.close()
    
    @classmethod
    def load(cls, file_name):
        print 'loading ', file_name
        file_name = os.path.join(RESULTS_DIR,
                                 file_name + cls._FILE_EXTENSION) 
        import cPickle
        load_file = open(file_name, 'r')
        soln = cPickle.load(load_file)        
        return soln 

###########################################
###########################################
###########################################
    
    def runBatch(self,
                  N_samples):
        for n in xrange(N_samples):
            MLSoln = self.runSingleTrain()
            self.addSolution(MLSoln);
    
    def runSingleTrain(self):
        '''Runs an entire tracking train'''
        target_spikes = list(self.targetSpikeTrain._spike_times)
        
        last_spike = 0;
        VWs = array([self.X_0]);
        As =  array([]);
        
        for target_spike in target_spikes:
            S_target = target_spike - last_spike;
            
            currentVWs, currentAs, currentIsi = self.runSingleISI(S_target,
                                                 VWs[-1,:])
            'store solution'
            VWs = r_[VWs,
                     currentVWs];
            'store controls'
            As = r_[As,
                    currentAs];
            'increment time'
            last_spike += currentIsi;
            
        
        '''Assume the very last applied control is 0;
         this is just to even the dimensions of all stored vectors'''
        As = r_[As,
                [.0]];
        ss = linspace(.0, self.dt*(len(VWs)-1), len(VWs));
        SingleTrainSoln = MLSolution(self.MLParams,
                                     ss, VWs, None, As)
        return SingleTrainSoln; 
    
    def runSingleISI(self,
                     S_target,
                     v_w):
        '''Runs a single ISI
        Returns the V-W Trajectory
                the applied control A(t)
                the elapsed time interval of the ISI (reset-to-reset)'''
        nondim_params = self.OUParams.getMuTauBeta();
        T_target = self.OUParams.nondimensionalizeTime(S_target)
        controlFunction = self.controlGenerator(nondim_params,
                                                 T_target);
        def timeStep(v_w, s, VWs, As, dWsList):
            if 0 == len(dWsList):
                dWsList.extend( self.get_dWs(sqrt(self.dt),
                                             N = 1000));
            dWs = dWsList.pop();
#            dWs = randn(2);
            
            '''compute drift / vol'''
            f = self._InnerSimulator.MLdrift(v_w);
            g = self._InnerSimulator.MLvolatility (v_w);
            
            '''Compute Control:'''
            t,x = self.OUParams.nondimensionalizeState(s, v_w[0])
            x = amin([1.0, x]);
            alpha = controlFunction(t,x)
            A = self.OUParams.dimensionalizeControl(alpha)
            
            '''Add control to drift function'''
            f[0] += 1/self.MLParams.C * A
            
            '''return incremented state + control'''
            v_w_new = v_w + f*self.dt + g*dWs
            VWs.append(v_w_new);
            As.append(A);
            s += self.dt;
#            print s, v_w_new, A;
            return (v_w_new[0] > self.V_thresh), v_w_new, s
                                                 
        s = 0;
        VWs = []; 
        As = [];
        dWsList = [];
        '''The time stepper function:'''                                        
        start =  time.clock();
        k=0
        spiked = False;
        'The Controlled Segment'
        while not spiked:
            spiked, v_w,s = timeStep(v_w, s, VWs, As, dWsList );
            k +=1
            
                             
        print 'SPIKING!!! at %.1f / %.1f'%(s, S_target)
        
        'The Refractory Segment'                                    
        controlFunction = lambda t,x: .0    
        for s in arange(s, s+self.refractory_time, self.dt):
            spiked, v_w, _s = timeStep(v_w, s, VWs, As, dWsList);
            k +=1
        
        end =  time.clock();
        print 'spike_iterates = %d, ISI_compute_time = %.2f'%(k,
                                                              end-start);
        return VWs, As, s
            
            
    def get_dWs(self,
                sqrt_dt,
                N=240000):
        return list( randn( N , 2)* sqrt_dt);
    
    #Analysis / Visualization Methods:
    def visualize(self, figs_tag = None,
                  N_plot_samples = None):
        N_samples = len(self.SolnsList);
        if None == N_plot_samples:
            N_plot_samples = N_samples
            
        target_spikes = self.targetSpikeTrain._spike_times;
        MLSoln = self.SolnsList[0]
        ts, Vs, Ws = MLSoln.get_ts_Vs_Ws()
        As = MLSoln._alphas
        
        y_up = 40.0 #amax(Vs);
        
        'Visualize example trajectory:'
        sub_sample = 10;
        exampleFig = figure()
        ax = subplot(311); 
        plot(ts[::sub_sample], Vs[::sub_sample]);
        ylabel('$V_s$', fontsize = xlabel_font_size)
        vlines(target_spikes, .0, y_up, colors='r', linestyles='dashed')
        
        subplot(312)
        plot(ts[::sub_sample], As[::sub_sample]);
        ylabel('$A(s)$', fontsize = xlabel_font_size)
        ylim((self.OUParams.A_min,
              self.OUParams.A_max));
        
        subplot(313); plot(ts[::10], Ws[::sub_sample] )
        ylabel('$W_s$',   fontsize = xlabel_font_size)
        xlabel('s', fontsize = xlabel_font_size)
        
        '''Visualize raster spikes Plot'''
        from TrainController import makeRasterPlot,\
             windowSmoothTargetTrain,\
             generateProbabilityOfFire
             
        TargetTrainsList = []
        for idx, MLSoln in enumerate(self.SolnsList):
            MLSoln._dBs = zeros_like(MLSoln._Xs[:-1,:])
            MLAnnSoln = MLAnnotatedSolution(MLSoln, 
                                            refractory_time=self.refractory_time,
                                            v_thresh=self.V_thresh)
            TargetTrainsList.append(MLAnnSoln.spike_times);
        '''raster plot it:'''
        rasterFig = makeRasterPlot(target_spikes,
                                       TargetTrainsList, 
                                       N_plot_samples,
                                       t_max=2e3,
                                       sigma_window = .1*100,
                                       window_delta = .1*100,
                                       time_symbol = 's') 
#        print figs_tag +
        N_sims = len(TargetTrainsList);
        N_spikes = len(target_spikes);
        tracking_spikes = array(TargetTrainsList).reshape((N_sims, N_spikes));
        time_errors = tracking_spikes - tile(target_spikes, (N_sims,1));
        rms_error = sqrt(mean(time_errors * time_errors));
        mean_error = mean(abs(time_errors));
        mean_interval = mean(diff(r_[0, target_spikes]));
        percent_rms_error = 100* rms_error / mean_interval;
        print '%s: rms error = %.2f ,\
         relative rms error =  %.2f;\
                  mean interval = %.2f'%(figs_tag,
                                      rms_error,
                                       percent_rms_error,
                                        mean_interval)

        if None != figs_tag:
            for label, fig in zip(['example', 'raster'],
                                      [exampleFig, rasterFig]):
                file_name = os.path.join(FIGS_DIR,
                                         'MLSim_%s_%s.pdf'%(label,
                                                            figs_tag))
                print file_name
                fig.savefig(file_name);


from MLBox import estimateOUfromML, OUParameters
def GenerateHarness():
    st_file_name = generateTargetTrainFromMLTrajectory(visualize=False);
    ST = SpikeTrain.load(st_file_name)
    
    file_name = 'Basic_Example_Type%s_%.1f'%('1', 2e3)
    MLAnnSoln = MLAnnotatedSolution.load(file_name)
    OU_estimates = estimateOUfromML(MLAnnSoln,
                                    visualize = False);
    meanISI = MLAnnSoln.getMeanISI();
    m, C, sigma = OU_estimates[:]
    A_bounds = [-20, 20]
    OUParams = OUParameters(MLAnnSoln.v_OU_bound,
                            MLAnnSoln.getYreset(),
                            m, C, sigma,
                            meanISI, A_bounds[0], A_bounds[1])                               

    print r'\hat T ' , meanISI
    
    original_target_spikes = ST._spike_times;
    nondimmed_target_spikes = original_target_spikes / OUParams.meanISI
    
    subplot(211);
    stem(original_target_spikes, ones_like(original_target_spikes))
    subplot(212);
    stem(nondimmed_target_spikes, ones_like(original_target_spikes))
    
from ControlGenerator import CLControlGenerator, ControlGenerator,\
    OLControlGenerator

from MLBox import generateOUFromMLAnnSoln
def SimulationHarness(OUparams, MLparams,
                      ST_target, controlGenerator,
                      simulate = True,
                      reload = True,
                      N_sampled_trajectories=1,
                      regime_tag = '1',
                      Tf = 2e3,
                      dt = 2e-1,
                      control_tag = 'cl',
                      figs_tag='cl'):
    '''If simulate and NOT reload:
         the old results will be overwritten with a new simulation
    If simulate and reload:
         the old results will be augmented'''
    
    sim_file_name = 'ML%sTrackingSim_%s'%(regime_tag,
                                          control_tag)
    if simulate:
        Sim = None;
        if reload:
            try:
                Sim = MLControlledSimulation.load(sim_file_name);
            except IOError as ioer: #No File found!!!
                print ioer
                reload = False
        
        if not reload:
            Sim = MLControlledSimulation(MLparams, OUparams, 
                                         ST_target, 
                                         controlGenerator,
                                         dt = dt)
        'Run:'
        start= time.clock()
        Sim.runBatch(N_sampled_trajectories)
        print 'Simulating %d %s::trajectories takes %.2f s'%(N_sampled_trajectories,
                                                           control_tag,
                                                           time.clock() - start);
        'Save:'
        Sim.save(sim_file_name)

    'Load:'
    Sim = MLControlledSimulation.load(sim_file_name);
    print 'N= ', len(Sim.SolnsList);
    
    Sim.visualize(figs_tag=figs_tag)


def TrajectoriesAvailableHarness(simulate = True,
                 reload = True,
                 regime_tag = '1',
                 Tf = 2e4,
                 A_max=10.0,
                 N_sampled_trajectories=1):
    '''If simulate and NOT reload:
         the old results will be overwritten with a new simulation
       If simulate and reload:
         the old results will be augmented with the new simulations'''
     
    file_name = 'Basic_Example_Type%s_%.1f'%(regime_tag,Tf)
    A_bounds = array([-A_max, A_max]);
    OUparams, MLparams  = generateOUFromMLAnnSoln(file_name,
                                                  A_bounds)
    
    m_C_sigma = OUparams.getMCSigma();
    print m_C_sigma;
    
    mu_tau_beta_params = OUparams.getMuTauBeta();
    print mu_tau_beta_params;
    
    alpha_bounds = OUparams.getAlphaBounds();
    print alpha_bounds;
    
    st_file_name = 'target_ML1_9' #%regime_tag
    ST_target = SpikeTrain.load(st_file_name)
#        ST_target._spike_times = ST_target._spike_times[:2]
    ISIs_target = diff(r_[.0, ST_target._spike_times]);
    print ISIs_target   
    
    max_ISI = amax(ISIs_target)
    
    max_Tf_2calculate = 3 * OUparams.nondimensionalizeTime(max_ISI)
        
    clGenerator = CLControlGenerator(mu_tau_beta_params,
                                      alpha_bounds=alpha_bounds,
                                      energy_eps = energy_eps,
                                      max_Tf=max_Tf_2calculate)
    olGenerator = OLControlGenerator(mu_tau_beta_params,
                                      alpha_bounds=alpha_bounds,
                                       energy_eps= energy_eps)
    

#    for controlGenerator, control_tag in zip([clGenerator, olGenerator],
#                                             ['cl', 'ol']):
    for controlGenerator, control_tag in zip([clGenerator],
                                             ['cl']):
#    for controlGenerator, control_tag in zip([olGenerator],
#                                             ['ol']):
        control_tag = '%s_Amax%d'%(control_tag,
                                   A_max)
        SimulationHarness(OUparams, MLparams,
                           ST_target, controlGenerator,
                            simulate=simulate,
                            reload=reload,
                             N_sampled_trajectories=N_sampled_trajectories,
                              regime_tag=regime_tag,
                               Tf=Tf,
                                control_tag=control_tag,
                                figs_tag = control_tag)
        

def SpikesOnlyHarness(simulate = True,
                      reload = True,
                      regime_tag = '1',
                      Tf = 2e4,
                      A_max=10.,
                      N_sampled_trajectories=1):
    '''If simulate and NOT reload:
         the old results will be overwritten with a new simulation
    If simulate and reload:
         the old results will be augmented'''
    
    file_name = 'Basic_Example_Type%s_%.1f'%(regime_tag,Tf)
    A_bounds = array([-A_max, A_max]);
    OUparams, MLparams  = generateOUSpikesOnlyFromMLAnnSoln(file_name,
                                                            A_bounds,
                                                            C_hat = 20.)
    
    print OUparams.getMCSigma();
    
    mu_tau_beta_params = OUparams.getMuTauBeta();
    print mu_tau_beta_params;

    alpha_bounds = OUparams.getAlphaBounds();
    print alpha_bounds;
    
    st_file_name = 'target_ML1_9' #%regime_tag
    ST_target = SpikeTrain.load(st_file_name)
    
    ISIs_target = diff(r_[.0, ST_target._spike_times]);
    print ISIs_target   
    
    olGenerator = OLControlGenerator(mu_tau_beta_params,
                                      alpha_bounds=alpha_bounds,
                                       energy_eps= energy_eps)
    control_tag = 'ol'
    control_tag = '%s_Amax%d_spikesonly_C%d'%(control_tag,
                                              A_max,
                                              OUparams.C);

    SimulationHarness(OUparams, MLparams,
                       ST_target, olGenerator,
                        simulate=simulate,
                        reload=reload,
                         N_sampled_trajectories=N_sampled_trajectories,
                          regime_tag=regime_tag,
                           Tf=Tf,
                            control_tag=control_tag,
                            figs_tag = control_tag)
    
###########################################################################
###########################################################################
    
def TimerHarness():
        regime_tag = '1'
        Tf = 2e3;
        file_name = 'Basic_Example_Type%s_%.1f'%(regime_tag,Tf)
        OUparams, MLparams  = generateOUFromMLAnnSoln(file_name);
        mu_tau_beta_params = OUparams.getMuTauBeta();
        alpha_bounds = OUparams.getAlphaBounds();
        
        st_file_name = 'target_ML1_9' #%regime_tag
        ST_target = SpikeTrain.load(st_file_name)
#        ST_target._spike_times = ST_target._spike_times[:]
        print ST_target._spike_times 
        ISIs_target = diff(r_[.0, ST_target._spike_times]);
        max_ISI = amax(ISIs_target)
        max_Tf_2calculate = 1.5 * OUparams.nondimensionalizeTime(max_ISI)
        
        start = time.clock()
        controlGenerator = CLControlGenerator(mu_tau_beta_params,
                                          alpha_bounds=alpha_bounds,
                                          energy_eps = energy_eps,
                                          max_Tf=max_Tf_2calculate*2);
        mid = time.clock()
        print 'Control Compute time = :',    mid-start;
        
        Sim = MLControlledSimulation(MLparams, OUparams, 
                                     ST_target, 
                                     controlGenerator,
                                     dt = 2e-1)
        
        Sim.runSingleTrain()
        end = time.clock()
        print 'Simulate time = :',    end-mid
    
#TIME IT:
def ProfileHarness():
    import cProfile, pstats, StringIO
    pr = cProfile.Profile()
    pr.enable()
    # ... do something ...
    TimerHarness()
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()
    
             
if __name__ == '__main__':
    from pylab import *   

##### ### ### ### ### ##### 
## GENERATE TARGET:
##### ### ### ### ### #####
#    GenerateHarness()

#################        
##### SIMULATE:
#################
    TrajectoriesAvailableHarness(simulate=False,     
                 reload =True,            
                 N_sampled_trajectories=1)
#    
    SpikesOnlyHarness(simulate=False,
                      reload = True,
                      N_sampled_trajectories=7)


#################        
##### Time/Profile:
#################
#    TimerHarness();
#    ProfileHarness();
#    print 'Not Showing'
    show()
    