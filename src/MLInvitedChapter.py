'''
Created on Jul 4, 2015

@author: alex
'''

from MLBox import *
from ML_TrainController import *


def IsweepBox(resimulate=False,
               Tf = 2000., dt = 1e-2,refractory_time = 40.,A_bounds=[-10,10]):
    '''We try to find values of I for ML-regime 1 which correspond to a quiescent (non-0firing
    cell (Sub-thereshold) and to spontaneous firing (Supra-Threshold)
    
    The final-decision is to use 
        I=38(Sub-Threshold)
        I=52(Super-Threshold)
    '''
    
    regime_tag = '1';
    seed(2014)
    ML_params = deepcopy(MLParametersDict[regime_tag])
    for I, lif_regime_tag in zip( [38, 48], ['SubT', 'SupraT']):
#    for I, lif_regime_tag in zip( [38 ], ['SubT' ]):
        
        file_name =  'Isweep_ML1_I=%d_%s'%(I, lif_regime_tag);
        ML_params.I = I
#        ML_params.sigma = 0;
#        ML_params.gamma = 0;
        
        if resimulate:
            MLSim = MLSimulator(ML_params);
            ts, Xs, dBs = MLSim.simulate(Tf=Tf, dt = dt,
                                     visualize=False)
            (MLSolution(ML_params, ts, Xs, dBs,
                     zeros_like(ts))).save(file_name)
    
        MLSoln = MLSolution.load(file_name)
        print 'Annotating'
        MLAnnSoln = MLAnnotatedSolution(MLSoln,
                                    refractory_time=refractory_time)
        MLAnnSoln.save(file_name)
        
        'Ensure that we can generate an OU from the solution'
        OUparams, MLparams  = generateOUFromMLAnnSoln(file_name, A_bounds)
        m_C_sigma = OUparams.getMCSigma();
        print 'mCs = ', m_C_sigma;
        
        mu_tau_beta_params = OUparams.getMuTauBeta();
        print 'mtb = ', mu_tau_beta_params;
        
        ts, Vs, Ws = MLSoln.get_ts_Vs_Ws()
        
        'Visualize:'
        sub_sample = 10;
        figure()
        subplot(211); 
        plot(ts[::sub_sample], Vs[::sub_sample]);
        ylabel('$V_s$', fontsize = xlabel_font_size)
        title('I=%d'%I)
        subplot(212); plot(ts[::10], Ws[::sub_sample] )
        ylabel('$W_s$',   fontsize = xlabel_font_size)
        xlabel('s', fontsize = xlabel_font_size)
        
        ss, Vs, _Ws = MLAnnSoln.MLSoln.get_ts_Vs_Ws();
        v_thresh = MLAnnSoln.v_thresh;
        v_OU_bound = MLAnnSoln.v_OU_bound;
        
        spike_times = MLAnnSoln.spike_times;
        reset_times = MLAnnSoln.reset_times;
        interval_ss = MLAnnSoln.interval_ss;
        interval_Vs = MLAnnSoln.interval_Vs;
        diffusion_ss = MLAnnSoln.diffusion_ss;
        diffusion_Vs = MLAnnSoln.diffusion_Vs;
        
        #        
        'Visualize:'
        figure(figsize = (17,12))
        subplots_adjust(hspace = .25)
        subplot(311); 
        subsample = 20;
        plot(ss[::subsample], Vs[::subsample]);
        ylabel('$V_s$',
                fontsize = xlabel_font_size); 
        hlines(v_OU_bound, ss[0], ss[-1], 
                colors='g', linestyles='dashed');
        hlines(v_thresh, ss[0], ss[-1], 
                colors='r', linestyles='dashed');
        up_val = amax(Vs) + 5.0;
        down_val = amin(Vs) -5.0;
        vlines(spike_times,  v_thresh, up_val, colors='r', linewidth =2)
        vlines(reset_times,  v_thresh, down_val, colors='g', linewidth = 2)
        ylim((-60,40))
        
        subplot(312);
        for ss, Vs in zip(diffusion_ss,
                          diffusion_Vs):
            plot(ss[::subsample], Vs[::subsample], 'b') 
        ylabel('$V_s$',
                fontsize = xlabel_font_size);
        xlabel('$s$',
                fontsize = xlabel_font_size);
        
        refractory_indx_interval = where(MLAnnSoln.MLSoln._ts == MLAnnSoln.refractory_time)[0][0]
        subplot(313)
        subsample = 4
        for ss, Vs in zip(interval_ss[:-1],
                          interval_Vs[:-1]):
            lss = ss[-refractory_indx_interval::subsample]
            Vs  = Vs[-refractory_indx_interval::subsample]
            plot(lss - lss[-1], Vs, 'g') 
        ylabel('$V_s$',
                fontsize = xlabel_font_size);
        xlabel('$s$',
                fontsize = xlabel_font_size);
        title('refractory time = %.2f ms'%refractory_time)
        
        
        
def SpikeTargetSimulateHarnessSubTSuperT(simulate = False,
                    reload = True,
                    A_max=10.0,
                    N_sampled_trajectories=1):
    '''If simulate and NOT reload:
         the old results will be overwritten with a new simulation
       If simulate and reload:
         the old results will be augmented with the new simulations'''
     
    A_bounds = array([-A_max, A_max]);
     
    'Sweep'
    for I, lif_regime_tag in zip( [38, 48], ['SubT', 'SupraT']):
        
        file_name =  'Isweep_ML1_I=%d_%s'%(I, lif_regime_tag);
        OUparams, MLparams  = generateOUFromMLAnnSoln(file_name, A_bounds)
        
        m_C_sigma = OUparams.getMCSigma();
        print 'mCs = ', m_C_sigma;
        
        mu_tau_beta_params = OUparams.getMuTauBeta();
        print 'mtb = ', mu_tau_beta_params;
        
        alpha_bounds = OUparams.getAlphaBounds();
        print 'a-, a+ =',alpha_bounds;
        
        st_file_name = 'target_ML1_9' #%regime_tag
        ST_target = SpikeTrain.load(st_file_name)
    #        ST_target._spike_times = ST_target._spike_times[:2]
    
        ISIs_target = diff(r_[.0, ST_target._spike_times]);
        print [0, ST_target._spike_times[-1]]
        print ISIs_target   
        
        'Define Control'
        max_ISI = amax(ISIs_target)
        max_Tf_2calculate = 3 * OUparams.nondimensionalizeTime(max_ISI)
        control_tag = 'cl'
        print    max_ISI  ,  max_Tf_2calculate ,         control_tag
    
        'the Control Generator (parameterized by I_k'
        clGenerator = CLControlGenerator(mu_tau_beta_params,
                                          alpha_bounds=alpha_bounds,
                                          energy_eps = energy_eps,
                                          max_Tf=max_Tf_2calculate) 
    

         
        SimulationHarness(OUparams, MLparams, ST_target, clGenerator,
                            simulate=simulate,
                            reload=reload,
                             N_sampled_trajectories=N_sampled_trajectories, 
                               regime_tag=lif_regime_tag,
                                control_tag=control_tag,
                                figs_tag = 'cl_%s'%lif_regime_tag)
        

def visualize(sub_sample = 10,
              example_soln_idx=2 ):
    'Sweep'
    from ML_TrainController import RESULTS_DIR as ml_sim_dir
    
    titleDict = {'SubT': 'Sub-Threshold Regime (I=38)', 
                 'SupraT':'Supra-Threshold Regime (I=48)'}
    for I, lif_regime_tag in zip( [38, 48], ['SubT', 'SupraT']):
#    for I, lif_regime_tag in zip( [38], ['SubT' ]):
        S = MLControlledSimulation.load( os.path.join(ml_sim_dir, 'ML%sTrackingSim_cl'%lif_regime_tag))
        
        target_spikes = S.targetSpikeTrain._spike_times;
        MLSoln = S.SolnsList[example_soln_idx]
        ts, Vs, Ws = MLSoln.get_ts_Vs_Ws()
        As = MLSoln._alphas
        
        y_up = 40.0 #amax(Vs);
        
        'Visualize example trajectory:'
        figure(figsize=(17, 15))
        ax = subplot(311); 
        plot(ts[::sub_sample], Vs[::sub_sample]);
        ylabel('$V_t$', fontsize = xlabel_font_size)
        vlines(target_spikes, .0, y_up, colors='r', linestyles='dashed')
        
        'Contorl Plot'
        ax1 = subplot(312)
        plot(ts[::sub_sample], As[::sub_sample]);
        ylabel('$A(s)$', fontsize = xlabel_font_size)
        ylim((S.OUParams.A_min,
              S.OUParams.A_max));
        ax.set_yticks([-40, 0, 40])
        ax1.set_yticks([-10, 0 ,10])
        for axx in [ax, ax1]:
            setp(axx.get_xticklabels(), visible=False)
            for label in axx.yaxis.get_majorticklabels():
                label.set_fontsize(label_font_size)
        
        'Raster Plot'
        ax2 = subplot(313); 
        
        TargetTrainsList = []
        for idx, MLSoln in enumerate(S.SolnsList):
            MLSoln._dBs = zeros_like(MLSoln._Xs[:-1,:])
            MLAnnSoln = MLAnnotatedSolution(MLSoln, 
                                            refractory_time=S.refractory_time,
                                            v_thresh=S.V_thresh)
            TargetTrainsList.append(MLAnnSoln.spike_times);
        N_plot_samples = len(TargetTrainsList)
            
        '''raster plot it:'''
        plot(target_spikes, ones_like(target_spikes), 'r.');
    
        ax2.vlines(target_spikes, -N_plot_samples-1., 2, colors='r', linestyles='dashed')
        for idx in xrange(N_plot_samples):
           generated_spikes = TargetTrainsList[idx]
           plot(generated_spikes,
                 (- 1 -idx) * ones_like(generated_spikes), 'b.')
        
        ylabel('Induced\nSpikes', fontsize =xlabel_font_size)  
        xlabel('$t$', fontsize = xlabel_font_size) 
        ax2.set_ylim(-N_plot_samples-1., 2);
        
        hlines(0, 0, ts[-1], colors='k')
    
        setp(ax2.get_yticklabels(), visible=False)
        for label in ax2.xaxis.get_majorticklabels():
            label.set_fontsize(label_font_size)
    

        save_file_name = os.path.join(FIGS_DIR, 'ML%s_ControledBatch.pdf'%lif_regime_tag)
        print 'saving to ', save_file_name
        savefig(save_file_name, dpi=300)
         
if __name__ == '__main__':
    from pylab import *
 
#    'Find the ML Parameters to use Train'
#    IsweepBox(resimulate=True, Tf=10000)
#    
#    'Generate the tracked-target trains (controled realizations)'
#    SpikeTargetSimulateHarnessSubTSuperT(simulate=True,
#                                          reload = False ,
#                                          N_sampled_trajectories=20 )
    
    'Generate Figures'
    visualize();
    
    show()
            