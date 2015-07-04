"""
Created on Mar 13, 2012

@author: alex
"""
from __future__ import division
from copy import deepcopy
from numpy import *
import numpy as np
from numpy.random import randn, rand
from scipy.interpolate.interpolate import interp1d
from HJBSolver import xlabel_font_size

RESULTS_DIR = '/home/alex/Workspaces/Python/OptSpike/Results/MLBox/'
FIGS_DIR = '/home/alex/Workspaces/Latex/OptSpike/Figs/MLBox/'
import os
for D in [RESULTS_DIR, FIGS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)

def get_ts_dWs(dt, tf):
    #Obtain the increments of a Wiener Process realization
    #returns: ts - the sampled time points of the realization
    #         dWs - the forward incrments of W_t at ts[:-1]
    #the square root of dt:
    sqrt_dt = sqrt(dt)
    #the time points:
    ts = r_[arange(.0, tf, dt), tf];
    #the Gaussian incrments:
    dWs = randn(len(ts)-1, 2)* sqrt_dt
    
    return ts, dWs

class MLParameters():
    ''' All the parameters of the ML Process:
    The default values are from Sussane's paper (CLASS II):'''
    def __init__(self,
                 regime_tag,
                 VK, VL, VCa, C, 
                 gL, gCa, gK,
                 V1, V2, V3, V4,
                 phi, I, sigma, gamma):
        #psi = c(VK, VL, VCa, C);
        #theta = c(gL, gCa, gK, V1, V2, V3, V4, sigma, gamma,VK, VL, phi, C, VCa, I)
        self.regime_tag = regime_tag;
        
        self.VK =VK 
        self.VL =VL 
        self.VCa =VCa 
        self.C =C 
        self.gL =gL 
        self.gCa =gCa 
        self.gK =gK 
        self.V1 =V1 
        self.V2 =V2 
        self.V3 =V3 
        self.V4 =V4 
        self.phi =phi 
        self.I =I 
        self.sigma =sigma 
        self.gamma =gamma 
        
    def printme(self, file_name=None):
        diag_str = 'ML::Type%s:\n'%self.regime_tag
        if None == file_name:
            diag_str += 'VK =%.2f '%self.VK  +\
                'VL =%.2f '%self.VL  +\
                'VCa =%.2f '%self.VCa  +\
                'C = %.2f \n'%self.C +\
                'gL =%.2f '%self.gL  +\
                'gCa =%.2f '%self.gCa  +\
                'gK =%.2f \n'%self.gK  +\
                'V1 =%.2f '%self.V1  +\
                'V2 =%.2f '%self.V2  +\
                'V3 =%.2f '%self.V3  +\
                'V4 =%.2f \n'%self.V4  +\
                'phi =%.2f '%self.phi  +\
                'I =%.2f \n'%self.I  +\
                'sigma =%.2f '%self.sigma  +\
                'gamma =%.2f '%self.gamma
            print diag_str
        else:
            diag_str += r'\\ VK =%.2f '%self.VK  +\
                'VL =%.2f '%self.VL  +\
                'VCa =%.2f '%self.VCa  +\
                r'C = %.2f \\'%self.C +\
                'gL =%.2f '%self.gL  +\
                'gCa =%.2f '%self.gCa  +\
                r'gK =%.2f \\'%self.gK  +\
                'V1 =%.2f '%self.V1  +\
                'V2 =%.2f '%self.V2  +\
                'V3 =%.2f '%self.V3  +\
                r'V4 =%.2f \\'%self.V4  +\
                'phi =%.2f '%self.phi  +\
                r'I =%.2f \\'%self.I  +\
                'sigma =%.2f '%self.sigma  +\
                r'gamma =%.2f \\'%self.gamma
            
            printFile = open(file_name, 'w');
            printFile.write(diag_str);
            print 'saving to params to ', printFile.name
            printFile.close()



MLType1_params = MLParameters(regime_tag = '1', VK = -84, VL = -60, VCa = 120, C = 20,
                  gL = 2, gCa = 4, gK = 8,
                  V1 = -1.2, V2 = 18, V3 = 12, V4 = 17.4,
                  phi = 0.067, I = 40, sigma = 0.2, gamma = 1)

MLType2_params = MLParameters(regime_tag = '2', VK = -84, VL = -60, VCa = 120, C = 20, 
                 gL = 2, gCa = 4.4, gK = 8,
                 V1 = -1.2, V2 = 18, V3 = 2, V4 = 30,
                 phi = 0.04, I = 90, sigma = 0.03, gamma = 1);
                 
MLParametersDict = {'1': MLType1_params,
                    '2': MLType2_params}


class OUParameters():
    def __init__(self, y_thresh, yreset,
                       m, C, sigma, 
                       meanISI, A_min, A_max):
        'Scaling constants:'
        self.y_thresh = y_thresh;
        self.yreset     = yreset;
        
        'Dimensional dynamic constants:'
        self.m = m;
        self.C = C;
        self.sigma = sigma;
        
        'Time constant:'
        self.meanISI = meanISI;
        
        'amin, amax'
        self.A_min = A_min
        self.A_max = A_max;
        
        'NOn-dimensional dynamic constants'
        self.mu = self.meanISI * (self.m - self.yreset) /\
                    ( self.C*(self.y_thresh - self.yreset));
        self.tau_char = C / self.meanISI;
        self.beta = self.sigma*sqrt(self.meanISI) / \
                        (self.y_thresh-self.yreset)
    
    def dimensionalizeControl(self, alpha):
        return self.C * (self.y_thresh-self.yreset) * alpha / self.meanISI;

    def nondimensionalizeControl(self, A):
        return self.meanISI*A / \
                ( self.C * (self.y_thresh-self.yreset));
    def nondimensionalizeState(self, s, Y):
        return   s / self.meanISI, \
                (Y - self.yreset) / (self.y_thresh - self.yreset)
    
    def nondimensionalizeControlBounds(self, A_bounds):
        return self.nondimensionalizeControl(array(A_bounds));
    def dimensionalizeTime(self, ts):
        return ts * self.meanISI
    def nondimensionalizeTime(self, ss):
        return ss / self.meanISI
    
    def getAlphaBounds(self):
        return self.nondimensionalizeControl(array([self.A_min,
                                                    self.A_max]));
    def getMuTauBeta(self):
        return self.mu, self.tau_char, self.beta;
    def getMCSigma(self):
        return self.m, self.C, self.sigma;
    
    def printme(self, file_name=None):
        if None == file_name:
            diag_str = 'OU:\n' +\
            'y_thresh = %.2f '%self.y_thresh +\
            'yreset = %.2f \n'%self.yreset +\
            'meanISI = %.2f \n' %self.meanISI +\
            'm = %.2f '%self.m +\
            'C = %.2f '%self.C +\
            'sigma = %.2f \n'%self.sigma +\
            'mu = %.2f '%self.mu +\
            'tau_char = %.2f '%self.tau_char +\
            'beta = %.2f \n'%self.beta +\
            'A_min, A_max = [%.2f, %.2f] \n'%(self.A_min, self.A_max) +\
            'alpha_min, alpha_max = [%.2f, %.2f]'%(self.nondimensionalizeControl(self.A_min),
                                                   self.nondimensionalizeControl(self.A_max)) 
            print diag_str
        else:
            latex_str = '' +\
            '\yth &= %.2f &'%self.y_thresh +\
            r'\yreset &= %.2f & '%self.yreset +\
            r'\hat I  &= %.2f  \\ ' %self.meanISI +\
            'm &= %.2f &'%self.m +\
            'C &= %.2f &'%self.C +\
            r'\sigma &= %.2f \\ '%self.sigma +\
            r'\mu &= %.2f &'%self.mu +\
            r'\tc &= %.2f &'%self.tau_char +\
            r'\beta &= %.2f \\'%self.beta +\
            r'A_{min} &= %.2f & A_{max} &= %.2f && \\'%(self.A_min, self.A_max) +\
            r'\alpha_{min}&= %.2f & \alpha_{max} =&= %.2f && \\'%(self.nondimensionalizeControl(self.A_min),
                                                                self.nondimensionalizeControl(self.A_max)) 
            
            latex_file_name = os.path.join(FIGS_DIR,
                                           file_name)
            latexFile = open(latex_file_name, 'w');
            print 'writing to ', latex_file_name
            latexFile.write(latex_str);
            latexFile.close()

def generateOUFromMLAnnSoln(ML_ann_file_name,
                            A_bounds = [-20, 20]):
        MLAnnSoln = MLAnnotatedSolution.load(ML_ann_file_name)
        OU_estimates = estimateOUfromML(MLAnnSoln,
                                        visualize = False);
        meanISI = MLAnnSoln.getMeanISI();
        m, C, sigma = OU_estimates[:]
        OUParams = OUParameters(MLAnnSoln.v_OU_bound,
                                MLAnnSoln.getYreset(),
                                m, C, sigma,
                                meanISI, A_bounds[0], A_bounds[1])
        return OUParams, MLAnnSoln.MLSoln._params;
             
def generateOUSpikesOnlyFromMLAnnSoln(ML_ann_file_name,
                                      A_bounds = [-20, 20],
                                      C_hat = 20.):
        MLAnnSoln = MLAnnotatedSolution.load(ML_ann_file_name)
        OU_estimates = estimateOUfromMLSpikesOnly(MLAnnSoln,
                                                  visualize = False,
                                                  C_hat = C_hat);
        meanISI = MLAnnSoln.getMeanISI();
        m, C, sigma = OU_estimates[:]
        OUParams = OUParameters(MLAnnSoln.v_OU_bound,
                                MLAnnSoln.getYreset(),
                                m, C, sigma,
                                meanISI, A_bounds[0], A_bounds[1])
        return OUParams, MLAnnSoln.MLSoln._params;


class MLSimulator():
    def __init__(self, ML_params):
        self.params = ML_params;
        
    def simulate(self, X_0 =  array([-26.0, 0.2]), 
                 dt = 1e-2, Tf= 1000.,
                 visualize = False):
        '''simulate Euler approximation of Morris Lecar system
            # INPUT
            # X0 = (V0, W0) : initial value of states
            # dt : step size of Euler scheme
            # Tf : length of the simulation '''
    
        
        ts, dWs = get_ts_dWs(dt, Tf)
        N = len(ts);
        Xs = empty((N, 2));
        Xs[0,:] = X_0;
        
        '''Integrate SDE:'''
        self.params.printme();
        for tdx in xrange(1, N):
            x = Xs[tdx-1,:];
            '''compute drift / vol'''
            f = self.MLdrift(x);
            g = self.MLvolatility (x);
            
            '''increment'''
            Xs[tdx, :] = x + f*dt + g*dWs[tdx-1,:]; 
#            print '%.3f, %.3f'%(Xs[tdx,0],Xs[tdx,1])
#            print '%.3f, %.3f'%(f[0],f[1])
            
        if visualize:
            figure()
            subplot(211); plot(ts[::10], Xs[::10, 0])
            subplot(212); plot(ts[::10], Xs[::10, 1])
        
        return ts, Xs, dWs
    
    def alpha_beta(self, v):
        'compute alpha(v), beta(v)'
        cosh_term = .5 * self.params.phi * cosh((v - self.params.V3)/(2.*self.params.V4))
        tanh_term = tanh((v - self.params.V3)/ self.params.V4 )
        a = cosh_term * (1 + tanh_term);
        b = cosh_term * (1 - tanh_term);
        return a,b
    def m_infinity(self, v):
        '''minfty.V = 1/2*(1 + tanh( (X[1]-V1)/V2 ));'''
        return .5 * (1 + tanh((v - self.params.V1)/ self.params.V2 ))
    
    def MLdrift(self, x):
        v,w = x[:];
        '''v_drift'''
        v_drift = 1/self.params.C * (- self.params.gCa * self.m_infinity(v) * (v - self.params.VCa) \
                                     - self.params.gK * w * (v - self.params.VK) \
                                     - self.params.gL * (v - self.params.VL) \
                                     + self.params.I);
        
        '''w_drift'''
        alpha, beta = self.alpha_beta(v);
        w_drift = alpha * (1.-w) - beta * w;
    
        'return:'
        return array([v_drift,
                      w_drift])
        
    def MLvolatility(self, x):
        ''' NOTE: We always use Model 2 formula for the W volatility!!!'''
        v,w = x[:]
        'v_volatility'
        v_volatility= self.params.gamma;
        
        'w_volatility NOTE: We always use Model 2!!!'
        alpha, beta = self.alpha_beta(v);
        sqrt_arg = 2 * alpha * beta / (alpha+beta) * (1 - w) * w;
        if sqrt_arg < .0:
            print w, sqrt_arg
        w_volatility = self.params.sigma * sqrt(sqrt_arg)
        
        'return:'
        return array([v_volatility,
                      w_volatility]);
  

######################################################
import cPickle
class MLSolution():
    _FILE_EXTENSION = '.mls'
    def __init__(self, 
                 MLparams,
                 ts, Xs, dBs,
                 alphas_transformed):
        '''A Solution for the ML Model.
        Input:
             MLParams :: instance of the class MLParameters
             ts :: time points:
             Xs :: The trajectory (Vs, Ws) of the voltage, refractory variable.
             dBs :: the gaussian increments used to generate the trajectory
             alphas_transformed :: the additive control (A(t))
             '''
        self._params = MLparams

        self._ts  = ts;
        self._Xs  = Xs;
        self._dBs = dBs;
        self._alphas = alphas_transformed;

#    @classmethod
#    def _default_file_name(cls, true_params, alpha_mi):
#        return 'MLSolution_default'
    def  get_ts_Vs_Ws(self):
        return self._ts, self._Xs[:,0], self._Xs[:,1]
    
    def save(self, file_name):
#        path_data = {'path' : self}
#        if None == file_name:
#            file_name = self._default_file_name(self._true_params,
#                                                self._alphas_f[0]);
        
        file_name = os.path.join(RESULTS_DIR,
                                 file_name + self._FILE_EXTENSION)
        print 'saving ML trajectories to ', file_name
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

########################
class MLAnnotatedSolution():
    _FILE_EXTENSION = '.mla'
    def __init__(self, MLSoln,
                 refractory_time = 40.,
                  v_OU_bound = -22.0,
                  v_thresh = -14.0  ):
        
        self.MLSoln = MLSoln;
        self.refractory_time = refractory_time
        self.v_OU_bound = v_OU_bound;
        self.v_thresh = v_thresh
        
        self.OU_cross_times, self.spike_times, self.reset_times, \
        self.interval_ss, self.interval_Vs, \
        self.diffusion_ss, self.diffusion_Vs, self.diffusion_dBs = self.annotate();
        
        
#        self.spike_ts = 
    def getMeanISI(self):
        ISIs = self.getISIs()
        '''Don't include the first and last ISI, b/c they are truncated (resp. 
        in the beginning and the end'''
        return mean (ISIs[1:]); 
    def getISIs(self):
        '''Don't consider the last interval b/c chances are it has not spikes:'''
        if len(self.spike_times) == len(self.reset_times) :
            return self.spike_times - self.reset_times
        elif  len(self.spike_times) == (len(self.reset_times)-2):
            return self.spike_times - self.reset_times[:-2]
        elif  len(self.spike_times) == (len(self.reset_times)-1):
            return self.spike_times - self.reset_times[:-1]
        else:
            raise RuntimeError('''spike times and reset times
                                 have weird lengths: %d, %d'''%(len(self.spike_times) ,
                                                                len(self.reset_times)))
                                                                                              
    def getYreset(self):
        Yrs = array([Ys[-1]  for Ys in self.interval_Vs])
        return mean(Yrs)
         
    def annotate(self):
        ss, Vs, __ws = self.MLSoln.get_ts_Vs_Ws()
        dBs = r_[self.MLSoln._dBs[:,0],0]
  
        refractory_indx_interval = where(ss == self.refractory_time)[0][0]

        OU_cross_indxs = []
        spike_indxs = []
        reset_indxs = [0]
        tk = 1;
        diffusion_regime = True;
        while tk < len(ss):
            V = Vs[tk];
            if True == diffusion_regime:
                if V > self.v_OU_bound:
                    OU_cross_indxs.append(tk);
                    diffusion_regime = False;
            else:
                if V > self.v_thresh:
                    spike_indxs.append(tk);
                    tk += refractory_indx_interval;
                    if tk<len(ss):
                        reset_indxs.append(tk)
                    else:
                        reset_indxs.append(len(ss)-1)
                    diffusion_regime = True
                        
            'increment time'       
            tk += 1
            
#        'add the last interval to diffusion regimes:'
#        if Vs[-1] < self.v_OU_bound and diffusion_regime==True:
#            OU_cross_indxs.append(len(ss)-1);
#            reset_indxs.append(len(ss)-1);
            
        'assign spike-times, reset-times, cross-times:'     
        spike_times = ss[spike_indxs];
        reset_times = ss[reset_indxs];
        OU_cross_times = ss[OU_cross_indxs];
        
        interval_ss = []
        interval_Vs = []
        
        diffusion_ss  = []
        diffusion_Vs  = []
        diffusion_dBs = []
#        print reset_indxs[:-1] ,OU_cross_indxs
        
        for rk_prev, rk_next, sk in zip(reset_indxs[:-1],
                                        reset_indxs[1:],
                                        OU_cross_indxs):
            OU_rip_indxs = (ss>= ss[rk_prev]) & (ss < ss[sk])
            interval_rip_indxs =  (ss>= ss[rk_prev]) & (ss < ss[rk_next])
            
            #Store diffusion part interval:
            diffusion_ss.append( ss[OU_rip_indxs]);
            diffusion_Vs.append( Vs[OU_rip_indxs] );
            diffusion_dBs.append( dBs[OU_rip_indxs] );
                                  
            #Store whole part of interval
            interval_ss.append( ss[interval_rip_indxs] );
            interval_Vs.append( Vs[interval_rip_indxs]);
        

        return OU_cross_times, spike_times, reset_times, \
                interval_ss, interval_Vs, \
                diffusion_ss, diffusion_Vs, diffusion_dBs   

    def save(self, file_name):
        file_name = os.path.join(RESULTS_DIR,
                                 file_name + self._FILE_EXTENSION)
        print 'saving ML annotated data to ', file_name
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

#######################################################################################
#######################################################################################
#######################################################################################

def MLSimulateHarness(resimulate=False, regime_tag = '1', Tf = 2000., dt = 1e-2):
    
    file_name = 'Basic_Example_Type%s_%.1f'%(regime_tag, Tf);
    
        
    if resimulate:
        seed(2014)
        ML_params = MLParametersDict[regime_tag]
        params_file_name = os.path.join(FIGS_DIR, 
                                        'MLParams_type%s.txt'%ML_params.regime_tag)
        ML_params.printme(file_name = params_file_name)
        
        
        MLSim = MLSimulator(ML_params);
        
        ts, Xs, dBs = MLSim.simulate(Tf=Tf, dt = dt,
                                     visualize=False)

        (MLSolution(ML_params, ts, Xs, dBs,
                     zeros_like(ts))).save(file_name)
    
    MLSoln = MLSolution.load(file_name)
    
    ts, Vs, Ws = MLSoln.get_ts_Vs_Ws()
    
    'Visualize:'
    sub_sample = 10;
    figure()
    subplot(211); 
    plot(ts[::sub_sample], Vs[::sub_sample]);
    ylabel('$V_s$', fontsize = xlabel_font_size)
    
    subplot(212); plot(ts[::10], Ws[::sub_sample] )
    ylabel('$W_s$',   fontsize = xlabel_font_size)
    xlabel('s', fontsize = xlabel_font_size)
    
    fig_file_name = os.path.join(FIGS_DIR,
                                 'ML_type%s_example_traj.pdf'%regime_tag)
    print fig_file_name
    savefig(fig_file_name, dpi=300)
    


def MLAnnotateHarness(refractory_time = 40., regime_tag = '1', Tf = 2000.):
    file_name = 'Basic_Example_Type%s_%.1f'%(regime_tag,  Tf)
    MLAnnSoln = None;
#    try:
#        MLAnnSoln = MLAnnotatedSolution.load(file_name)
#    except:
    MLSoln = MLSolution.load(file_name)
    MLAnnSoln = MLAnnotatedSolution(MLSoln,
                                    refractory_time=refractory_time)
    MLAnnSoln.save(file_name)       
    
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
             
    get_current_fig_manager().window.showMaximized()  
     
    fig_file_name = os.path.join(FIGS_DIR,
                                 'ML_estimates_annotated_%d.pdf'%Tf)
    print fig_file_name
    savefig(fig_file_name, dpi=300)


def generateStationaryOUData(Tf, m_C_sigma,
                       save_file_name = 'OU_streams'):
    m,C, sigma = m_C_sigma[:]

    print 'Generating OU Data with ', m,C,sigma
    
    '''load original ML data (as a basis of comparison'''
    ML_file_name = 'Basic_Example_Type%s_%.1f'%('1',  Tf)

    MLAnnSoln = MLAnnotatedSolution.load(ML_file_name);
    
    ML_ts, ML_Xs, ML_dWs  = MLAnnSoln.diffusion_ss,\
                            MLAnnSoln.diffusion_Vs,\
                            MLAnnSoln.diffusion_dBs,\
       
    dt = MLAnnSoln.MLSoln._ts[2] - MLAnnSoln.MLSoln._ts[1];
    def simulateX(ts, dWs, X_0):
    #Simulates a path of the OU process using the Millstein scheme:
        #Obtain a Wiener realization:
#        dWs =randn(len(ts)-1)* sqrt_dt     
        #allocate memory for soln:
        Xs = empty_like(ts)
        Xs[0]  =  X_0; 
        #evolve in time:
        for idx,t in enumerate(ts[:-1]):
            x = Xs[idx]
            #the drift:
            f = (m - x) / C
            #the volatility:
            g = sigma
                          
            Xs[idx+1] = x + f*dt + g*dWs[idx]  
        
        return ts, Xs
    
    N_trajectories = len(ML_Xs) 
    
    OUXS = []
    for stk in xrange(N_trajectories):
        X0 = m+100.* (-1)**(stk)*sigma
        ts, Xs = simulateX(ML_ts[stk], ML_dWs[stk], X0);
        OUXS.append(Xs)
        
    '''patch it up to look like original data:'''
    OU_params = array([m, C, sigma])

    MLAnnSoln.MLSoln._params = OU_params;
    MLAnnSoln.diffusion_Vs = OUXS;
    
    '''Visualize:'''
    figure(figsize = (17,8));
    subplots_adjust(left=.1, right=.975,
                    top = .95, bottom = .1,
                    hspace = .2)
    subplot(211)
    plot(MLAnnSoln.MLSoln._ts,
          MLAnnSoln.MLSoln._Xs[:,0], 'b')
    ylabel('$V_s$', fontsize = xlabel_font_size)
    subplot(212);
    max_shown = 40
    for stk in xrange(min(max_shown, N_trajectories)):
#            plot(ML_ts[stk],
#                  ML_Xs[stk], 'g', label='ML : $V_s$');
        plot(ML_ts[stk],
              OUXS[stk], 'r', label='OU : $Y_s$');
        if 0 == stk:
            legend(loc='lower center');
    ylabel('$Y_s, V_s$', fontsize = xlabel_font_size)
    xlabel('$s$', fontsize = xlabel_font_size)
    ylim((-55,-15))
        
    '''Save to disk:'''
    OU_file_name = '%s_T%d'%(save_file_name,
                             Tf)
    MLAnnSoln.save(OU_file_name)
    
    

def generateFakeMLDataFromOU(regime_tag = '1',
                              Tf = 20000.,
                              m_C_sigma = [-28.81,  34.718 ,  0.999],
                              save_file_name = 'MLFromOU',
                              visualize=True):
    m,C, sigma = m_C_sigma[:]

    print 'Generating OU Data with ', m,C,sigma
    
    '''load original ML data (as a basis of comparison'''
    ML_file_name = 'Basic_Example_Type%s_%.1f'%(regime_tag,  Tf)

    MLAnnSoln = MLAnnotatedSolution.load(ML_file_name);
    
    ML_ts, ML_Xs, ML_dWs  = MLAnnSoln.diffusion_ss,\
                            MLAnnSoln.diffusion_Vs,\
                            MLAnnSoln.diffusion_dBs,\
       
    dt = MLAnnSoln.MLSoln._ts[2] - MLAnnSoln.MLSoln._ts[1];
    def simulateX(ts, dWs, X_0):
    #Simulates a path of the OU process using the Millstein scheme:
        #Obtain a Wiener realization:
#        dWs =randn(len(ts)-1)* sqrt_dt     
        #allocate memory for soln:
        Xs = empty_like(ts)
        Xs[0]  =  X_0; 
        #evolve in time:
        for idx,t in enumerate(ts[:-1]):
            x = Xs[idx]
            #the drift:
            f = (m - x) / C
            #the volatility:
            g = sigma
                          
            Xs[idx+1] = x + f*dt + g*dWs[idx]  
        
        return ts, Xs
    
    N_trajectories = len(ML_Xs) 
    
    OUXS = []
    for stk in xrange(N_trajectories):
        ts, Xs = simulateX(ML_ts[stk], ML_dWs[stk], ML_Xs[stk][0]);
        OUXS.append(Xs)
        
    '''patch it up to look like original data:'''
    OU_params = array([m, C, sigma])

    MLAnnSoln.MLSoln._params = OU_params;
    MLAnnSoln.diffusion_Vs = OUXS;
    
    '''Visualize:'''
    if visualize:
        figure(figsize = (17,8));
        subplots_adjust(left=.1, right=.975,
                        top = .95, bottom = .1,
                        hspace = .2)
        subplot(211)
        plot(MLAnnSoln.MLSoln._ts,
              MLAnnSoln.MLSoln._Xs[:,0], 'b')
        ylabel('$V_s$', fontsize = xlabel_font_size)
        subplot(212);
        max_shown = 40
        for stk in xrange(min(max_shown, N_trajectories)):
            plot(ML_ts[stk],
                  ML_Xs[stk], 'g', label='ML : $V_s$');
            plot(ML_ts[stk],
                  OUXS[stk], 'r', label='OU : $Y_s$');
            if 0 == stk:
                legend(loc='lower center');
        ylabel('$Y_s, V_s$', fontsize = xlabel_font_size)
        xlabel('$s$', fontsize = xlabel_font_size)
        ylim((-55,-15))
        
        fig_name = os.path.join(FIGS_DIR,
                                'OU_from_ML_type%s_Tf%d.pdf'%(regime_tag,
                                                              Tf));
        print fig_name
        savefig(fig_name, dpi=300)
    
    '''Save to disk:'''
    OU_file_name = '%s_T%d'%(save_file_name,
                             Tf)
    MLAnnSoln.save(OU_file_name)
    
########################################################################################
########################################################################################
########################################################################################

def estimateOUfromML(MLAnnSoln,
                     visualize=False):
    'Find the OU parameters mu,tau,sigma from the diffusion part of an ML process'
    ts, Xs = MLAnnSoln.diffusion_ss,\
                MLAnnSoln.diffusion_Vs
      
    
    delta = MLAnnSoln.MLSoln._ts[2]-MLAnnSoln.MLSoln._ts[1];
    
    N_trajectories = len(Xs);
    
    def mu_hat_function(beta):
        mu_hat =0
        exp_beta_delta = exp(-beta*delta);
        N=0;
        for stk in xrange(N_trajectories):
            Xn = Xs[stk][1:] 
            Xm = Xs[stk][:-1] 
            N += len(Xn)
            mu_hat += sum( Xn - exp_beta_delta*Xm ) 
        mu_hat /= (N * (1 - exp_beta_delta))
        return mu_hat;
    
    def sigma_hat_function(beta, mu):
        sigma_squared = .0;
        exp_beta_delta = exp(-beta *delta);
        N=0;
        for stk in xrange(N_trajectories):
            Xn = Xs[stk][1:] 
            Xm = Xs[stk][:-1] 
            N += len(Xn)
            
            squared_term =  Xn - mu - (Xm-mu) * exp_beta_delta;
            sigma_squared += dot(squared_term,
                                 squared_term);
            
        sigma_squared *=  2. * beta /  \
                          (N * (1-exp(-2*beta * delta))) 
    
        return sqrt(sigma_squared);

    def root_functionAI(beta):
        #calc mu:
        mu = mu_hat_function(beta);
        
        #an exponential that occurs often:
        K = N_trajectories
        
        data_error_term = 1.0;
        for stk in xrange(N_trajectories):
            Xn = Xs[stk][1:] 
            Xm = Xs[stk][:-1] 
            
            if 0 == Xn.size:
                'WARNING: Why can it ever be empty...'
                continue
            numerator = dot(Xn-mu,
                            Xm-mu)
            denominator = dot(Xm-mu,
                              Xm-mu);
            #the reduced log-likelihood:               
            data_error_term *= numerator/denominator;
            
        b_error =  exp(-K*delta * beta) - data_error_term;
        
        return b_error
    
   
    #Perform Root finding:

    def root_function(beta):
        #calc mu:
        mu = mu_hat_function(beta);
        
        #an exponential that occurs often:
        K = N_trajectories
        
        numerator = .0
        denominator = .0;
        for stk in xrange(N_trajectories):
            Xn = Xs[stk][1:] 
            Xm = Xs[stk][:-1] 
            
            if 0 == Xn.size:
                'WARNING: Why can it ever be empty...'
                continue
            numerator += dot(Xn-mu,
                            Xm-mu)
            denominator += dot(Xm-mu,
                              Xm-mu);
        #the reduced log-likelihood:               
        data_error_term = numerator/denominator;
            
        b_error =  exp(-delta * beta) - data_error_term;
        
        return b_error
    
#    root_function = root_functionSD
    b_a = 0.001
    b_b =  1.
    
    if visualize:
        figure();
        bs = arange(b_a, b_b, .002)
        ers = array( [root_function(b) for b in bs])
        plot (bs, ers);
        hlines(.0, bs[0], bs[-1])
        try:
            vlines(1./MLAnnSoln.MLSoln._params[1], 0, amax(ers))
        except:
            pass

    from scipy.optimize import brentq
    'root-find beta'
    beta_hat = brentq(root_function, b_a, b_b,
                      xtol = 1e-8)
    'Compute the other two params'
    mu_hat = mu_hat_function(beta_hat)
    sigma_hat = sigma_hat_function(beta_hat, mu_hat)
        
    return array([mu_hat, 1.0/beta_hat, sigma_hat])


def estimateOUfromMLSpikesOnly(MLAnnSoln,
                               visualize=False,
                               C_hat = 20.):
    #HACK:
#    C_hat = 39.46
#    C_hat = 20.;
    
    spikes = MLAnnSoln.spike_times
    ISIs = diff(spikes) - MLAnnSoln.refractory_time;
    
#    figure()
#    subplot(211)
#    stem(spikes, ones_like(MLAnnSoln.spike_times));
#    subplot(212)
#    hist(ISIs, 16)
    
    
    ISIs /= C_hat
    
#    a,b = estimateNormalizedOUSpikes(ISIs,
#                                     theta_init=[1., 1.]);
    a,b = estimateNormalizedOUSpikes(ISIs,
                                     theta_init=[ 0.81,  0.267]);
    
    y_reset = MLAnnSoln.getYreset();
    diffusion_interval = MLAnnSoln.v_OU_bound - y_reset 
    
    print 'difusion intervals:',  MLAnnSoln.v_OU_bound, y_reset,diffusion_interval 
    
    mu_hat = a*diffusion_interval + y_reset
    sigma_hat = b*diffusion_interval / sqrt(C_hat)

    return array([mu_hat,
                   C_hat,
                    sigma_hat])
    
def estimateNormalizedOUSpikes(Is ,
                               theta_init = [1.0, 1.0]):
    print 'Fortet-Estimation starting from ',theta_init
    from scipy.optimize import fmin
    from scipy.stats import norm
    
    Is = sort(Is);
    N = len(Is)
        
    Phi = norm.cdf;
    
    def generateFokkerPlankSide(alpha, beta):
        def f (t) :
            numerator = sqrt(2)*(alpha* (1- exp(-t)) - 1);  
            denominator =  beta * sqrt(1 - exp(-2*t));
            return Phi( numerator / denominator );

        return f
    
    def generateFortetSide(alpha, beta):
        def g(t):
            taus = t - Is[Is <= t]
            exp_term = exp( - taus );
            sqrt_term = (1 - exp_term) / (1 + exp_term);
            phis = Phi(sqrt(2) * ( alpha - 1) / beta * sqrt( sqrt_term ));
            return sum(phis) / N;
       
        return g

    def loss_function(theta):
        #Rip args:               
        a = theta[0]; b= theta[1];
        
        #SIMPLE: Penalize negative a's, we want a positive, b/c for a<0, the algorithm is different:
        if a<.0:
            return 1e6

        ls = zeros_like(Is);
        f_FP = generateFokkerPlankSide(a,b)
        g_Fortet = generateFortetSide(a,b)

        for idx in xrange(len(ls)):
            s_n = Is[idx];
            ls[idx] = abs( f_FP(s_n) - g_Fortet(s_n) );
        return max(ls);
    
#    start = time.clock()
#    theat_opt= minimize(loss_function, theta_init, method='nelder-mead',
#                       options={'xtol': 1e-4, 'disp': True}) #v0.11
    theta_opt, fopt, iter,\
     funcalls, warnflag= fmin(loss_function, theta_init,
                               xtol=1e-2, ftol = 1e-2,
                                disp=0, full_output =1);
#    print 'compute_time = ' , time.clock()-start
#    print 'nelder_mead = ', theta_opt
#    print 'fval = ', fopt
    print 'Fortet Estimates ',theta_opt
    return theta_opt[0], theta_opt[1];


def MLEstimateHarness(regime_tag=1, Tf = 20000.0,
                      latexify=False):    
#    file_name = 'Basic_Example__type%s_%.1f'%(regime_tag,Tf)
#    file_name = 'MLFromOU'

    est_pars = []
    ML_file_name = 'Basic_Example_Type%s_%.1f'%(regime_tag,Tf)
    try:
        MLAnnSoln = MLAnnotatedSolution.load(ML_file_name)
    except:
        MLSoln = MLSolution.load(ML_file_name)
        MLAnnSoln = MLAnnotatedSolution(MLSoln)
        MLAnnSoln.save(ML_file_name)
    
    OU_MLestimates = estimateOUfromML(MLAnnSoln,
                                      visualize =False);
    
    OU_file_name = 'MLFromOU_T%d'%Tf
#    try:
#        MLAnnSoln = MLAnnotatedSolution.load(OU_file_name)
#    except:

    generateFakeMLDataFromOU(regime_tag, Tf,
                              m_C_sigma=OU_MLestimates)
    MLAnnSoln = MLAnnotatedSolution.load(OU_file_name)
    
    OU_OUestimates = estimateOUfromML(MLAnnSoln,
                                      visualize = False);

    print 'Tf:',Tf
    print 'OU_from_ML:', OU_MLestimates
    est_pars.append(list(OU_MLestimates));                           
    print 'OU_from_OU:', OU_OUestimates
    est_pars.append(list(OU_OUestimates));     

    'LAtexify'        
    if latexify:
        latex_str = r'Parameter, & ML1 & OU \\';
        for pname, idx in zip(['$\hat m$','$\hat C$','$\hat \sigma $'],
                              arange(3)):
            latex_str += r'%s & %.2f & %.2f \\' %(pname,
                                                     est_pars[0][idx],
                                                     est_pars[1][idx])
        print latex_str 
        
        latex_file_name = os.path.join(FIGS_DIR,
                                       'OU_from_ML1_estimates_T%d.txt'%Tf)
        latexFile = open(latex_file_name, 'w');
        print 'writing to ', latex_file_name
        latexFile.write(latex_str);
        latexFile.close()
        


def OUEstimateHarness(Tf = 160000.0):    
    OU_file_name = 'OU_from_ML'
    m_C_sigma = array([-28, 30, 1.]);
    generateFakeMLDataFromOU('1', Tf,
                              m_C_sigma=m_C_sigma,
                              save_file_name = OU_file_name)
    
    OU_file_name = 'OU_stat_trajs'
#    generateStationaryOUData(Tf, m_C_sigma=m_C_sigma,
#                       save_file_name = OU_file_name)
    
    MLAnnSoln = MLAnnotatedSolution.load('%s_T%d'%(OU_file_name,
                                                   Tf));
        
#    figure();
#    for k in xrange(len(MLAnnSoln.diffusion_ss)):
#        plot(MLAnnSoln.diffusion_ss[k],\
#                MLAnnSoln.diffusion_Vs[k]);
#    return
    
    OU_OUestimates = estimateOUfromML(MLAnnSoln,
                                      visualize = True);

    print m_C_sigma
    print OU_OUestimates     


def MLSpikesOnlyEstimateHarness(regime_tag=1, Tf = 20000.0,
                                refractory_time=40.0):    
    
    ML_file_name = 'Basic_Example_Type%s_%.1f'%(regime_tag,Tf)
    try:
        MLAnnSoln = MLAnnotatedSolution.load(ML_file_name)
    except:
        MLSoln = MLSolution.load(ML_file_name)
        MLAnnSoln = MLAnnotatedSolution(MLSoln)
        MLAnnSoln.save(ML_file_name)
    
    OU_MLestimates = estimateOUfromMLSpikesOnly(MLAnnSoln,
                                                  visualize =True);
    
                
    est_pars = []
    OU_file_name = 'MLFromOU_spikes_only_T%d'%Tf
    
    generateFakeMLDataFromOU(regime_tag, Tf,
                              m_C_sigma=OU_MLestimates,
                              save_file_name='MLFromOU_spikes_only')
    MLAnnSoln = MLAnnotatedSolution.load(OU_file_name)
    
    OU_OUestimates = estimateOUfromMLSpikesOnly(MLAnnSoln,
                                                visualize = False);

    print 'Tf:',Tf
    print 'OU_from_ML:', OU_MLestimates
    est_pars.append(list(OU_MLestimates));                           
    print 'OU_from_OU:', OU_OUestimates
    est_pars.append(list(OU_OUestimates)); 
    print est_pars
    
    latex_str = r'Parameter, & ML1(spikes-only) \\';
    for pname, idx in zip(['$\hat m$','$\hat C$','$\hat \sigma $'],
                          arange(3)):
        latex_str += r'%s & %.2f \\' %(pname,
                                             est_pars[0][idx])
    print latex_str 
    
    latex_file_name = os.path.join(FIGS_DIR,
                                   'OU_from_ML1_estimates_spikes_only.txt')
    latexFile = open(latex_file_name, 'w');
    print 'writing to ', latex_file_name
    latexFile.write(latex_str);
    latexFile.close()

def UnitsConversionHarness(regime_tag='1', Tf = 2000.0):
    file_name = 'Basic_Example_Type%s_%.1f'%(regime_tag,Tf)
    
    OUParams, MLParams  = generateOUFromMLAnnSoln(file_name,
                                                  A_bounds = [-10, 10])
    
    ss = arange(0,OUParams.meanISI,1e-1)
    ts = OUParams.nondimensionalizeTime(ss)
    
    alphas = sin(ts*(2*pi))
#    figure(); plot(ts,alphas)
    X_0 = .0;
    mu, tau_char, beta =  OUParams.getMuTauBeta()
    def simulateX(xis):
        Xs = empty_like(ts)
        Xs[0]  =  X_0;
        dt = ts[2]-ts[1];
        sqrt_dt = sqrt(dt); 
        #evolve in time:
        for idx,t in enumerate(ts[:-1]):
            x = Xs[idx]
            #the drift:
            f = (mu + alphas[idx] - x/tau_char)
            #the volatility:
            g = beta
                          
            Xs[idx+1] = x + f*dt + g*sqrt_dt * xis[idx]  
        
        return Xs
    As = OUParams.dimensionalizeControl(alphas);
    Y_0 = OUParams.yreset
    def simulateY(xis):
        m, C, sigma = OUParams.getMCSigma()
        Ys = empty_like(ss);
        Ys[0] = Y_0;
        ds = ss[2]-ss[1];
        sqrt_ds = sqrt(ds); 
        #evolve in time:
        for idx,t in enumerate(ss[:-1]):
            y = Ys[idx]
            #the drift:
            f = (m + As [idx] - y)/C
            #the volatility:
            g = sigma
                          
            Ys[idx+1] = y + f*ds + g*sqrt_ds * xis[idx]  
        
        return Ys
    
    xis = randn(len(ts)-1);
    
    Xs = simulateX(xis);
    Ys = simulateY(xis);
    t_ss, X_ys  = OUParams.nondimensionalizeState(ss, Ys)
    
    figure(figsize=(17,8))
    
    subplot(211)
    sub_sample = 8;
    plot(ss[::sub_sample], Ys[::sub_sample], 'b');
    subplot(212)
    plot(ts[::sub_sample],Xs[::sub_sample], 'rx',
          label=r'$X_t$-simulated')
    plot(t_ss[::sub_sample], X_ys[::sub_sample], 'b',
           label=r'$Y_t$-non-dimensionalized')
    legend(loc='lower right')
    fig_name = os.path.join(FIGS_DIR,
                            'OUDim2NonDimConversion.pdf')
    print fig_name
    savefig(fig_name);
    
    OUParams.printme()
    OUParams.printme('OUDims2Nondims.txt')
    

def SpikeOnlyEstimates(regime_tag='1', Tf = 20000.0):
    file_name = 'Basic_Example_Type%s_%.1f'%(regime_tag,Tf)
    
    OUParams, MLParams  = generateOUSpikesOnlyFromMLAnnSoln(file_name,
                                                            A_bounds = [-10, 10])
    
    OUParams.printme()
    OUParams.printme('OUDims2Nondims_spikesonly.txt')
    
    
if __name__ == '__main__':
    from pylab import *
    
#    MLSimulateHarness(resimulate=False, regime_tag='1', Tf = 2e3);
#    MLSimulateHarness(resimulate=False, regime_tag='2')

#    MLAnnotateHarness()
#    MLEstimateHarness(Tf=2e3);

    
    Tfs = [2e3, 2e4, 4e4, 16e4, 64e4];
#    for Tf in Tfs[0:2]:
#        MLEstimateHarness(Tf=Tf,
#                          latexify=True)


#    UnitsConversionHarness(Tf=2e4)
#    SpikeOnlyEstimates()

###################################################
##################  SPIKES ONLY:    ###############
###################################################
#    MLSimulateHarness(resimulate=True, Tf=2e4, dt = 1e-1)

#    MLSpikesOnlyEstimateHarness(Tf=2e4)
#    for Tf in [2e4, 16e4]:
#        OUEstimateHarness(Tf=Tf)
       
    show()
            