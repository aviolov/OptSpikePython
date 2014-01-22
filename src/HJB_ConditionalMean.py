'''
Created on Oct 26, 2012

@author: alex
'''

from FPSolver import FPSolver

from HJBSolver import FIGS_DIR
import os;

if __name__ == '__main__':
    from pylab import *
    betas =  [.3, .9, 1.5]
    
    #Visualize:    
    mpl.rcParams['figure.subplot.left'] = .1
    mpl.rcParams['figure.subplot.right'] = .95
    mpl.rcParams['figure.subplot.bottom'] = .1
    mpl.rcParams['figure.subplot.top'] = .9
    mpl.rcParams['figure.subplot.hspace'] = .25
#    mpl.rcParams['figure.subplot.vspace'] = .25
    
    
    figure()
    
    for beta, sim_idx in zip(betas, arange(1,len(betas)+1)):
    
        atb = [2., .6, beta];
        alpha,tauchar, beta = atb[0], atb[1], atb[2];
        
        Tf = 1.5;
        energy_eps = .1;
        
        dx = .025
        x_min = FPSolver.calculate_xmin(Tf, atb);    
        print 'Xmin = ', x_min, '| Tf = ', Tf;
        S = FPSolver(dx, FPSolver.calculate_dt(dx, atb, x_min), Tf, x_min)
        
        ts, F =  S.solve(atb)
        Fth = F[:,-1]
        
        xest = 1.0 - sum(F,axis = 1)*dx / Fth
        
        mean_x = atb[0] * atb[1] * (1 - exp( -ts /atb[1]) )
        
        subplot(len(betas),1,sim_idx); hold(True)        
        plot(ts, xest,'b', label='$\hat{x}_t$')
        plot(ts, mean_x, 'r', label='$m_x$')
        legend(loc='upper left'); ylabel('$x$', fontsize=24)
        if (1 == sim_idx):
            title(r'$\alpha=%.2f, \tau=%.2f, \beta=%.2f$'%(alpha,tauchar, beta), fontsize = 24)
        else:
            title(r'$\beta=%.2f$'%(beta), fontsize = 24)
        if 3 == sim_idx:
            xlabel('$t$', fontsize=24);
            
    get_current_fig_manager().window.showMaximized()
    
    save_fig_name = 'ConditionalVsUnconditionalMoments_a=%.0f_t=%.0f_betasf'%(10*alpha,10*tauchar)
    file_name = os.path.join(FIGS_DIR, save_fig_name+'.png')
    print 'saving to ', file_name
    savefig(file_name) 
    
    
    show()
                                
                                
                                 