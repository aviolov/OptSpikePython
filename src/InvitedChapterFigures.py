'''
Created on Jun 30, 2015

@author: alex
'''

import os
from ControlSimulator import ControlSimulation, SpikedPath
from ControlSimulator import visualizeTrajectoriesPaper
from ControlSimulator import mu_low, tau_char, beta_high

from HJBSolver import FIGS_DIR as hjb_figs_dir, HJBSolution, xlabel_font_size, label_font_size

from pylab import *


#def DeterministicOptimalControls():
#    regimeParams = 
#    regimeLabels = 
#    visualizeRegimesSinglePlot(regimeParams, 
#                                regimeLabels,
#                                 Tf=  1.0, 
#                                  energy_eps = .1,
#                                   fig_name = 'RegimesChapter'):

def ExampleControlTrajectoriesEpsilonLowHigh(): 
    visualizeTrajectoriesPaper([mu_low/tau_char, tau_char, beta_high], 1.5, 
                               loweps_file_name='LowEps_Trajs',
                               higheps_file_name='HighEps_Trajs',
                               controls_show_list=['CL'],
                               paths_to_show = [10],       #[6,8,11] ~ used in JNE Paper                        
                               fig_name = 'CompositeChapter')

def ExampleHJBSolnSurfaces(Tf=1.5, energy_eps = .001,  alpha_bounds = [-2,2] ):
    #Visualize:
    from mpl_toolkits.mplot3d.axes3d import Axes3D,proj3d
    line_width = 3
    
    tau_char = .5;
    beta_high = 1.5
    mu_high = 1.5
    
    params=[mu_high/tau_char, tau_char, beta_high];
    ################################
    ### CONTROL SURFACES:
    ####################################
    fig = figure(figsize = (17, 8))
    fig.hold(True)
    subplots_adjust(hspace = .15, wspace = .05,
                     left= .1, right=.9,
                     top = .95, bottom = .05) 
    hjbSoln = HJBSolution.load(mu_beta_Tf = params[::2]+[Tf],
                                energy_eps = energy_eps)
    print energy_eps
    print 'mu,tc,b = %.2f,%.2f,%.2f'%(hjbSoln._mu,hjbSoln._tau_char, hjbSoln._beta) 
    ts,xs,vs,cs = hjbSoln._ts, hjbSoln._xs, hjbSoln._vs, hjbSoln._cs
    xs = xs[1:-1];
    cs = cs[1:, :]
    print 'delta t = %.4f, delta x = %.4f'%(ts[2]-ts[1], xs[2]-xs[1])
    
    ax = fig.add_subplot(1,2,1,
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
        slice_xs = xs;
        slice_cs = cs[:, tk];
        slice_ts = ones_like(slice_cs)*ts[tk];
        ax.plot(slice_ts, slice_xs, slice_cs,
                color = slice_color,
                linewidth = 2*line_width)
    ax.set_xlim((ts[0]-xoffset,ts[-1]+xoffset))
    ax.set_ylim((xs[0]-yoffset,xs[-1]+yoffset))       
    ax.set_xlabel('$t$', fontsize = xlabel_font_size); 
 
    ax.text2D(.125, .2, '$x$',
        horizontalalignment='center', verticalalignment='center',
        transform=ax.transAxes,
        fontsize = xlabel_font_size)
    ax.text2D(.04, .6, r'$\alpha(x,t)$',
                     fontsize = xlabel_font_size,
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=ax.transAxes)
        
    ticks = [ts[0], ts[-1]]
    ax.set_xticks(ticks)
    ax.set_xticklabels(['$0$', '$t^*$'])
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
        
    ####################################
    ### VALUE FUNC SURFACES:
    #################################### 
    ax = fig.add_subplot(1,2,2,
                          projection='3d')
    ts,xs,vs,cs = hjbSoln._ts, hjbSoln._xs, hjbSoln._vs, hjbSoln._cs
    
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
#    ax.set_zlabel('$w(x,t)$', fontsize = xlabel_font_size);
    ax.text2D(1.025, .4, r'$w(x,t)$',
                     fontsize = xlabel_font_size,
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=ax.transAxes)
    ax.text2D(.75, .025, '$x$',
        horizontalalignment='center', verticalalignment='center',
        transform=ax.transAxes,
        fontsize = xlabel_font_size)
    
    ticks = [ts[0], ts[-1]]
    ax.set_xticks(ticks)
    ax.set_xticklabels(['$0$', '$t^*$'])
    ticks = [.0, 1]
    ax.set_yticks(ticks)
    ax.set_yticklabels([r'$%.0f$'%tick for tick in ticks])
    max_v = 2.0 #amax(vs)
    ticks = [.0, max_v]
    ax.set_zticks(ticks)
    ax.set_zticklabels([r'$%.0f$'%tick for tick in ticks])
    
    for label in ax.xaxis.get_majorticklabels():
        label.set_fontsize(label_font_size)
    for label in ax.yaxis.get_majorticklabels():
        label.set_fontsize(label_font_size  )
    for label in ax.zaxis.get_majorticklabels():
        label.set_fontsize(label_font_size  ) 
    
    get_current_fig_manager().window.showMaximized()        
    lfig_name = os.path.join(hjb_figs_dir, 'Chapter_SupraTHighNoise_r_value_control_surf.pdf')
    print 'saving to ', lfig_name
    savefig(lfig_name)

def ExampleML(tag='supra_threshold'):
    pass
    
    
if __name__ == '__main__':
    
    'Figure 2:'
#    ExampleHJBSolnSurfaces()
    
    'Figure 3:'
#    ExampleControlTrajectoriesEpsilonLowHigh();
    
    
    'Figure 4a'
#    ExampleML(tag='supra_threshold')
    
    'Figure 4b'
#    ExampleML(tag='sub_threshold')
    
    show();
    
    