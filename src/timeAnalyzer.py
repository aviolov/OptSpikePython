'''
Created on Feb 11, 2014

@author: alex
'''

if __name__ == '__main__':
    from pylab import *
    lif_output = '''spike_iterates = 119, ISI_compute_time = 0.00
spike_iterates = 1444, ISI_compute_time = 0.03
spike_iterates = 500, ISI_compute_time = 0.01
spike_iterates = 1182, ISI_compute_time = 0.02
spike_iterates = 563, ISI_compute_time = 0.01
spike_iterates = 540, ISI_compute_time = 0.02
spike_iterates = 540, ISI_compute_time = 0.01
spike_iterates = 1130, ISI_compute_time = 0.02
spike_iterates = 1104, ISI_compute_time = 0.02
spike_iterates = 884, ISI_compute_time = 0.02
spike_iterates = 122, ISI_compute_time = 0.00
spike_iterates = 1537, ISI_compute_time = 0.02
spike_iterates = 443, ISI_compute_time = 0.01
spike_iterates = 471, ISI_compute_time = 0.00
spike_iterates = 798, ISI_compute_time = 0.00
spike_iterates = 1315, ISI_compute_time = 0.00'''
    
    ml_output = '''
spike_iterates = 1104, ISI_compute_time = 0.16
spike_iterates = 2404, ISI_compute_time = 0.30
spike_iterates = 2121, ISI_compute_time = 0.27
spike_iterates = 646, ISI_compute_time = 0.09
spike_iterates = 734, ISI_compute_time = 0.09
spike_iterates = 703, ISI_compute_time = 0.08
spike_iterates = 622, ISI_compute_time = 0.08
spike_iterates = 906, ISI_compute_time = 0.11
spike_iterates = 560, ISI_compute_time = 0.07
    '''.strip()
    
    for tag, raw_output in zip(['LIF', 'ML'],
                               [lif_output, ml_output]):
    
        lines = raw_output.split('\n');
        Times = []
    
        for line in lines:
            new_tuple =  line.split(',')
            Times.append( (int(new_tuple[0].split('=')[1].strip()),
                              float(new_tuple[1].split('=')[1].strip())));
        
        Times = array(Times)
    
#            print lifTimes
        print 'per iterate = %.6f'%(mean(Times[:,1]/ \
                                         Times[:,0] ))
        