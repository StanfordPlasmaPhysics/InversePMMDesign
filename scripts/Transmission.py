import numpy as np
from PMM.PMMInverse import PMMI
import os

a = 0.01
res = 40
nx = 28
ny = 12
dpml = 2
output = os.getcwd()+'/../outputs'
fname = 'WR229TransStudy_noplasma_res40'

## Set up domain geometry #####################################################
PPC = PMMI(a, res, nx, ny, dpml) #Initialize PMMI object
PPC.Add_Block((0, 8.9), (28, 0.3), -1000.0) #Add wvg wall
PPC.Add_Block((0, 2.8), (28, 0.3), -1000.0) #Add wvg wall


## Set up Sources and Sim #####################################################
w = np.linspace(0.001,0.25,40) #Source frequency

src_start = np.array([3,3.1])
src_end = np.array([3,8.9])
prb_start = np.array([20,3.1])
prb_end = np.array([20,8.9])
uhat = np.array([1,0,0])

PPC.Transmission_Spect(src_start, src_end, uhat, prb_start, prb_end, uhat, 'ez',\
                       w, output+'/plots/'+fname+'.pdf', show = True)

#PPC.Add_Source(src_start, src_end, 0.1, 'src', 'ez')
#PPC.Viz_Sim_Poynting(['src'], uhat, output+'/plots/'+fname+'_fields.pdf', perm = True)
