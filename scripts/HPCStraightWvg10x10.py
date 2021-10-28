import numpy as np
from PMM.PMMInverse import PMMI
import os

a = 0.015
res = 80
nx = 20
ny = 20
dpml = 2
b_o = 0.0075/a
b_i = 0.0065/a
output = os.getcwd()+'/../outputs'
fname = '10by10straightwaveguide_ez_w025_wpmax035_gam1GHz_res80_coldstart'
run_no = ['_r0','_r1']

## Set up domain geometry #####################################################
PPC = PMMI(a, res, nx, ny, dpml) #Initialize PMMI object
PPC.Add_Block_static((0, 8.5), (5, 0.5), -1000.0) #Add entrance wvg
PPC.Add_Block_static((0, 11), (5, 0.5), -1000.0) #Add entrance wvg
PPC.Add_Block_static((15, 8.5), (5, 0.5), -1000.0) #Right exit wvg
PPC.Add_Block_static((15, 11), (5, 0.5), -1000.0) #Right exit wvg
PPC.Add_Block_static((8.5, 15), (0.5, 5), -1000.0) #Top exit wvg
PPC.Add_Block_static((11, 15), (0.5, 5), -1000.0) #Top exit wvg
PPC.Design_Region((5, 5), (10, 10)) #Specify Region where elements are being optimized
PPC.Rod_Array_train(b_i, (5.5, 5.5), (10, 10), bulbs = True,\
                    d_bulb = (b_i, b_o), eps_bulb = 3.8, uniform = False) #Rod ppc array


## Set up Sources and Sim #####################################################
w = 0.25 #Source frequency
wpmax = 0.35
gamma = PPC.gamma(1e9)

PPC.Add_Source(np.array([3,9]), np.array([3,11]), w, 'src', 'ez')
PPC.Add_Probe(np.array([17,9]), np.array([17,11]), w, 'prb', 'ez')
PPC.Add_Probe(np.array([9,17]), np.array([11,17]), w, 'prbl', 'ez')

rod_eps = 0.999*np.ones((10, 10)) #Rod perm values
rho = PPC.Eps_to_Rho(epsr = rod_eps, plasma = True, w_src = w, wp_max = wpmax) #Initial Parameters
#rho = PPC.Read_Params(output+'/params/'+fname+run_no[0]+'.csv')
#Norms = PPC.Read_Params(output+'/params/'+fname+'_norms.csv')

rho_opt, obj, E0, E0l = PPC.Optimize_Waveguide_Penalize(rho, 'src', 'prb', 'prbl',\
               0.0001, 500, plasma = True, wp_max = wpmax, gamma = gamma, uniform = False,\
               param_evolution = True, param_out = output+'/run_params')
#               param_evolution = True, param_out = output+'/run_params',\
#               E0 = Norms[0], E0l = Norms[1])

## Save parameters and visualize ##############################################
PPC.Save_Params(rho_opt, output+'/params/'+fname+run_no[1]+'.csv')
PPC.Save_Params(np.array([E0, E0l]), output+'/params/'+fname+'_norms.csv') 
PPC.Params_to_Exp(rho = rho_opt, src = 'src', plasma = True)
PPC.Save_Params(obj, output+'/plots/'+fname+'_obj'+run_no[1]+'.csv')
