import numpy as np
from PMM.PMMInverse import PMMI
import os

a = 0.020
res = 50
nx = 19
ny = 20
dpml = 2
b_o = 0.0075/a
b_i = 0.0065/a
entrance = 0.04/a
output = os.getcwd()+'/../outputs'
fname = 'Hex5Wvg_ez_w033_wpmax047_gam0GHz_res50_coldstart'
run_no = ['_r0','_r1']

## Set up domain geometry #####################################################
PPC = PMMI(a, res, nx, ny, dpml) #Initialize PMMI object
PPC.Add_INFOMW_Horn(np.array([6.5, 10]), np.array([1,0]), 6.5, pol ='TM')
PPC.Add_INFOMW_Horn(4*np.array([0.5, 3**0.5/2])+np.array([10.5,10]), np.array([-0.5,-3**0.5/2]), 7, pol ='TM')
PPC.Add_INFOMW_Horn(4*np.array([0.5, -3**0.5/2])+np.array([10.5,10]), np.array([-0.5,3**0.5/2]), 7, pol ='TM')
PPC.Design_Region((6.5, 5.5), (8, 9)) #Specify Region where elements are being optimized

uniform = True
PPC.Rod_Array_Hexagon_train(np.array([10.5,10]), 5, b_i/2**0.5, 1,\
                          a_basis = np.array([[0,1],[np.sqrt(3)/2,1./2]]),\
                          bulbs = True, r_bulb = (b_i, b_o), eps_bulb = 3.8) #Rod ppc array


## Set up Sources and Sim #####################################################
w = 0.33 #Source frequency
wpmax = 0.47
gamma = 0#PPC.gamma(1e9)

horn_dir_1 = np.array([0.5,3**0.5/2])
horn_dir_2 = np.array([0.5,-3**0.5/2])
open_dir_1 = np.array([3**0.5/2,-0.5])
open_dir_2 = np.array([3**0.5/2,0.5])

PPC.Add_Source(np.array([2.5,9]), np.array([2.5,11]), w, 'src', 'ez')
PPC.Add_Probe(-4*horn_dir_1 + 2*open_dir_1 + 10.5,\
              -4*horn_dir_1 - 2*open_dir_1 + 10.5, w, 'prb1', 'ez')
PPC.Add_Probe(-4*horn_dir_2 + 2*open_dir_2 + 10.5,\
              -4*horn_dir_2 - 2*open_dir_2 + 10.5, w, 'prb2', 'ez')

rod_eps = 0.999*np.ones(61) #Rod perm values
rho = PPC.Eps_to_Rho(epsr = rod_eps, plasma = True, w_src = w, wp_max = wpmax) #Initial Parameters
#rho = PPC.Read_Params(output+'/params/'+fname+run_no[0]+'.csv')
#E0 = PPC.Read_Params(output+'/params/'+fname+'_norm_src.csv')
#E0l = PPC.Read_Params(output+'/params/'+fname+'_norm_prb.csv')

rho_opt, obj, E0, E0l = PPC.Optimize_Waveguide_Penalize(rho, 'src', 'prb1', ['prb2'],\
               0.005, 100, plasma = True, wp_max = wpmax, gamma = gamma, uniform = uniform,\
               param_evolution = True, param_out = output+'/run_params')
#               param_evolution = True, param_out = output+'/run_params',\
#               E0 = E0, E0l = E0l)

## Save parameters and visualize ##############################################
PPC.Save_Params(rho_opt, output+'/params/'+fname+run_no[1]+'.csv')
PPC.Save_Params(np.array([E0]), output+'/params/'+fname+'_norm_src.csv') 
PPC.Save_Params(np.array(E0l), output+'/params/'+fname+'_norm_prb.csv')
print(PPC.Rho_to_Eps(rho = rho_opt, plasma = True, w_src = w))
PPC.Params_to_Exp(rho = rho_opt, src = 'src', plasma = True)
PPC.Viz_Sim_abs_opt(rho_opt, ['src'], output+'/plots/'+fname+run_no[1]+'.pdf',\
                    plasma = True, wp_max = wpmax, uniform = uniform, gamma = gamma)
PPC.Save_Params(obj, output+'/plots/'+fname+'_obj'+run_no[1]+'.csv')
PPC.Viz_Obj(obj, output+'/plots/'+fname+'_obj'+run_no[1]+'.pdf')