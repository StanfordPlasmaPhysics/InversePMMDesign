import numpy as np
from PMM.PMMInverse import PMMI
import os

a = 0.020
res = 50
nx = 19
ny = 19
dpml = 2
b_o = 0.0075/a
b_i = 0.0065/a
entrance = 0.04/a
output = os.getcwd()+'/../outputs'
fname = '6by6StrWvg_ez_w025_wpmax047_gam0GHz_res50_coldstart'
run_no = ['_r0','_r1']

## Set up domain geometry #####################################################
PPC = PMMI(a, res, nx, ny, dpml) #Initialize PMMI object
PPC.Add_INFOMW_Horn(np.array([6.5, 9.5]), np.array([1,0]), 6.5, pol ='TM')
PPC.Add_INFOMW_Horn(np.array([12.5, 9.5]), np.array([-1,0]), 6.5, pol ='TM')
PPC.Add_INFOMW_Horn(np.array([9.5, 6.5]), np.array([0,-1]), 6.5, pol ='TM')
PPC.Add_INFOMW_Horn(np.array([9.5, 12.5]), np.array([0,1]), 6.5, pol ='TM')
PPC.Design_Region((6.5, 6.5), (6, 6)) #Specify Region where elements are being optimized

uniform = True
PPC.Rod_Array_train(b_i, (7, 7), (6, 6), bulbs = True,\
                    d_bulb = (b_i, b_o), eps_bulb = 3.8, uniform = uniform) #Rod ppc array


## Set up Sources and Sim #####################################################
w = 0.25 #Source frequency
wpmax = 0.47
gamma = 0#PPC.gamma(1e9)

PPC.Add_Source(np.array([3,8.5]), np.array([3,10.5]), w, 'src', 'ez')
PPC.Add_Probe(np.array([8.5,16]), np.array([10.5,16]), w, 'prb1', 'ez')
PPC.Add_Probe(np.array([16,8.5]), np.array([16,10.5]), w, 'prb2', 'ez')
PPC.Add_Probe(np.array([8.5,3]), np.array([10.5,3]), w, 'prb1', 'ez')

rod_eps = 0.999*np.ones((6, 6)) #Rod perm values
rho = PPC.Eps_to_Rho(epsr = rod_eps, plasma = True, w_src = w, wp_max = wpmax) #Initial Parameters
#rho = PPC.Read_Params(output+'/params/'+fname+run_no[0]+'.csv')
#Norms = PPC.Read_Params(output+'/params/'+fname+'_norms.csv')

rho_opt, obj, E0, E0l = PPC.Optimize_Waveguide_Penalize(rho, 'src', 'prb', 'prbl',\
               0.005, 100, plasma = True, wp_max = wpmax, gamma = gamma, uniform = uniform,\
               param_evolution = True, param_out = output+'/run_params')
#               param_evolution = True, param_out = output+'/run_params',\
#               E0 = Norms[0], E0l = Norms[1])

## Save parameters and visualize ##############################################
PPC.Save_Params(rho_opt, output+'/params/'+fname+run_no[1]+'.csv')
PPC.Save_Params(np.array([E0, E0l]), output+'/params/'+fname+'_norms.csv') 
print(PPC.Rho_to_Eps(rho = rho_opt, plasma = True, w_src = w))
PPC.Params_to_Exp(rho = rho_opt, src = 'src', plasma = True)
PPC.Viz_Sim_abs_opt(rho_opt, ['src'], output+'/plots/'+fname+run_no[1]+'.pdf',\
                    plasma = True, wp_max = wpmax, uniform = uniform, gamma = gamma)
PPC.Save_Params(obj, output+'/plots/'+fname+'_obj'+run_no[1]+'.csv')
PPC.Viz_Obj(obj, output+'/plots/'+fname+'_obj'+run_no[1]+'.pdf')
