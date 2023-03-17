import numpy as np
from PMM.PMMInverse import PMMI
import os

a = 0.020
res = 50
nx = 18
ny = 22
dpml = 2
b_o = 0.0075/a
b_i = 0.0065/a
entrance = 0.04/a
output = os.getcwd()+'/../outputs'
fname = 'Hex5Wvg_up_ez_w6GHz_wpmax8GHz_gam0GHz_res50_uniformstart_noleak'
run_no = ['_r0','_r1']

## Set up domain geometry #####################################################
PPC = PMMI(a, res, nx, ny, dpml) #Initialize PMMI object
PPC.Add_INFOMW_Horn(np.array([6.5, 11]), np.array([1,0]), 6.5, pol ='TM')
PPC.Add_INFOMW_Horn(4*np.array([0.5, 3**0.5/2])+np.array([10.5,11]), np.array([-0.5,-3**0.5/2]), 8, pol ='TM')
PPC.Add_INFOMW_Horn(4*np.array([0.5, -3**0.5/2])+np.array([10.5,11]), np.array([-0.5,3**0.5/2]), 8, pol ='TM')
PPC.Design_Region((6.5, 6.5), (8, 9)) #Specify Region where elements are being optimized

uniform = False
PPC.Rod_Array_Hexagon_train(np.array([10.5,11]), 5, b_i, 1,\
                          a_basis = np.array([[0,1],[np.sqrt(3)/2,1./2]]),\
                          bulbs = True, r_bulb = (b_i, b_o), eps_bulb = 3.8,
                          uniform = uniform) #Rod ppc array


## Set up Sources and Sim #####################################################
w = PPC.gamma(6.001e9) #Source frequency
wpmax = PPC.gamma(8e9)
gamma = 0#PPC.gamma(1e9)

ew = 0.048/a/2-0.004/a
hd = 0.089/a
x = np.array([1,0])
y = np.array([0,1])
horn_dir_1 = np.array([0.5,3**0.5/2])
horn_dir_2 = np.array([0.5,-3**0.5/2])
open_dir_1 = np.array([3**0.5/2,-0.5])
open_dir_2 = np.array([3**0.5/2,0.5])
cen = np.array([10.5,11])

PPC.Add_Source(np.array([6.5-hd,11-ew]), np.array([6.5-hd,11+ew]), w, 'src', 'ez')
PPC.Add_Probe((4+hd)*horn_dir_1 + ew*open_dir_1 + cen,\
              (4+hd)*horn_dir_1 - ew*open_dir_1 + cen, w, 'prb1', 'ez')
PPC.Add_Probe((4+hd)*horn_dir_2 + ew*open_dir_2 + cen,\
              (4+hd)*horn_dir_2 - ew*open_dir_2 + cen, w, 'prb2', 'ez')
PPC.Add_Probe(-4.8*(3**0.5/2)*horn_dir_1 + 2.05*open_dir_1 + cen,\
              -4.8*(3**0.5/2)*horn_dir_1 - 2.05*open_dir_1 + cen, w, 'loss_ul', 'ez')
PPC.Add_Probe(-4.8*(3**0.5/2)*horn_dir_2 + 2.05*open_dir_2 + cen,\
              -4.8*(3**0.5/2)*horn_dir_2 - 2.05*open_dir_2 + cen, w, 'loss_ll', 'ez')
PPC.Add_Probe(4.8*(3**0.5/2)*x - 2.05*y + cen,\
              4.8*(3**0.5/2)*x + 2.05*y + cen, w, 'loss_R', 'ez')
    
#rod_eps = 0.999*np.ones(61) #Rod perm values
#rho = PPC.Eps_to_Rho(epsr = rod_eps, plasma = True, w_src = w, wp_max = wpmax) #Initial Parameters
rho = PPC.Read_Params(output+'/params/Hex5Wvg_up_ez_w6GHz_wpmax8GHz_gam0GHz_res50_coldstart_noleak_r3.csv')
#rho = PPC.Read_Params(output+'/params/'+fname+run_no[0]+'.csv')
#E0 = PPC.Read_Params(output+'/params/'+fname+'_norm_src.csv')
#E0l = PPC.Read_Params(output+'/params/'+fname+'_norm_prb.csv')

rho_opt, obj, E0, E0l = PPC.Optimize_Waveguide_Penalize(rho, 'src', 'prb1', ['prb2', 'loss_ul', 'loss_ll', 'loss_R'],\
               0.01, 600, plasma = True, wp_max = wpmax, gamma = gamma, uniform = uniform,\
               param_evolution = True, param_out = output+'/run_params')
#               param_evolution = True, param_out = output+'/run_params',\
#               E0 = E0, E0l = E0l)

## Save parameters and visualize ##############################################
PPC.Save_Params(rho_opt, output+'/params/'+fname+run_no[1]+'.csv')
PPC.Save_Params(np.array([E0]), output+'/params/'+fname+'_norm_src.csv') 
PPC.Save_Params(np.array(E0l), output+'/params/'+fname+'_norm_prb.csv')
print(PPC.Rho_to_Eps(rho = rho_opt, plasma = True, w_src = w))
PPC.Params_to_Exp(rho = rho_opt, src = 'src', plasma = True, wp_max = wpmax)
PPC.Viz_Sim_abs_opt(rho_opt, ['src'], output+'/plots/'+fname+run_no[1]+'.pdf',\
                    plasma = True, wp_max = wpmax, uniform = uniform, gamma = gamma)
PPC.Save_Params(obj, output+'/plots/'+fname+'_obj'+run_no[1]+'.csv')
PPC.Viz_Obj(obj, output+'/plots/'+fname+'_obj'+run_no[1]+'.pdf')