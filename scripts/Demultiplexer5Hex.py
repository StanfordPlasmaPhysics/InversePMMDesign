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
fname = 'Hex5Demult_ez_w1_03_w2_033_wpmax047_gam1GHz_res50_0GHzstart'
run_no = ['_r0','_r1']

## Set up domain geometry #####################################################
PPC = PMMI(a, res, nx, ny, dpml) #Initialize PMMI object
PPC.Add_INFOMW_Horn(np.array([6.5, 10]), np.array([1,0]), 6.5, pol ='TM')
PPC.Add_INFOMW_Horn(4*np.array([0.5, 3**0.5/2])+np.array([10.5,10]), np.array([-0.5,-3**0.5/2]), 7, pol ='TM')
PPC.Add_INFOMW_Horn(4*np.array([0.5, -3**0.5/2])+np.array([10.5,10]), np.array([-0.5,3**0.5/2]), 7, pol ='TM')
PPC.Design_Region((6.5, 5.5), (8, 9)) #Specify Region where elements are being optimized

uniform = False
PPC.Rod_Array_Hexagon_train(np.array([10.5,10]), 5, b_i/2**0.5, 1,\
                          a_basis = np.array([[0,1],[np.sqrt(3)/2,1./2]]),\
                          bulbs = True, r_bulb = (b_i, b_o), eps_bulb = 3.8,
                          uniform = uniform) #Rod ppc array


## Set up Sources and Sim #####################################################
w1 = 0.30 #Source frequency
w2 = 0.33
wpmax = 0.47
gamma = PPC.gamma(1e9)

horn_dir_1 = np.array([0.5,3**0.5/2])
horn_dir_2 = np.array([0.5,-3**0.5/2])
open_dir_1 = np.array([3**0.5/2,-0.5])
open_dir_2 = np.array([3**0.5/2,0.5])
cen = np.array([10.5,10])

PPC.Add_Source(np.array([2.5,9]), np.array([2.5,11]), w1, 'src_1', 'ez')
PPC.Add_Source(np.array([2.5,9]), np.array([2.5,11]), w2, 'src_2', 'ez')
PPC.Add_Probe(8*horn_dir_1 + open_dir_1 + cen,\
               8*horn_dir_1 - open_dir_1 + cen, w1, 'prb_1', 'ez')
PPC.Add_Probe(8*horn_dir_2 + open_dir_2 + cen,\
              8*horn_dir_2 - open_dir_2 + cen, w2, 'prb_2', 'ez')

#rod_eps = 0.999*np.ones(61) #Rod perm values
#rho = PPC.Eps_to_Rho(epsr = rod_eps, plasma = True, w_src = w1, wp_max = wpmax) #Initial Parameters
#rho = PPC.Read_Params(output+'/params/'+fname+run_no[0]+'.csv')
rho = PPC.Read_Params(output+'/params/Hex5Demult_ez_w1_03_w2_033_wpmax047_gam0GHz_res50_uniformstart_r1.csv')
#Norms = PPC.Read_Params(output+'/params/'+fname+'_norms.csv')

rho_opt, obj, E01, E02, E01l, E02l = PPC.Optimize_Multiplexer_Penalize(rho, 'src_1', 'src_2', 'prb_1',\
                                            'prb_2', 0.01, 100, plasma = True,\
                                             wp_max = wpmax, gamma = gamma, uniform = uniform,\
                                             param_evolution = True, param_out = output+'/run_params')
#                                             param_evolution = True, param_out = output+'/run_params',\
#                                             E01 = Norms[0], E02 = Norms[1],\
#                                             E01l = Norms[2], E02l = Norms[3])

## Save parameters and visualize ##############################################
PPC.Save_Params(rho_opt, output+'/params/'+fname+run_no[1]+'.csv')
PPC.Save_Params(np.array([E01, E02, E01l, E02l]), output+'/params/'+fname+'_norms.csv') 
print(PPC.Rho_to_Eps(rho = rho_opt, plasma = True, w_src = w1, wp_max = 0.47))
PPC.Params_to_Exp(rho = rho_opt, src = 'src_1', plasma = True,  wp_max = 0.47)
PPC.Viz_Sim_abs_opt(rho_opt,  ['src_1', 'src_2'], output+'/plots/'+fname+run_no[1]+'.pdf',\
                    plasma = True, wp_max = wpmax, uniform = uniform, gamma = gamma)
PPC.Save_Params(obj, output+'/plots/'+fname+'_obj'+run_no[1]+'.csv')
PPC.Viz_Obj(obj, output+'/plots/'+fname+'_obj'+run_no[1]+'.pdf')