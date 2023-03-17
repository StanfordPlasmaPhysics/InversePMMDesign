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
fname = 'Hex5Demult_ez_w1_4GHz_w2_6GHz_wpmax8GHz_gam1GHz_res50_0GHzstart_noleak'
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
w1 = PPC.gamma(4e9) #Source frequency
w2 = PPC.gamma(6.01e9)
wpmax = PPC.gamma(8e9)
gamma = PPC.gamma(1e9)

ew = 0.048/a/2-0.004/a
hd = 0.089/a
x = np.array([1,0])
y = np.array([0,1])
horn_dir_1 = np.array([0.5,3**0.5/2])
horn_dir_2 = np.array([0.5,-3**0.5/2])
open_dir_1 = np.array([3**0.5/2,-0.5])
open_dir_2 = np.array([3**0.5/2,0.5])
cen = np.array([10.5,11])

PPC.Add_Source(np.array([6.5-hd,11-ew]), np.array([6.5-hd,11+ew]), w1, 'src_1', 'ez')
PPC.Add_Source(np.array([6.5-hd,11-ew]), np.array([6.5-hd,11+ew]), w2, 'src_2', 'ez')
PPC.Add_Probe((4+hd)*horn_dir_1 + ew*open_dir_1 + cen,\
              (4+hd)*horn_dir_1 - ew*open_dir_1 + cen, w1, 'prb_1', 'ez')
PPC.Add_Probe((4+hd)*horn_dir_2 + ew*open_dir_2 + cen,\
              (4+hd)*horn_dir_2 - ew*open_dir_2 + cen, w2, 'prb_2', 'ez')
PPC.Add_Probe(-4.8*(3**0.5/2)*horn_dir_1 + 2.05*open_dir_1 + cen,\
              -4.8*(3**0.5/2)*horn_dir_1 - 2.05*open_dir_1 + cen, w1, 'loss_ul', 'ez')
PPC.Add_Probe(-4.8*(3**0.5/2)*horn_dir_2 + 2.05*open_dir_2 + cen,\
              -4.8*(3**0.5/2)*horn_dir_2 - 2.05*open_dir_2 + cen, w1, 'loss_ll', 'ez')
PPC.Add_Probe(4.8*(3**0.5/2)*x - 2.05*y + cen,\
              4.8*(3**0.5/2)*x + 2.05*y + cen, w1, 'loss_R', 'ez')

#rod_eps = 0.999*np.ones(61) #Rod perm values
#rho = PPC.Eps_to_Rho(epsr = rod_eps, plasma = True, w_src = w1, wp_max = wpmax) #Initial Parameters
#rho = PPC.Read_Params(output+'/params/'+fname+run_no[0]+'.csv')
rho = PPC.Read_Params(output+'/params/Hex5Demult_ez_w1_4GHz_w2_6GHz_wpmax8GHz_gam0GHz_res50_uniformstart_noleak_r4.csv')
#Norms = PPC.Read_Params(output+'/params/'+fname+'_norms.csv')

rho_opt, obj, E01, E02, E01l, E02l, E0l = PPC.Optimize_Multiplexer_Penalize(rho, 0.01, 200, 'src_1', 'src_2',\
                                            'prb_1', 'prb_2', ['loss_ul', 'loss_ll', 'loss_R'], plasma = True,\
                                             wp_max = wpmax, gamma = gamma, uniform = uniform,\
                                             param_evolution = True, param_out = output+'/run_params')
#                                             param_evolution = True, param_out = output+'/run_params',\
#                                             E01 = Norms[0], E02 = Norms[1],\
#                                             E01l = Norms[2], E02l = Norms[3], E0l = Norms[4:])

## Save parameters and visualize ##############################################
PPC.Save_Params(rho_opt, output+'/params/'+fname+run_no[1]+'.csv')
PPC.Save_Params(np.array([E01, E02, E01l, E02l]+E0l.tolist()), output+'/params/'+fname+'_norms.csv') 
print(PPC.Rho_to_Eps(rho = rho_opt, plasma = True, w_src = w1, wp_max = wpmax))
PPC.Params_to_Exp(rho = rho_opt, src = 'src_1', plasma = True,  wp_max = wpmax)
PPC.Viz_Sim_abs_opt(rho_opt,  ['src_1', 'src_2'], output+'/plots/'+fname+run_no[1]+'.pdf',\
                    plasma = True, wp_max = wpmax, uniform = uniform, gamma = gamma)
PPC.Save_Params(obj, output+'/plots/'+fname+'_obj'+run_no[1]+'.csv')
PPC.Viz_Obj(obj, output+'/plots/'+fname+'_obj'+run_no[1]+'.pdf')