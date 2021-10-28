import numpy as np
from PMM.PMMInverse import PMMI
import os

a = 0.015
res = 75
nx = 20
ny = 20
dpml = 2
b_o = 0.007/a
b_i = 0.0065/a
output = os.getcwd()+'/../outputs'
fname = '10by10multiplexer_ez_w025_wpmax035_gam1GHz_res75_coldstart'
run_no = ['_r0','_r1']

## Set up domain geometry #####################################################
PPC = PMMI(a, res, nx, ny, dpml) #Initialize PMMI object
PPC.Add_Block_static((0, 8.5), (5, 0.5), -1000.0) #Add entrance wvg
PPC.Add_Block_static((0, 11), (5, 0.5), -1000.0) #Add entrance wvg
PPC.Add_Block_static((15, 5.5), (5, 0.5), -1000.0) #Bottom exit wvg
PPC.Add_Block_static((15, 8), (5, 0.5), -1000.0) #Bottom exit wvg
PPC.Add_Block_static((15, 11.5), (5, 0.5), -1000.0) #Top exit wvg
PPC.Add_Block_static((15, 14), (5, 0.5), -1000.0) #Top exit wvg
PPC.Design_Region((5, 5), (10, 10)) #Specify Region where elements are being optimized
PPC.Rod_Array_train(b_i, (5.5, 5.5), (10, 10), bulbs = True,\
                    d_bulb = (b_i, b_o), eps_bulb = 3.8, uniform = False) #Rod ppc array


## Set up Sources and Sim #####################################################
w1 = 0.25 #Source frequency
w2 = 0.27
wpmax = 0.35 
gamma = PPC.gamma(1e9)


PPC.Add_Source(np.array([3,9]), np.array([3,11]), w1, 'src_1', 'ez')
PPC.Add_Source(np.array([3,9]), np.array([3,11]), w2, 'src_2', 'ez')
PPC.Add_Probe(np.array([17,6]), np.array([17,8]), w1, 'prb_1', 'ez')
PPC.Add_Probe(np.array([17,12]), np.array([17,14]), w2, 'prb_2', 'ez')

rod_eps = 0.999*np.ones((10, 10)) #Rod perm values
rho = PPC.Eps_to_Rho(epsr = rod_eps, plasma = True, w_src = w1, wp_max = wpmax) #Initial Parameters
#rho = PPC.Read_Params(output+'/params/'+fname+run_no[0]+'.csv')
#Norms = PPC.Read_Params(output+'/params/'+fname+'_norms.csv')

rho_opt, obj, E01, E02, E01l, E02l = PPC.Optimize_Multiplexer_Penalize(rho, 'src_1', 'src_2', 'prb_1',\
                                            'prb_2', 0.00001, 100, plasma = True,\
                                             wp_max = wpmax, gamma = gamma, uniform = False,\
                                             param_evolution = True, param_out = output+'/run_params')
#                                             param_evolution = True, param_out = output+'/run_params',\
#                                             E01 = Norms[0], E02 = Norms[1],\
#                                             E01l = Norms[2], E02l = Norms[3])

## Save parameters and visualize ##############################################
PPC.Save_Params(rho_opt, output+'/params/'+fname+run_no[1]+'.csv')
PPC.Save_Params(np.array([E01, E02, E01l, E02l]), output+'/params/'+fname+'_norms.csv') 
print(PPC.Rho_to_Eps(rho = rho_opt, plasma = True, w_src = w1))
PPC.Params_to_Exp(rho = rho_opt, src = 'src_1', plasma = True)
PPC.Viz_Sim_abs_opt(rho_opt, ['src_1', 'src_2'],\
                    output+'/plots/'+fname+run_no[1]+'.pdf', plasma = True,\
                    wp_max = wpmax, uniform = False, gamma = gamma)
PPC.Save_Params(obj, output+'/plots/'+fname+'_obj'+run_no[1]+'.csv')
PPC.Viz_Obj(obj, output+'/plots/'+fname+'_obj'+run_no[1]+'.pdf')