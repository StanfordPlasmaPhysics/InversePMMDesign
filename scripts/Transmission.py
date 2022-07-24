import numpy as np
from PMM.PMMInverse import PMMI
import os

a = 0.018
res = 75
nx = 20
ny = 20
dpml = 2
b_o = 0.0075/a
b_i = 0.0065/a
output = os.getcwd()+'/../outputs'
fname = 'WaveguideOpt_res75'

## Set up domain geometry #####################################################
PPC = PMMI(a, res, nx, ny, dpml) #Initialize PMMI object
w = 0.25 #Source frequency
wpmax = 0.35
gamma = -PPC.gamma(1e9)

PPC.Add_Block((0, 8.5), (20, 0.5), -1000.0) #Add entrance wvg
PPC.Add_Block((0, 11), (20, 0.5), -1000.0) #Add entrance wvg

PPC.Add_Source(np.array([3,9]), np.array([3,11]), w, 'src', 'ez')
PPC.Add_Source(np.array([10,9]), np.array([10,11]), w, 'src_cen', 'ez')

s11 = [[np.array([[3.1,9],[3.1,11]]),np.array([1,0,0])]]
s11_tot = [[[np.array([[3.0,9],[3.0,11]]),np.array([1,0,0])],[np.array([[3.01,9],[3.01,11]]),np.array([1,0,0])]]]
s11_cen = [[np.array([[10,9],[10,11]]),np.array([0,1,0])]]
s11_cen2 = [[np.array([[10,9],[10,11]]),np.array([1,0,0])]]
s11_cen3 = [[np.array([[0,8.4],[20,8.4]]),np.array([0,1,0])]]
s11_cen4 = [[np.array([[0,11.6],[20,11.6]]),np.array([0,1,0])]]
s11_cen_tot = [[[np.array([[10.1,9],[10.1,11]]),np.array([1,0,0])],[np.array([[9.9,9],[9.9,11]]),np.array([-1,0,0])]]]

# denom = PPC.Get_Trans_Denom(['src_cen'], s11_cen, savepath = output+'/plots/SParam_denom.pdf')
# denom2 = PPC.Get_Trans_Denom(['src_cen'], s11_cen2, plot = False, savepath = output+'/plots/SParam_denom.pdf')
# denom3 = PPC.Get_Trans_Denom(['src_cen'], s11_cen3, plot = False, savepath = output+'/plots/SParam_denom.pdf')
# denom4 = PPC.Get_Trans_Denom(['src_cen'], s11_cen4, plot = False, savepath = output+'/plots/SParam_denom.pdf')
# Total_Flux = PPC.Get_Total_Flux(['src_cen'], s11_cen_tot, plot = False)
# PPC.Viz_Sim_Fields(['src_cen'], output+'/plots/SParam_denom_fields.pdf', perm = True)
# PPC.Clear_fields()

# print('denom:', denom, denom2, denom3, denom4)

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
s21 = [[[np.array([[17,9],[17,11]]), np.array([1,0,0])],[np.array([[9,17],[11,17]]), np.array([0,1,0])]]]
prb = [[np.array([[17,9],[17,11]]), np.array([0,1,0])]]

rho = PPC.Read_Params(output+'/params/10by10straightwaveguide_ez_w025_wpmax035_gam1GHz_res75_idealstart_r8.csv')
Total_Flux_crystal = PPC.Get_Total_Flux(['src'], s11_tot, opt = True, rho = rho, plasma = True,\
                                plot = True, wp_max = wpmax, gamma = gamma,\
                                uniform = False)
# Left = PPC.Get_Trans_Denom(['src'], [s11_tot[0][0]],  plot = False, savepath = output+'/plots/SParam_denom.pdf')
# Right = PPC.Get_Trans_Denom(['src'], [s11_tot[0][1]], plot = False, savepath = output+'/plots/SParam_denom.pdf')
F_out_Right = PPC.Get_Trans_Denom(['src'], [[np.array([[15,5],[15,15]]), np.array([1,0,0])]], plot = False, savepath = output+'/plots/SParam_denom.pdf')
F_out_Left = PPC.Get_Trans_Denom(['src'], [[np.array([[5,5],[5,15]]), np.array([-1,0,0])]], plot = False, savepath = output+'/plots/SParam_denom.pdf')
F_out_Top = PPC.Get_Trans_Denom(['src'], [[np.array([[5,15],[15,15]]), np.array([0,1,0])]], plot = False, savepath = output+'/plots/SParam_denom.pdf')
F_out_Bot = PPC.Get_Trans_Denom(['src'], [[np.array([[5,5],[15,5]]), np.array([0,-1,0])]], plot = False, savepath = output+'/plots/SParam_denom.pdf')
print(F_out_Right)
print(F_out_Left)
print(F_out_Top)
print(F_out_Bot)
# PPC.Calc_SParams_opt(rho, ['src'], denom, s11, s21, output+'/plots/'+fname+'.pdf',\
#                     plasma = True, plot = True, wp_max = wpmax, gamma = gamma,\
#                     uniform = False)

# PPC.Viz_Sim_Poynting_opt(rho, u_s, ['src'], output+'/plots/'+fname+'.pdf', plasma = True,\
#                          show = True, mult = False, wp_max = wpmax, gamma = gamma, uniform = False)
# PPC.Viz_Sim_abs_opt(rho, ['src'], output+'/plots/'+fname+'.pdf', plasma = True,\
#                             show = True, mult = False, wp_max = wpmax, gamma = gamma, uniform = False,\
#                             perturb = 0)

#PPC.Add_Source(src_start, src_end, 0.1, 'src', 'ez')
#PPC.Viz_Sim_Poynting(['src'], uhat, output+'/plots/'+fname+'_fields.pdf', perm = True)
