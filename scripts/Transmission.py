import numpy as np
from PMM.PMMInverse import PMMI
import os

a = 0.020
res = 50
nx = 21
ny = 21
dpml = 2
b_o = 0.0075/a
b_i = 0.0065/a
entrance = 0.04/a
output = os.getcwd()+'/../outputs'
fname = '8by8BntWvg_ez_w033_wpmax047_gam0GHz_res50_coldstart_r2'

## Set up domain geometry #####################################################
PPC = PMMI(a, res, nx, ny, dpml) #Initialize PMMI object
w = 0.33 #Source frequency
wpmax = 0.47
gamma = 0#PPC.gamma(1e9)

PPC.Add_INFOMW_Horn(np.array([6.5, 10.5]), np.array([1,0]), 6.5, pol ='TM')
PPC.Add_INFOMW_Horn(np.array([14.5, 10.5]), np.array([-1,0]), 6.5, pol ='TM')
PPC.Add_INFOMW_Horn(np.array([10.5, 6.5]), np.array([0,1]), 6.5, pol ='TM')
PPC.Add_INFOMW_Horn(np.array([10.5, 14.5]), np.array([0,-1]), 6.5, pol ='TM')
PPC.Design_Region((6.5, 6.5), (8, 8)) #Specify Region where elements are being optimized

uniform = True
PPC.Rod_Array_train(b_i/2**.5, (7, 7), (8, 8), bulbs = True,\
                    d_bulb = (b_i, b_o), eps_bulb = 3.8, uniform = uniform) #Rod ppc array

PPC.Add_Source(np.array([3,9.5]), np.array([3,11.5]), w, 'src', 'ez')

src_right = [[np.array([[3.025,9.5],[3.025,11.5]]),np.array([1,0,0])]]
src_left = [[np.array([[3,9.5],[3,11.5]]),np.array([1,0,0])]]
src_exit = [[np.array([[6.5,8.1],[6.5,12.9]]),np.array([1,0,0])]]
crys_right = [[np.array([[14.5,6.5],[14.5,14.5]]),np.array([1,0,0])]]
crys_left = [[np.array([[6.5,6.5],[6.5,14.5]]),np.array([1,0,0])]]
crys_top = [[np.array([[6.5,14.5],[14.5,14.5]]),np.array([0,1,0])]]
crys_bot = [[np.array([[6.5,6.5],[14.5,6.5]]),np.array([0,1,0])]]

s21 = [[np.array([[9.5,18],[11.5,18]]),np.array([0,1,0])]]
s31 = [[np.array([[18,9.5],[18,11.5]]),np.array([1,0,0])]]
s41 = [[np.array([[9.5,3],[11.5,3]]),np.array([0,-1,0])]]

# right_nc = PPC.Get_Trans_Denom(['src'], src_right, plot = True, savepath = output+'/plots/SParam_denom.pdf')
# left_nc = PPC.Get_Trans_Denom(['src'], src_left, plot = False, savepath = output+'/plots/SParam_denom.pdf')
# exit_nc = PPC.Get_Trans_Denom(['src'], src_exit, plot = False, savepath = output+'/plots/SParam_denom.pdf')

# print('no crystal left:', left_nc)
# print('no crystal right:', right_nc)
# print('no crystal exit:', exit_nc)
# print('no crystal insertion loss: ', -(1-exit_nc/((right_nc[0]-left_nc[0])/2)))
# print('no crystal insertion loss (dB):', 10*np.log10(exit_nc/((right_nc[0]-left_nc[0])/2)))

# PPC.Clear_fields()

## Set up Sources and Sim #####################################################
rho = PPC.Read_Params(output+'/params/'+fname+'.csv')
right = PPC.Get_Trans_Denom(['src'], src_right, opt = True, rho = rho, plasma = True,\
                                plot = True, wp_max = wpmax, gamma = gamma,\
                                uniform = uniform, savepath = output+'/plots/SParam_denom.pdf')
left = PPC.Get_Trans_Denom(['src'], src_left, plot = False, savepath = output+'/plots/SParam_denom.pdf')
exit_ = PPC.Get_Trans_Denom(['src'], src_exit, plot = False, savepath = output+'/plots/SParam_denom.pdf')
c_right = PPC.Get_Trans_Denom(['src'], crys_right, plot = False, savepath = output+'/plots/SParam_denom.pdf')
c_left = PPC.Get_Trans_Denom(['src'], crys_left, plot = False, savepath = output+'/plots/SParam_denom.pdf')
c_top = PPC.Get_Trans_Denom(['src'], crys_top, plot = False, savepath = output+'/plots/SParam_denom.pdf')
c_bot = PPC.Get_Trans_Denom(['src'], crys_bot, plot = False, savepath = output+'/plots/SParam_denom.pdf')
s2 = PPC.Get_Trans_Denom(['src'], s21, plot = False, savepath = output+'/plots/SParam_denom.pdf')
s3 = PPC.Get_Trans_Denom(['src'], s31, plot = False, savepath = output+'/plots/SParam_denom.pdf')
s4 = PPC.Get_Trans_Denom(['src'], s41, plot = False, savepath = output+'/plots/SParam_denom.pdf')

print('left:', left)
print('right:', right)
print('exit:', exit_)
print('insertion loss: ', -(1-exit_[0]/((right[0]-left[0])/2)))
print('insertion loss (dB):', 10*np.log10(exit_[0]/((right[0]-left[0])/2)))
print('crystal top:', c_top)
print('crystal bottom:', c_bot)
print('crystal left:', c_left)
print('crystal right:', c_right)
print('loss to crystal:', (c_top[0]-c_bot[0]+c_right[0]-c_left[0])/((right[0]-left[0])/2))
print('transmission 2:', s2)
print('transmission 3:', s3)
print('S11:', 10*np.log10((((right[0]-left[0])/2)-right[0])/((right[0]-left[0])/2)))
print('S21:', 10*np.log10(s2[0]/((right[0]-left[0])/2)))
print('S31:', 10*np.log10(s3[0]/((right[0]-left[0])/2)))
print('S41:', 10*np.log10(s4[0]/((right[0]-left[0])/2)))
print('S21 neglecting insertion loss:', 10*np.log10(s2[0]/exit_))
print('S31 neglecting insertion loss:', 10*np.log10(s3[0]/exit_))
print('S41 neglecting insertion loss:', 10*np.log10(s4[0]/exit_))

