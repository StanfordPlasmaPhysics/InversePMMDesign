import numpy as np
from PMM.PMMInverse import PMMI
import os

a = 0.018
res = 50
nx = 20
ny = 20
dpml = 2
b_o = 0.0075/a
b_i = 0.0065/a
output = os.getcwd()+'/../outputs'
fname = 'WaveguideOpt_res50'

## Set up domain geometry #####################################################
PPC = PMMI(a, res, nx, ny, dpml) #Initialize PMMI object
w = 0.25 #Source frequency
wpmax = 0.35
gamma = PPC.gamma(1e9)

PPC.Add_Block((0, 8.5), (5, 0.5), -1000.0) #Add entrance wvg
PPC.Add_Block((0, 11), (5, 0.5), -1000.0) #Add entrance wvg

PPC.Add_Source(np.array([3,9]), np.array([3,11]), w, 'src', 'ez')

src_right = [[np.array([[3.025,9],[3.025,11]]),np.array([1,0,0])]]
src_left = [[np.array([[3,9],[3,11]]),np.array([1,0,0])]]
src_exit = [[np.array([[5.025,9],[5.025,11]]),np.array([1,0,0])]]
crys_right = [[np.array([[15,5],[15,15]]),np.array([1,0,0])]]
crys_left = [[np.array([[5,5],[5,15]]),np.array([1,0,0])]]
crys_top = [[np.array([[5,15],[15,15]]),np.array([0,1,0])]]
crys_bot = [[np.array([[5,5],[15,5]]),np.array([0,1,0])]]

s21 = [[np.array([[17,9],[17,11]]),np.array([1,0,0])]]
s31 = [[np.array([[9,17],[11,17]]),np.array([0,1,0])]]

# right_nc = PPC.Get_Trans_Denom(['src'], src_right, plot = True, savepath = output+'/plots/SParam_denom.pdf')
# left_nc = PPC.Get_Trans_Denom(['src'], src_left, plot = False, savepath = output+'/plots/SParam_denom.pdf')
# exit_nc = PPC.Get_Trans_Denom(['src'], src_exit, plot = False, savepath = output+'/plots/SParam_denom.pdf')

# print('no crystal left:', left_nc)
# print('no crystal right:', right_nc)
# print('no crystal exit:', exit_nc)
# print('no crystal insertion loss: ', -(1-exit_nc/((right_nc[0]-left_nc[0])/2)))
# print('no crystal insertion loss (dB):', 10*np.log10(exit_nc/((right_nc[0]-left_nc[0])/2)))

# PPC.Clear_fields()

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
rho = PPC.Read_Params(output+'/params/10by10bentwaveguide_ez_w025_wpmax042_gam1GHz_res50_coldstart_r5.csv')
right = PPC.Get_Trans_Denom(['src'], src_right, opt = True, rho = rho, plasma = True,\
                                plot = True, wp_max = wpmax, gamma = gamma,\
                                uniform = False, savepath = output+'/plots/SParam_denom.pdf')
left = PPC.Get_Trans_Denom(['src'], src_left, plot = False, savepath = output+'/plots/SParam_denom.pdf')
exit_ = PPC.Get_Trans_Denom(['src'], src_exit, plot = False, savepath = output+'/plots/SParam_denom.pdf')
c_right = PPC.Get_Trans_Denom(['src'], crys_right, plot = False, savepath = output+'/plots/SParam_denom.pdf')
c_left = PPC.Get_Trans_Denom(['src'], crys_left, plot = False, savepath = output+'/plots/SParam_denom.pdf')
c_top = PPC.Get_Trans_Denom(['src'], crys_top, plot = False, savepath = output+'/plots/SParam_denom.pdf')
c_bot = PPC.Get_Trans_Denom(['src'], crys_bot, plot = False, savepath = output+'/plots/SParam_denom.pdf')
s2 = PPC.Get_Trans_Denom(['src'], s21, plot = False, savepath = output+'/plots/SParam_denom.pdf')
s3 = PPC.Get_Trans_Denom(['src'], s31, plot = False, savepath = output+'/plots/SParam_denom.pdf')

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
print('S21 neglecting insertion loss:', 10*np.log10(s2[0]/exit_))
print('S31 neglecting insertion loss:', 10*np.log10(s3[0]/exit_))

