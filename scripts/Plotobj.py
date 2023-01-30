import numpy as np
from PMM.PMMInverse import PMMI
import os

a = 0.018
res = 50
nx = 20
ny = 20
dpml = 2
PPC = PMMI(a, res, nx, ny, dpml) #Initialize PMMI object so you can use Viz_Obj
output = os.getcwd()+'/../outputs'
fname = '10by10bentwaveguide_ez_w025_wpmax035_gam1GHz_res50_idealstart'
runs = 3
obj = np.array([])

for i in range(runs):
    obj_i = PPC.Read_Params(output+'/plots/'+fname+'_obj_r'+str(i+1)+'.csv')
    obj = np.append(obj,obj_i)
    
PPC.Viz_Obj(obj, output+'/plots/'+fname+'_obj_full.pdf')