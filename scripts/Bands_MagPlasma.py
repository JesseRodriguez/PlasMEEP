from plasmeep.lib import Plasmeep as pm
import numpy as np
import csv
import matplotlib.pyplot as plt
import os

output = os.getcwd()+'/../outputs/plots/'

# Parameters for simulation
res = 100
nx = 1
ny = 1
r = 0.065 #m
B = 0.05 #T
a_l = np.linspace(0.02,0.04,11)

fs = 8*10**9 #Hz
fp = 8*10**9 #Hz
gamma = 0 #Hz
F = np.array([fs,fp,gamma])
F = model.Nondimensionalize_Freq(F)

for a in a_l:
    savefile_B = output+'bands_'+str(a)+'mm_B_on.csv'
    savefile_noB = output+'bands_'+str(a)+'mm_B_off.csv'
    model_B = pm(a, res, 0, nx, ny, B=np.array([0,0,B]))
    model_noB = pm(a, res, 0, nx, ny, B=np.array([0,0,0]))

    ## Build Geometry
    model_B.Add_Rod(r/a, [0,0,0], eps = 1, wp = F[1], gamma = F[2], axis = [0,0,1])
    model_noB.Add_Rod(r/a, [0,0,0], eps = 1, wp = F[1], gamma = F[2], axis = [0,0,1])

    ## Simulate
    model_B.Bands2D(30, F[0], [0,0,1], E = False, savefile = savefile_B)
    model_B.BandPlot(30, model.Dimensionalize_Freq(np.array([0,1.5]))/10**9,\
                   savefile_B, dims = True, dimlabel = 'GHz')
    model_noB.Bands2D(30, F[0], [0,0,1], E = False, savefile = savefile_noB)
    model_noB.BandPlot(30, model.Dimensionalize_Freq(np.array([0,1.5]))/10**9,\
                   savefile_noB, dims = True, dimlabel = 'GHz')
