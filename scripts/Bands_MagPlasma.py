from plasmeep.lib import Plasmeep as pm
import meep as mp
import numpy as np
import csv
import matplotlib.pyplot as plt
import os

output = os.getcwd()+'/../outputs/plots/'

# Parameters for simulation
res = 64
nx = 3**0.5
ny = 1
r = 0.005 #m
B = 0.05 #T
a_l = np.linspace(0.02,0.04,11)

fs = 8*10**9 #Hz
fp = 8*10**9 #Hz
gamma = 0#10**9 #Hz
Fs = np.array([fs,fp,gamma])
k_ps = 30

Pol = [0,0,1]
size = mp.Vector3(0,0,0)
E = False

for a in a_l:
    savefile_B = output+'bands_%.2gmm_B_on.csv' %a
    savefile_noB = output+'bands_%.2gmm_B_off.csv' %a
    foldsavefile_B = output+'foldedbands_%.2gmm_B_on.csv' %a
    foldsavefile_noB = output+'foldedbands_%.2gmm_B_off.csv' %a
    model_B = pm(a, res, 0, nx, ny, B=np.array([0,0,B]))
    model_noB = pm(a, res, 0, nx, ny, B=np.array([0,0,0]))
    model_B_folded = pm(a, res, 0, nx, ny, B=np.array([0,0,B]))
    model_noB_folded = pm(a, res, 0, nx, ny, B=np.array([0,0,0]))

    F = model_B.Nondimensionalize_Freq(Fs)

    ## Build Geometry
    model_B.Add_Rod(r/a, [0,0,0], wp = F[1], gamma = F[2], axis = [0,0,1])
    model_B.Add_Rod(r/a, [np.sqrt(3)/2,0.5,0], wp = F[1], gamma = F[2], axis = [0,0,1])
    model_B.Add_Rod(r/a, [np.sqrt(3)/2,-0.5,0], wp = F[1], gamma = F[2], axis = [0,0,1])
    model_B.Add_Rod(r/a, [-np.sqrt(3)/2,0.5,0], wp = F[1], gamma = F[2], axis = [0,0,1])
    model_B.Add_Rod(r/a, [-np.sqrt(3)/2,-0.5,0], wp = F[1], gamma = F[2], axis = [0,0,1])
    
    model_noB.Add_Rod(r/a, [0,0,0], wp = F[1], gamma = F[2], axis = [0,0,1])
    model_noB.Add_Rod(r/a, [np.sqrt(3)/2,0.5,0], wp = F[1], gamma = F[2], axis = [0,0,1])
    model_noB.Add_Rod(r/a, [np.sqrt(3)/2,-0.5,0], wp = F[1], gamma = F[2], axis = [0,0,1])
    model_noB.Add_Rod(r/a, [-np.sqrt(3)/2,0.5,0], wp = F[1], gamma = F[2], axis = [0,0,1])
    model_noB.Add_Rod(r/a, [-np.sqrt(3)/2,-0.5,0], wp = F[1], gamma = F[2], axis = [0,0,1])
    
    model_B_folded.Add_Rod(r/a, [0,0,0], wp = F[1], gamma = F[2], axis = [0,0,1])
    model_B_folded.Add_Rod(r/a, [np.sqrt(3)/2,0.5,0], wp = F[1], gamma = F[2], axis = [0,0,1])
    model_B_folded.Add_Rod(r/a, [np.sqrt(3)/2,-0.5,0], wp = F[1], gamma = F[2], axis = [0,0,1])
    model_B_folded.Add_Rod(r/a, [-np.sqrt(3)/2,0.5,0], wp = F[1], gamma = F[2], axis = [0,0,1])
    model_B_folded.Add_Rod(r/a, [-np.sqrt(3)/2,-0.5,0], wp = F[1], gamma = F[2], axis = [0,0,1])
    
    model_noB_folded.Add_Rod(r/a, [0,0,0], wp = F[1], gamma = F[2], axis = [0,0,1])
    model_noB_folded.Add_Rod(r/a, [np.sqrt(3)/2,0.5,0], wp = F[1], gamma = F[2], axis = [0,0,1])
    model_noB_folded.Add_Rod(r/a, [np.sqrt(3)/2,-0.5,0], wp = F[1], gamma = F[2], axis = [0,0,1])
    model_noB_folded.Add_Rod(r/a, [-np.sqrt(3)/2,0.5,0], wp = F[1], gamma = F[2], axis = [0,0,1])
    model_noB_folded.Add_Rod(r/a, [-np.sqrt(3)/2,-0.5,0], wp = F[1], gamma = F[2], axis = [0,0,1])

    ## Add sources for folded diagrams
    f_scales = np.array([[1,5], [0.2,0.2], [0.5,0.75], [1.5,3]])
    for i in range(4):
        r = np.clip(0.5*np.random.randn(3), -0.45, 0.45)
        loc = [model_B_folded.cell[0]*r[0], model_B_folded.cell[1]*r[1], model_B_folded.cell[2]*r[2]]
        model_B_folded.Add_Gauss_Source(F[0]*f_scales[i,0], F[0]*f_scales[i,1], Pol, loc, size, E)
        model_noB_folded.Add_Gauss_Source(F[0]*f_scales[i,0], F[0]*f_scales[i,1], Pol, loc, size, E)
    
    sim_B = model_B_folded.Get_Sim()
    sim_noB = model_noB_folded.Get_Sim()
    if os.path.exists(foldsavefile_B):
        os.remove(foldsavefile_B)
    if os.path.exists(foldsavefile_noB):
        os.remove(foldsavefile_noB)
    
    ## Simulate
    model_B.Bands2DTri(k_ps, fs = F[0], Pol = Pol, size = size, E = E,\
                   freq_range = [0,1.2], savefile = savefile_B, tol = 5*10**(-2))
    model_B.BandPlot(k_ps, np.array([2,10]), savefile_B, dims = True, dimlabel = 'GHz_zoomed',\
                     lattice = 'triangular')
    model_noB.Bands2DTri(k_ps, fs = F[0], Pol = Pol, size = size, E = E,\
                   freq_range = [0,1.2], savefile = savefile_B, tol = 5*10**(-2))
    model_noB.BandPlot(k_ps, np.array([2,10]), savefile_noB, dims = True, dimlabel = 'GHz_zoomed',\
                     lattice = 'triangular')
    
    freqs = sim_B.run_k_points(300, mp.interpolate(k_points, [mp.Vector3(0,0,0),\
                                   mp.Vector3(1./np.sqrt(3),1./3,0),\
                                   mp.Vector3(1./np.sqrt(3),0,0), mp.Vector3(0,0,0)]))
    model_B_folded.BandPlot(k_points, [0,1.2], freq_list = freqs, lattice = 'triangular',\
                   savefile = foldsavefile_B)
    
    freqs = sim_noB.run_k_points(300, mp.interpolate(k_points, [mp.Vector3(0,0,0),\
                                   mp.Vector3(1./np.sqrt(3),1./3,0),\
                                   mp.Vector3(1./np.sqrt(3),0,0), mp.Vector3(0,0,0)]))
    model_B_folded.BandPlot(k_points, [0,1.2], freq_list = freqs, lattice = 'triangular',\
                   savefile = foldsavefile_noB)
