"""
The module composed in this file is meant to serve as a set of tools to make
simulating the EM response of plasma-based optical devices a bit easier.
Jesse A Rodriguez, 11/11/2021
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import meep as mp
import os
import sys
import csv

###############################################################################
## Utility Functions and Globals
###############################################################################
c = 299792458
e = 1.60217662*10**(-19)
epso = 8.8541878128*10**(-12)
me = 9.1093837015*10**(-31)

def WP(n):
    """
    Function for calculating plasma frequency given density

    n: electron density in m^(-3)
    """
    return (n*e**2/me/epso)**(1/2)

def n_e(wp):
    """
    Function for calculating electron density given plasma frequency

    wp: plasma frequency in rad/s
    """
    return wp**2*me*epso/e**2

class Logger(object):
    """
    Object for piping Harminv outputs to both file and terminal
    """
    def __init__(self, savefile):
        self.terminal = sys.stdout
        self.log = open(savefile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def close(self):
        self.log.close()

    def flush(self):
        self.terminal.flush()

###############################################################################
## Plasma-focused meep class 
###############################################################################
class Plasmeep:
    def __init__(self, a, res, dpml, nx, ny, nz = 0, B = np.array([0,0,0]), units = "T"):
        """
        Initialize plasmeep object

        Args:
            a: dimensionalized unit length (e.g. 0.01 m) for translating to and
               from non-dimensionalized units
            res: pixels per a unit
            dpml: thickness of PML boundary in a units
            nx: number of a units in x-direction for active sim region
            ny: number of a units in y-direction for active sim region
            nz: number of a units in z-direction for active sim region
            B: externally applied b-field, vector with cyclotron frequency in a
               units as magnitude
        """
        self.a = a
        self.res = res
        self.cell = mp.Vector3(nx, ny, nz)
        self.pml = [mp.PML(thickness=dpml)]
        if units == "T":
            B = e*B/me/2/np.pi/c*a
        elif units == "Hz":
            B = B*a/c
        elif units == "rad/s":
            B = B/2/np.pi/c*a
        else:
            raise RuntimeError("Unrecognized units.")
        self.b = mp.Vector3(B[0],B[1],B[2])
        if B[0] != 0 or B[1] != 0 or B[2] != 0:
            self.is_magnetized = True
        else:
            self.is_magnetized = False
        self.geometry = [] # Objects in simulation domain
        self.sources = []

    ###########################################################################
    ## Simulation Domain Stuff
    ###########################################################################
    def Get_Med(self, eps, wp = 0, gamma = 0, PEC = False, PMC = False):
        """
        Return a meep medium object with the correct susceptibility.

        Args:
            eps: 0th order relative permittivity
            wp: plasma frequency in a units
            gamma: collision frequency in a units
            PEC/PMC: bools, is perfect electric/magnetic conductor?
        """
        if PEC:
            return mp.perfect_electric_conductor
        if PMC:
            return mp.perfect_magnetic_conductor

        if wp > 0:
            if self.is_magnetized:
                gyro_susc = [mp.GyrotropicDrudeSusceptibility(frequency=wp,\
                             gamma=gamma, sigma=1, bias=self.b)]
                return mp.Medium(epsilon=eps, mu=1,\
                                 E_susceptibilities=gyro_susc)
            else:
                drude = [mp.DrudeSusceptibility(frequency=wp, gamma=gamma)]
                return mp.Medium(epsilon=eps, mu=1,\
                                 E_susceptibilites=drude)
        else:
            return mp.Medium(epsilon=eps, mu=1)
    
    
    def Add_Rod(self, r, center, eps = 1, wp = 0, gamma = 0,\
                axis = [0,0,1], height = mp.inf):
        """
        Add a single rod with radius r at center to geometry.

        Args:
            r: radius of the rod in a units
            center: x,y,z coords of the rod center in a units
            eps: 0th order relative permittivity
            wp: plasma frequency in a units
            gamma: collision frequency in a units
            axis: 1-hot vector selecting cylinder axis
            height: cylinder height, for 3D designs.
        """
        medium = self.Get_Med(eps, wp, gamma)
        self.geometry.append(mp.Cylinder(r, material=medium,\
            center=mp.Vector3(center[0],center[1],center[2]),\
            axis=mp.Vector3(axis[0],axis[1],axis[2]), height=height))


    def Add_Bulb(self, r_bulb, center, wp = 0, gamma = 0,\
                 axis = [0,0,1], height = None, profile = 0):
        """
        Add a single plasma bulb to the geometry list

        Args:
            r_bulb: (inner, outer) radius of the rod in a units
            center: x,y,z coords of the rod center in a units
            eps: 0th order relative permittivity
            wp: plasma frequency in a units
            gamma: collision frequency in a units
            axis: 1-hot vector selecting cylinder axis
            height: cylinder height, for 3D designs.
        """
        quartz = self.Get_Med(3.8)
        if profile == 0:
            plasma = self.Get_Med(1, wp, gamma)
            vac = self.Get_Med(1)
            self.geometry.append(mp.Cylinder(r_bulb[1], material=quartz,\
                center=mp.Vector3(center[0],center[1],center[2]),\
                axis=mp.Vector3(axis[0],axis[1],axis[2]), height=height))
            self.geometry.append(mp.Cylinder(r_bulb[0], material=vac,\
                center=mp.Vector3(center[0],center[1],center[2]),\
                axis=mp.Vector3(axis[0],axis[1],axis[2]), height=height))
            self.geometry.append(mp.Cylinder(4.6*r_bulb[0]/6.5, material=plasma,\
                center=mp.Vector3(center[0],center[1],center[2]),\
                axis=mp.Vector3(axis[0],axis[1],axis[2]), height=height))

        elif profile == 6:
            self.geometry.append(mp.Cylinder(r_bulb[1], material=quartz,\
                center=mp.Vector3(center[0],center[1],center[2]),\
                axis=mp.Vector3(axis[0],axis[1],axis[2]), height=height))
            shells = self.res*r_bulb[0]
            for i in range(shells):
                r = r_bulb[0]*(1-i/shells)
                fp = wp*(4*(4.6/6.5)**2*(1-(r/r_bulb[0])**6)/3)**0.5
                shell = self.Get_Med(1, fp, gamma)
                self.geometry.append(mp.Cylinder(r, material=shell,\
                    center=mp.Vector3(center[0],center[1],center[2]),\
                    axis=mp.Vector3(axis[0],axis[1],axis[2]), height=height))

        else:
            raise RuntimeError("Density profile not implemented yet.")


    def Add_Block(self, center, size, eps = 1, wp = 0, gamma = 0, PEC = False,\
                  PMC = False):
        """
        Add a single rod with radius r and rel. permittivity eps to epsr.

        Args:
            center: x,y,z coords of the bottom left corner in a units
            size: z, y, and z size of the block in a units
            eps: relative permittivity of the block
            wp: plasma frequency in a units
            gamma: collision frequency in a units
        """
        medium = self.Get_Med(eps, wp, gamma, PEC, PMC)
        self.geometry.append(mp.Block(mp.Vector3(size[0],size[1],size[2]),\
                center=mp.Vector3(center[0],center[1],center[2]),\
                material=medium))


    def Add_Prism(self, vertices, height = mp.inf, eps = 1, wp = 0, gamma = 0):
        """
        Add a single rod with radius r and rel. permittivity eps to the static
        elems array.

        Args:
            vertices: x,y,z coords of the vertices of the prism in a units
            height: height of prism in axis direction
            axis: one-hot vector, direction of height dimension
            eps: relative permittivity of the block
            wp: plasma frequency in a units
            gamma: collision frequency in a units
        """
        medium = self.Get_Med(eps, wp, gamma)
        Vertices = []
        for i in range(vertices.shape[0]):
            Vertices.append(mp.Vector3(vertices[i,0], vertices[i,1],\
                                       vertices[i,2]))
        Axis = mp.Vector3(axis[0],axis[1],axis[2])
        self.geometry.append(mp.Prism(vertices=Vertices, height=height,\
                axis=Axis))


    def Rod_Array(self, r, xyz_start, a_l, rod_eps, axis, height = mp.inf):
        """
        Add a 2D cubic rod array to the domain.

        Args:
            r: radius of rods in a units
            xyz_start: coords of the bottom left of the array in a units
            axis: axis of the rods e.g. [0,0,1] for z
            rod_eps: np array of size (nrods_x, nrods_y, 3) giving the 
                     permittivity parameters of each of the rods. e.g. 
                     rod_eps[i,j,:] = [eps, fp, gamma]
            a_l: lattice spacing
            height: height of the rods in a units
        """
        for k in range(3):
            if axis[k] == 1 and k == 0:
                pass
            elif axis[k] == 1 and k == 1:
                pass
            elif  axis[k] == 1 and k == 2:
                for i in range(rod_eps.shape[0]):
                    for j in range(rod_eps.shape[1]):
                        x = xyz_start[0] + i*a_l
                        y = xyz_start[1] + j*a_l
                        self.Add_Rod(r, [x,y,xyz_start[2]], rod_eps[i,j,0],\
                                rod_eps[i,j,1], rod_eps[i,j,2], axis, height)
            else:
                pass


    def Rod_Array_Hex(self, r, xy_start, array_dims, bulbs = False,\
                      d_bulb = (0, 0), eps_bulb = 1):
        return


    ###########################################################################
    ## Simulation
    ###########################################################################
    def Add_Cont_Source(self, fs, Pol, center, size, E = True):
        """
        Add a source to the domain.

        Args:
            fs: Source frequency
            Pol: np array, unit vector pointing in the polarization direction,
                 also controls amplitude
            center: location of the center of the source
            size: size of source (can choose (0,0,0) for point-dipole)
            E: bool determining whether source is E or H.
        """
        for i in range(3):
            if np.abs(Pol[i]) > 0:
                if E:
                    if i == 0:
                        comp = mp.Ex
                    elif i == 1:
                        comp = mp.Ey
                    else:
                        comp = mp.Ez
                else:
                    if i == 0:
                        comp = mp.Hx
                    elif i == 1:
                        comp = mp.Hy
                    else:
                        comp = mp.Hz

                self.sources.append(mp.Source(\
                mp.ContinuousSource(frequency=fs, is_integrated=True),\
                component=comp,\
                center=mp.Vector3(center[0],center[1],center[2]),\
                size=mp.Vector3(size[0],size[1],size[2]), amplitude=Pol[i]))


    def Add_Gauss_Source(self, fs, df, Pol, center, size, E = True):
        """
        Add a source to the domain.

        Args:
            fs: Source frequency
            Pol: np array, unit vector pointing in the polarization direction,
                 also controls amplitude
            center: location of the center of the source
            size: size of source (can choose (0,0,0) for point-dipole)
        """
        for i in range(3):
            if np.abs(Pol[i]) > 0:
                if E:
                    if i == 0:
                        comp = mp.Ex
                    elif i == 1:
                        comp = mp.Ey
                    else:
                        comp = mp.Ez
                else:
                    if i == 0:
                        comp = mp.Hx
                    elif i == 1:
                        comp = mp.Hy
                    else:
                        comp = mp.Hz

                self.sources.append(mp.Source(\
                mp.GaussianSource(frequency=fs, fwidth=df),\
                component=comp,\
                center=mp.Vector3(center[0],center[1],center[2]),\
                size=mp.Vector3(size[0],size[1],size[2]), amplitude=Pol[i]))


    def Get_Sim(self):
        return mp.Simulation(cell_size=self.cell, boundary_layers=self.pml,\
                geometry=self.geometry, sources=self.sources,\
                resolution=self.res, default_material=mp.Medium(epsilon=1,mu=1))


    def Bands2D(self, k_points, fs, Pol, size = mp.Vector3(0,0,0), E = True,\
                freq_range = np.array([0,1.5]), savefile = 'bands.csv',\
                plot = True):
        """
        Create a 2-D band diagram given the domain as a single cell of a square
        lattice photonic crystal

        Args:
            k_points: number of points to interpolate between Gamma-X, X-M, and
                      M-Gamma. (k_points*3+4 total points will be simulated)
            fs: Source frequency, best set to 1, but could set to a small value
                if one is interested in resolving bands close to f=0, for ex.
            Pol: np array, unit vector pointing in the polarization direction,
                 also controls amplitude
            size: size of source (can choose (0,0,0) for point-dipole)
            E: bool determining whether source is E or H.
            freq_range: array, [min, max] of frequency range for band diagram
            savefile: Filename for bands data, must be .csv
        """
        f_scales = np.array([[1,5], [0.5,0.5], [1.5,1], [1.2,3]])
        for i in range(4):
            r = np.abs(np.random.randn(3)+0.5)
            loc = mp.Vector3(self.cell[0]*r[0], self.cell[1]*r[1],\
                             self.cell[2]*r[2])
            self.Add_Gauss_Source(fs*f_scales[i,0], fs*f_scales[i,1], Pol, loc,\
                                  size, E)

        if self.cell[2] != 0:
            raise RuntimeError('Can only generate band diagrams for 2D X-Y \
                                systems right now')

        sim = self.Get_Sim()
        if os.path.exists(savefile):
            os.remove(savefile)
        sys.stdout = Logger(savefile[:-4]+'_temp.txt')
        sim.run_k_points(300, mp.interpolate(k_points,\
                         [mp.Vector3(0,0,0), mp.Vector3(0.5,0,0),\
                         mp.Vector3(0.5,0.5,0), mp.Vector3(0,0,0)]))
        sys.stdout.flush()
        sys.stdout.close()

        with open(savefile, 'w') as sfile:
            with open(savefile[:-4]+'_temp.txt') as file:
                for line in file:
                    if line.startswith('freqs:'):
                        sfile.write(line[7:])
        os.remove(savefile[:-4]+'_temp.txt')

        if plot:
            self.BandPlot(k_points, freq_range, savefile)

        return


    ############################################################################
    ## Visualization
    ############################################################################
    def BandPlot(self, k_points, freq_range, savefile, dims = False,\
                 dimlabel = 'GHz'):
        """
        Plots band diagram given .csv output from Bands2D. See that function for
        arg descriptions.
        """
        freqs = np.array([])
        indices = np.array([])
        idx = 0
        with open(savefile) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                num = len(row[4:])
                freqs = np.append(freqs, np.array(row[4:]).astype(float))
                indices = np.append(indices, idx*np.ones(num))
                idx += 1

        fig, ax = plt.subplots(1,1,figsize=(8,6))

        xlabels = ['']*(3*(k_points+1)+1)
        xlabels[0] = '$\Gamma$'
        xlabels[k_points+1] = '$X$'
        xlabels[2*(k_points+1)] = '$M$'
        xlabels[3*(k_points+1)] = '$\Gamma$'

        if dims:
            freqs = self.Dimensionalize_Freq(freqs)/10**9
            ax.set_ylabel('$f$ ('+dimlabel+')', fontsize = 18)
        else:
            ax.set_ylabel('$\omega/(2\pi c/a)$', fontsize = 18)
        ax.tick_params(labelsize = 16)
        ax.set_xticks(range(3*(k_points+1)+1), minor=False)
        ax.set_xticklabels(xlabels, minor=False)
        
        ax.scatter(indices,freqs)
        ax.set_xlim([-(1/50)*(3*(k_points+1)),(51/50)*(3*(k_points+1))])
        ax.set_ylim(freq_range)

        fig.tight_layout()
        if dims:
            plt.savefig(savefile[:-4]+'_'+dimlabel+'.pdf', dpi=1500)
        else:
            plt.savefig(savefile[:-4]+'.pdf', dpi=1500)

        return


    ############################################################################
    ## Dimensionalization
    ############################################################################
    def Nondimensionalize_Freq(self, f):
        """
        Gives frequency in a units given f in Hz
        """
        f0 = c/self.a

        return f/f0

    
    def Dimensionalize_Freq(self, f):
        """
        Gives frequency in Hz given f in a units
        """
        f0 = c/self.a

        return f*f0


    def Nondimensionalize_B(self, B):
        """
        Gives cyclotron frequency in a units given B in T
        """
        fc = e*B/me/2/np.pi
        f0 = c/self.a

        return fc/f0


