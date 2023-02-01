"""
The module composed in this file is meant to serve as a platform for
designing plasma metamaterial devices and then optimizing the plasma density
of the elements composing the metamaterial to achieve a certain functionality.
It is built atop ceviche, an autograd compliant FDFD/FDTD EM simulation tool 
(https://github.com/fancompute/ceviche).
Jesse A Rodriguez, 09/15/2020
"""

import numpy as np
import autograd.numpy as npa
import matplotlib as mpl
mpl.rcParams['figure.dpi']=100
import matplotlib.pylab as plt
plt.rc('font', family='tahoma')
font = 18
plt.rc('xtick', labelsize=font)
plt.rc('ytick', labelsize=font)
from autograd.scipy.signal import convolve as conv
from skimage.draw import disk, rectangle, polygon
import ceviche
from ceviche import fdfd_ez, jacobian, fdfd_hz
from ceviche.optimizers import adam_optimize
from ceviche.modes import insert_mode
import collections
from functools import partial
from scipy.sparse import csr_matrix
import os
os.environ["KMP_WARNINGS"] = "FALSE"


###############################################################################
## Utility Functions and Globals
###############################################################################
c = 299792458
e = 1.60217662*10**(-19)
epso = 8.8541878128*10**(-12)
muo = 4*np.pi*10**(-7)
me = 9.1093837015*10**(-31)

def mode_overlap(E1, E2):
    """
    Defines an overlap integral between the sim field and desired field

    Args:
        E1, E2: Matrices with solved field values
    """
    return npa.abs(npa.sum(npa.multiply(npa.conj(E1),E2)))*1e6


def sum_overlap(E, mask):
    """
    Sums up masked values

    Args:
        E: Matrix with solved field values
        mask: One-hot mask designating where to sum
    """
    return npa.sum(npa.multiply(E,mask))


def Sp_ez(Hx, Hy, Ez):
    """
    Calculate Poynting vector from ez field sim results

    Args:
        Hx, Hy, Ez: Matrices with solved field values
    """
    S = np.zeros((Hx.shape[0], Hx.shape[1], 3)).astype(np.complex128)
    S[:,:,0] = -np.multiply(Ez, npa.conj(Hy))*1e6
    S[:,:,1] = np.multiply(Ez, npa.conj(Hx))*1e6
    return S


def Sp_hz(Ex, Ey, Hz):
    """
    Calculate Poynting vector from hz field sim results

    Args:
        Hx, Hy, Ez: Matrices with solved field values
    """
    S = np.zeros((Ex.shape[0], Ex.shape[1], 3)).astype(np.complex128)
    S[:,:,0] = np.multiply(Ey, npa.conj(Hz))*1e6
    S[:,:,1] = -np.multiply(Ex, npa.conj(Hz))*1e6
    return S


def Sp_u(Sp, u):
    """
    Get a matrix with the poynting component in the direction of u

    Args:
        Sp: Poynting vector matrix
        u: Unit vector in direction of interest
    """
    assert np.dot(u,u) == 1, "Second arg. is not a unit vector!"

    return np.dot(Sp, u)


def Sp_mag(Sp):
    """
    Get a matrix with the magnitude of the Poynting Vector

    Args:
        Sp: Poynting vector matrix
        u: Unit vector in direction of interest
    """
    S = np.zeros((Sp.shape[0],Sp.shape[1]))
    for i in range(Sp.shape[0]):
        for j in range(Sp.shape[1]):
            S[i,j] = np.sqrt(np.dot(np.real(Sp[i,j,:]),np.real(Sp[i,j,:])))
            
    return S


def Calc_Transmission(Fx, Fy, Fz, pol, denom, prb, u_p):
    """
    Calculate absolute transmission between a source and a probe

    Args:
        Fx,Fy,Fz: simulated fields
        pol: str, specifies 'ez' or 'hz' polarization
        denom: denominator for transmission calculation (in case of reflection)
        prb: probe mask
        u_p: normal unit vector to prb plane
    """
    if pol == 'ez':
        Sp = Sp_ez(Fx, Fy, Fz)
    elif pol == 'hz':
        Sp = Sp_hz(Fx, Fy, Fz)
    else:
        RuntimeError("Invalid polarization")

    Sp_p = npa.real(Sp_u(Sp, u_p))

    return sum_overlap(Sp_p, prb)/denom


def field_mag_int(E, mask):
    """
    Computes the integral of |E|^2 over a given mask

    Args:
        E: matrix containing field values
        mask: matrix where all entries are zero except the entries over which
        you would like to integrate
    """
    return npa.abs(npa.sum(npa.multiply(npa.multiply(npa.conj(E), E), mask)))*1e6


def callback_params(iteration, of_list, rho, dir = ''):
    """
    Callback function to save params at each iteration for a given run. This 
    function will overwrite data from a previous run, and will not overwrite
    all the files from a run which had more iterations.
    """
    np.savetxt(dir+'/iter_%d.csv' % iteration, rho, delimiter=',')


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

###############################################################################
## Inverse design of plasma metamaterials class 
###############################################################################
class PMMI:
    def __init__(self, a, res, nx, ny, dpml):
        """
        Initialize PMMI object

        Args:
            a: dimensionalized unit length (e.g. 0.01 m)
            res: pixels per a unit
            nx: number of a units in x-direction for active sim region
            ny: number of a units in y-direction for active sim region
            dpml: thickness of PML in a units
        """
        self.a = a
        self.res = res
        self.nx = nx
        self.ny = ny
        self.dpml = dpml
        self.dl = a/res #meters per pixel
        self.Npml = dpml*res #Number of pixels in PML layer
        self.Nx = nx*res #Number of pixels in x-direction
        self.Ny = ny*res #Number of pixels in y-direction
        self.epsr = np.ones((self.Nx, self.Ny)) #Initialize relative perm array
        self.design_region = np.zeros((self.Nx,self.Ny)) #Design region array
        self.train_elems = [] #Masks for trainable elements
        self.static_elems = np.ones((self.Nx, self.Ny)) #Array for static elements
                                                         #outside the training region.
        self.sources = {} #Empty dict to hold source arrays
        self.probes = {} #Empty dict to hold probe arrays
        self.rod_shells = 0 #Integer that tells you how many shells exist in a
                            #single trainable rod.
        self.fields = []

    ###########################################################################
    ## Design Region Stuff
    ###########################################################################
    def Design_Region(self, bot_left, extent):
        """
        Specify design region selector

        Args:
            bot_left: coords of bottom left ocrner of design region in a units
            extent: width and height of design region
        """
        X = int(round(bot_left[0]*self.res))
        Y = int(round(bot_left[1]*self.res))
        W = int(round(extent[0]*self.res))
        H = int(round(extent[1]*self.res))
        rr, cc = rectangle((X, Y), extent = (W, H), shape = self.epsr.shape)

        self.design_region[rr, cc] = 1


    def Add_Rod(self, r, center, eps):
        """
        Add a single rod with radius r and rel. permittivity eps to epsr.

        Args:
            r: radius of the rod in a units
            center: x,y coords of the rod center in a units
            eps: relative permittivity
        """
        R = int(round(r*self.res))
        X = int(round(center[0]*self.res))
        Y = int(round(center[1]*self.res))
        rr, cc = disk((X, Y), R, shape = self.epsr.shape)
        
        self.epsr[rr, cc] = eps


    def Add_Bulb(self, d_bulb, center, eps):
        """
        Add a single rod with radius r and rel. permittivity eps to epsr.

        Args:
            r: tuple in a units (inner bulb radius, outer bulb radius)
            center: x,y coords of the rod center in a units
            eps: relative permittivity of bulb
        """
        Ri = int(round(d_bulb[0]*self.res))
        Ro = int(round(d_bulb[1]*self.res))
        X = int(round(center[0]*self.res))
        Y = int(round(center[1]*self.res))
        rri, cci = disk((X, Y), Ri, shape = self.epsr.shape)
        rro, cco = disk((X, Y), Ro, shape = self.epsr.shape)
        
        self.design_region[rro, cco] += eps - 1
        self.design_region[rri, cci] += 1 - eps


    def Add_Rod_train(self, r, center):
        """
        Add a single rod with radius r to the trainable element array.

        Args:
            r: radius of the rod in a units
            center: x,y coords of the rod center in a units
        """
        R = int(round(r*self.res))
        X = int(round(center[0]*self.res))
        Y = int(round(center[1]*self.res))
        rr, cc = disk((X, Y), R, shape = self.epsr.shape)
        
        train_elem = np.zeros((self.epsr.shape))
        train_elem[rr, cc] = 1
        train_elem = csr_matrix(train_elem)
        self.train_elems.append(train_elem)


    def Add_Rod_train_radial_shells(self, r, center):
        """
        Add a single rod with radius r to the trainable element array with 
        radius-dependent permittivity (i.e. shells)

        Args:
            r: radius of the rod in a units
            center: x,y coords of the rod center in a units
        """
        R = int(round(r*self.res))
        X = int(round(center[0]*self.res))
        Y = int(round(center[1]*self.res))
        self.rod_shells = R

        for i in range(R):
            if i > 0:
                rri, cci = disk((X, Y), i, shape = self.epsr.shape)
            rro, cco = disk((X, Y), i+1, shape = self.epsr.shape)
        
            train_elem = np.zeros((self.epsr.shape))
            train_elem[rro, cco] = 1
            if i > 0:
                train_elem[rri, cci] = 0

            train_elem = csr_matrix(train_elem)
            self.train_elems.append(train_elem)


    def Add_Block(self, bot_left, extent, eps):
        """
        Add a single rod with radius r and rel. permittivity eps to epsr.

        Args:
            bot_left: x,y coords of the bottom left corner in a units
            extent: width and height of the block in a units
            eps: relative permittivity of the block
        """
        H = int(round(extent[1]*self.res))
        W = int(round(extent[0]*self.res))
        X = int(round(bot_left[0]*self.res))
        Y = int(round(bot_left[1]*self.res))
        rr, cc = rectangle((X, Y), extent = (W, H), shape = self.epsr.shape)

        self.epsr[rr, cc] = eps


    def Add_Block_static(self, bot_left, extent, eps):
        """
        Add a single rod with radius r and rel. permittivity eps to the static
        elems array.

        Args:
            bot_left: x,y coords of the bottom left corner in a units
            extent: width and height of the block in a units
            eps: relative permittivity of the block
        """
        H = int(round(extent[1]*self.res))
        W = int(round(extent[0]*self.res))
        X = int(round(bot_left[0]*self.res))
        Y = int(round(bot_left[1]*self.res))
        rr, cc = rectangle((X, Y), extent = (W, H), shape = self.epsr.shape)

        self.static_elems[rr, cc] = eps


    def Add_INFOMW_Horn(self, xy_open_cen, horn_dir, total_length,\
                           pol = 'TM', eps=-1000):
        """
        Add a microwave horn that matches our experimental facilities, in the
        TM orientation.

        Args:
            xy_open_cen: The location of the center of the aperture of the 
                         horn
            horn_dir: magnitude-1 vector giving horn direction
            total_length: how long the entire horn is including entrance wvg
            eps: relative permittivity
        """
        horn_dir_orth = np.array([horn_dir[1], -horn_dir[0]])
        
        if pol == 'TM':
            wall_thickness = 0.004/self.a
            width_open = 0.104/self.a
            width_base = 0.048/self.a
            depth = 0.089/self.a
            entrance_length = total_length - depth
        else:
            wall_thickness = 0.004/self.a
            width_open = 0.078/self.a
            width_base = 0.043/self.a
            depth = 0.089/self.a
            entrance_length = total_length - depth


        left_open = np.array([xy_open_cen + horn_dir_orth*width_open,\
            xy_open_cen + horn_dir_orth*(width_open-wall_thickness),\
            xy_open_cen + horn_dir_orth*(width_base-wall_thickness) -\
                horn_dir*depth,\
            xy_open_cen + horn_dir_orth*width_base - horn_dir*depth])

        left_entrance = np.array([xy_open_cen + horn_dir_orth*\
                (width_base-wall_thickness) - horn_dir*depth,\
            xy_open_cen + horn_dir_orth*width_base - horn_dir*depth,\
            xy_open_cen + horn_dir_orth*width_base - horn_dir*(depth +\
                entrance_length),\
            xy_open_cen + horn_dir_orth*(width_base-wall_thickness) -\
                horn_dir*(depth+entrance_length)])

        right_open = np.array([xy_open_cen - horn_dir_orth*width_open,\
            xy_open_cen - horn_dir_orth*(width_open-wall_thickness),\
            xy_open_cen - horn_dir_orth*(width_base-wall_thickness) -\
                horn_dir*depth,\
            xy_open_cen - horn_dir_orth*width_base - horn_dir*depth])

        right_entrance = np.array([xy_open_cen - horn_dir_orth*\
                (width_base-wall_thickness) - horn_dir*depth,\
            xy_open_cen - horn_dir_orth*width_base - horn_dir*depth,\
            xy_open_cen - horn_dir_orth*width_base - horn_dir*(depth +\
                entrance_length),\
            xy_open_cen - horn_dir_orth*(width_base-wall_thickness) -\
                horn_dir*(depth+entrance_length)])

        for i in range(4):
            for j in range(2):
                left_open[i,j] = int(round(left_open[i,j]*self.res))
                right_open[i,j] = int(round(right_open[i,j]*self.res))
                left_entrance[i,j] = int(round(left_entrance[i,j]*self.res))
                right_entrance[i,j] = int(round(right_entrance[i,j]*self.res))

        lor, loc = polygon(left_open[:,0], left_open[:,1], shape = self.epsr.shape)
        self.static_elems[lor,loc] = eps
        ror, roc = polygon(right_open[:,0], right_open[:,1], shape = self.epsr.shape)
        self.static_elems[ror,roc] =eps
        ler, lec = polygon(left_entrance[:,0], left_entrance[:,1], shape = self.epsr.shape)
        self.static_elems[ler,lec] = eps
        rer, rec = polygon(right_entrance[:,0], right_entrance[:,1], shape = self.epsr.shape)
        self.static_elems[rer,rec] = eps


    def Add_Source(self, xy_begin, xy_end, w, src_name, pol):
        """
        Add a source to the domain.

        Args:
            xy_begin: coords of the start of the source in a units (np.array)
            xy_end: coords of the end of the source in a units (np.array)
            w: Source frequency in c/a units
            src_name: string that serves as key for the source in the dict
            pol: string specifying the polarization of the source e.g. 'hz'
        """
        XY_beg = (np.rint(xy_begin*self.res)).astype(int)
        XY_end = (np.rint(xy_end*self.res)).astype(int)
        if XY_beg[0] == XY_end[0]:
            src_y = (np.arange(XY_beg[1],XY_end[1])).astype(int)
            src_x = XY_beg[0]*np.ones(src_y.shape, dtype=int)
        elif XY_beg[1] == XY_end[1]:
            src_x = (np.arange(XY_beg[0], XY_end[0])).astype(int)
            src_y = XY_beg[1]*np.ones(src_x.shape, dtype=int)
        else:
            raise RuntimeError("Source needs to be 1-D")

        omega = 2*np.pi*w*c/self.a
        src = insert_mode(omega, self.dl, src_x, src_y, self.epsr, m = 1)
        mask = np.zeros((self.Nx, self.Ny))
        for i in range(len(src_x)):
            mask[src_x[i], src_y[i]] = 1

        self.sources[src_name] = (src, omega, pol, mask)
        

    def Add_Probe(self, xy_begin, xy_end, w, prb_name, pol):
        """
        Add a probe to the domain.

        Args:
            xy_begin: coords of the start of the probe in a units (np.array)
            xy_end: coords of the end of the probe in a units (np.array)
            w: Probe frequency in c/a units
            src_name: string that serves as key for the probe in the dict
            pol: string specifying the polarization of the probe e.g. 'hz'
        """
        XY_beg = (np.rint(xy_begin*self.res)).astype(int)
        XY_end = (np.rint(xy_end*self.res)).astype(int)
        if XY_beg[0] == XY_end[0]:
            prb_y = (np.arange(XY_beg[1],XY_end[1])).astype(int)
            prb_x = XY_beg[0]*np.ones(prb_y.shape, dtype=int)
        elif XY_beg[1] == XY_end[1]:
            prb_x = (np.arange(XY_beg[0], XY_end[0])).astype(int)
            prb_y = XY_beg[1]*np.ones(prb_x.shape, dtype=int)
        else:
            raise RuntimeError("Probe needs to be 1-D")

        omega = 2*np.pi*w*c/self.a
        prb = insert_mode(omega, self.dl, prb_x, prb_y, self.epsr, m = 1)
        mask = np.zeros((self.Nx, self.Ny))
        for i in range(len(prb_x)):
            mask[prb_x[i], prb_y[i]] = 1

        self.probes[prb_name] = (prb, omega, pol, mask)


    def Rod_Array(self, r, xy_start, rod_eps, d_x = 1, d_y = 1):
        """
        Add a 2D rectangular rod array to epsr. All rods are spaced 1 a
        in the x and y direction by default.

        Args:
            r: radius of rods in a units
            x_start: x-coord of the bottom left of the array in a units
            y_start: y-coord of the bottom right of the array in a units
            rod_eps: np array of size nrods_x by nrods_y giving the relative 
                     permittivity of each of the rods
            d_x: lattice spacing in x-direction in a units
            d_y: lattice spacing in y-direction in a units
        """
        for i in range(rod_eps.shape[0]):
            for j in range(rod_eps.shape[1]):
                x = xy_start[0] + i*d_x
                y = xy_start[1] + j*d_y
                rod_e = rod_eps[i,j]
                if not isinstance(rod_eps[i,j], np.float64):
                    rod_e = rod_eps[i,j]._value
                self.Add_Rod(r, (x, y), rod_e)


    def Rod_Array_train(self, r, xy_start, array_dims, d_x = 1, d_y = 1,\
                        bulbs = False, d_bulb = (0, 0), eps_bulb = 1,\
                        uniform = True):
        """
        Add a 2D rectangular rod array to the train elems. All rods are spaced 1 a
        in the x and y direction by default.

        Args:
            r: radius of rods in a units
            x_start: x-coord of the bottom left of the array in a units
            y_start: y-coord of the bottom right of the array in a units
            d_x: lattice spacing in x-direction in a units
            d_y: lattice spacing in y-direction in a units
            bulbs: bool specifying if dicharge glass is included
            d_bulb: tuple in a units (inner radius of glass, outer radius of glass)
            eps_bulb: rel. perm. of bulb glass
        """
        for i in range(array_dims[0]):
            for j in range(array_dims[1]):
                x = xy_start[0] + i*d_x
                y = xy_start[1] + j*d_y
                if uniform:
                    self.Add_Rod_train(r, (x, y))
                else:
                    self.Add_Rod_train_radial_shells(r, (x, y))
                if bulbs:
                    self.Add_Bulb(d_bulb, (x, y), eps_bulb)


    def Rod_Array_Hex_train(self, r, xy_start, array_dims, bulbs = False,\
                            d_bulb = (0, 0), eps_bulb = 1):
        """
        Add a 2D hexagonal rod array to the train elems. All rods are spaced 1 a
        in the x and y direction by default.

        Args:
            r: radius of rods in a units
            x_start: x-coord of the bottom left of the array in a units
            y_start: y-coord of the bottom right of the array in a units
            d_x: lattice spacing in x-direction in a units
            d_y: lattice spacing in y-direction in a units
            bulbs: bool specifying if dicharge glass is included
            d_bulb: tuple in a units (inner radius of glass, outer radius of glass)
            eps_bulb: rel. perm. of bulb glass
        """
        for i in range(array_dims[0]):
            if i%2 == 0:
                y_offset = 0
                num_rods = array_dims[1]
            if i%2 == 1:
                y_offset = 0.5
                num_rods = array_dims[1]
            for j in range(num_rods):
                x = xy_start[0] + i*np.sqrt(3)/2
                y = y_offset + xy_start[1] + j
                self.Add_Rod_train(r, (x, y))
                if bulbs:
                    self.Add_Bulb(d_bulb, (x, y), eps_bulb)


    ###########################################################################
    ## Plotting Functions
    ###########################################################################
    def Viz_Sim_abs(self, src_names, savepath):
        """
        Solve and visualize an static simulation with certain sources active
        
        Args:
            src_names: list of strings that indicate which sources you'd like to simulate
        """
        fig, ax = plt.subplots(1, len(src_names)+1, constrained_layout=False,\
                               figsize=(9*len(src_names),4))
        for i in range(len(src_names)):
            if i > 0:
                if src_names[i] != src_names[i-1]:
                    self.Clear_fields()
            if self.sources[src_names[i]][2] == 'hz':
                simulation = fdfd_hz(self.sources[src_names[i]][1], self.dl, self.epsr,\
                            [self.Npml, self.Npml])
                Ex, Ey, Hz = simulation.solve(self.sources[src_names[i]][0])
                cbar = plt.colorbar(ax[i].imshow(np.abs(Hz.T), cmap='magma'), ax=ax[i])
                cbar.set_ticks([])
                cbar.ax.set_ylabel('H-Field Magnitude', fontsize=font)
                ax[i].axes.xaxis.set_visible(False)
                ax[i].axes.yaxis.set_visible(False)
                #ax[i].contour(epsr_opt.T, 2, colors='w', alpha=0.5)
            elif self.sources[src_names[i]][2] == 'ez':
                simulation = fdfd_ez(self.sources[src_names[i]][1], self.dl, self.epsr,\
                            [self.Npml, self.Npml])
                Hx, Hy, Ez = simulation.solve(self.sources[src_names[i]][0])
                cbar = plt.colorbar(ax[i].imshow(np.abs(Ez.T), cmap='magma'), ax=ax[i])
                cbar.set_ticks([])
                cbar.ax.set_ylabel('E-Field Magnitude', fontsize=font)
                ax[i].axes.xaxis.set_visible(False)
                ax[i].axes.yaxis.set_visible(False)
                #ax[i].contour(epsr_opt.T, 2, colors='w', alpha=0.5)
            else:
                raise RuntimeError('The polarization associated with this source is\
                                    not valid.')
                
        cbar = plt.colorbar(ax[len(src_names)].imshow(self.epsr.T, cmap='RdGy',\
                            vmin = np.min(self.epsr), vmax = np.max(self.epsr)),\
                            ax=ax[len(src_names)])
        cbar.ax.set_ylabel('Relative Permittivity', fontsize=font)
        ax[len(src_names)].axes.xaxis.set_visible(False)
        ax[len(src_names)].axes.yaxis.set_visible(False)
        plt.savefig(savepath, dpi=1500)
        plt.show()


    def Viz_Sim_Fields(self, src_names, savepath, perm = True):
        """
        Solve and visualize a static simulation with certain sources active
        
        Args:
            src_names: list of strings that indicate which sources you'd like to simulate
        """
        if perm:
            fig, ax = plt.subplots(1, len(src_names)+1, constrained_layout=False,\
                                   figsize=(4.5*(len(src_names)+1),4))
        else:
            fig, ax = plt.subplots(1, len(src_names), constrained_layout=False,\
                                   figsize=(4.5*len(src_names),3))

        for i in range(len(src_names)):
            if i > 0:
                if src_names[i] != src_names[i-1]:
                    self.Clear_fields()
            if self.sources[src_names[i]][2] == 'hz':
                if self.fields == []:
                    simulation = fdfd_hz(self.sources[src_names[i]][1], self.dl,\
                                         self.epsr, [self.Npml, self.Npml])
                    Ex, Ey, Hz = simulation.solve(self.sources[src_names[i]][0])
                    self.fields = [Ex, Ey, Hz]
                    
                cbar = plt.colorbar(ax[i].imshow(np.real(self.fields[2].T), cmap='RdBu'), ax=ax[i])
                cbar.set_ticks([])
                cbar.ax.set_ylabel('H-Field', fontsize=font)
                ax[i].axes.xaxis.set_visible(False)
                ax[i].axes.yaxis.set_visible(False)
                #ax[i].contour(epsr_opt.T, 1, colors='k', alpha=0.5)
            elif self.sources[src_names[i]][2] == 'ez':
                if self.fields == []:
                    simulation = fdfd_hz(self.sources[src_names[i]][1], self.dl,\
                                         self.epsr, [self.Npml, self.Npml])
                    Hx, Hy, Ez = simulation.solve(self.sources[src_names[i]][0])
                    self.fields = [Hx, Hy, Ez]
                    
                cbar = plt.colorbar(ax[i].imshow(np.real(self.fields[2].T), cmap='RdBu'), ax=ax[i])
                cbar.set_ticks([])
                cbar.ax.set_ylabel('E-Field', fontsize=font)
                ax[i].axes.xaxis.set_visible(False)
                ax[i].axes.yaxis.set_visible(False)
                #ax[i].contour(epsr_opt.T, 1, colors='k', alpha=0.5)
            else:
                raise RuntimeError('The polarization associated with this source is\
                                   not valid.')
        
        if perm:
            cbar = plt.colorbar(ax[len(src_names)].imshow(self.epsr.T, cmap='RdGy',\
                        vmin = np.min(self.epsr), vmax = np.max(self.epsr)),\
                        ax=ax[len(src_names)])
            cbar.ax.set_ylabel('Relative Permittivity', fontsize=font)
            ax[len(src_names)].axes.xaxis.set_visible(False)
            ax[len(src_names)].axes.yaxis.set_visible(False)
            
        plt.savefig(savepath, dpi=1500)
        plt.show()


    def Viz_Sim_abs_opt(self, rho, src_names, savepath, bounds = [], plasma = False,\
                        show = True, mult = False, wp_max = 0, gamma = 0, uniform = True,\
                        perturb = 0, amp = 1):
        """
        Solve and visualize an optimized simulation with certain sources active
        
        Args:
            rho: optimal parameters
            src_names: list of strings that indicate which sources you'd like to simulate
            savepath = save path
            bounds: Upper and lower bounds for parameters
            plasma: bool specifying if params map to wp
            show: bool determining if the plot is shown
            mult: bool determining if multiple sources are activated at once
            perturb: sigma value for gaussian perturbation of rods. 0.05 would
                     result in random perturbations of each element of ~5%.
            amp: turn up fields outside of input port
        """
        fig, ax = plt.subplots(1, len(src_names)+1, constrained_layout=False,\
                               figsize=(9*len(src_names),4))

        if perturb > 0:
            pmat = self.Pmat(rho, perturb)
        else:
            pmat = np.empty(0)
        for i in range(len(src_names)):
            if i > 0:
                if src_names[i] != src_names[i-1]:
                    self.Clear_fields()
            if mult:
                w_src = self.sources[src_names[i][0]][1]*self.a/2/np.pi/c
                pol = self.sources[src_names[i][0]][2]
                w = self.sources[src_names[i][0]][1]
                src = self.sources[src_names[i][0]][0]
                for j in range(len(src_names[i])-1):
                    src = src + self.sources[src_names[i][j+1]][0]
            else:
                w_src = self.sources[src_names[i]][1]*self.a/2/np.pi/c
                pol = self.sources[src_names[i]][2]
                w = self.sources[src_names[i]][1]
                src = self.sources[src_names[i]][0]

            if plasma:
                epsr_opt = self.Rho_Parameterization_wp(rho, w_src, wp_max, gamma,\
                                                        uniform, pmat = pmat)
            else:
                epsr_opt = self.Rho_Parameterization(rho, bounds, pmat = pmat)

            if pol == 'hz':
                if self.fields == []:
                    simulation = fdfd_hz(w, self.dl, epsr_opt, [self.Npml, self.Npml])
                    Ex, Ey, Hz = simulation.solve(src)
                    self.fields = [Ex, Ey, Hz]
                
                max_field = np.max(np.abs(self.fields[2]))
                cbar = plt.colorbar(ax[i].imshow(amp*np.abs(self.fields[2].T), cmap='magma',\
                                    vmin = np.min(amp*np.abs(self.fields[2])),\
                                    vmax = max_field), ax=ax[i])
                cbar.set_ticks([])
                cbar.ax.set_ylabel('H-Field Magnitude', fontsize=font)
                ax[i].axes.xaxis.set_visible(False)
                ax[i].axes.yaxis.set_visible(False)
            elif pol == 'ez':
                if self.fields == []:
                    simulation = fdfd_ez(w, self.dl, epsr_opt, [self.Npml, self.Npml])
                    Hx, Hy, Ez = simulation.solve(src)
                    self.fields = [Hx, Hy, Ez]

                max_field = np.max(np.abs(self.fields[2]))
                cbar = plt.colorbar(ax[i].imshow(amp*np.abs(self.fields[2].T), cmap='magma',\
                                    vmin = np.min(amp*np.abs(self.fields[2])),\
                                    vmax = max_field), ax=ax[i])
                cbar.set_ticks([])
                cbar.ax.set_ylabel('E-Field Magnitude', fontsize=font)
                ax[i].axes.xaxis.set_visible(False)
                ax[i].axes.yaxis.set_visible(False)
            else:
                raise RuntimeError('The polarization associated with this source is\
                                    not valid.')

        cbar = plt.colorbar(ax[len(src_names)].imshow(np.real(epsr_opt).T, cmap='RdGy',\
                            vmin = np.min(self.design_region*np.real(epsr_opt)),\
                            vmax = np.max(np.real(epsr_opt))), ax=ax[len(src_names)])
        cbar.ax.set_ylabel('Relative Permittivity', fontsize=font)
        ax[len(src_names)].axes.xaxis.set_visible(False)
        ax[len(src_names)].axes.yaxis.set_visible(False)
        plt.savefig(savepath, dpi=1500)

        if show:
            plt.show()


    def Viz_Sim_Poynting(self, src_names, u, savepath, perm = True):
        """
        Solve and visualize a static simulation with certain sources active
        
        Args:
            src_names: list of strings that indicate which sources you'd like to simulate
        """
        if perm:
            fig, ax = plt.subplots(1, len(src_names)+1, constrained_layout=False,\
                                   figsize=(8*len(src_names),2))

            for i in range(len(src_names)):
                if i > 0:
                    if src_names[i] != src_names[i-1]:
                        self.Clear_fields()
                if self.sources[src_names[i]][2] == 'hz':
                    simulation = fdfd_hz(self.sources[src_names[i]][1], self.dl, self.epsr,\
                                [self.Npml, self.Npml])
                    Ex, Ey, Hz = simulation.solve(self.sources[src_names[i]][0])
                    S = Sp_u(Sp_hz(Ex, Ey, Hz), u)
                    cbar = plt.colorbar(ax[i].imshow(np.real(S.T), cmap='RdBu'), ax=ax[i])
                    cbar.set_ticks([])
                    cbar.ax.set_ylabel('Poynting Flux', fontsize=font)
                    ax[i].axes.xaxis.set_visible(False)
                    ax[i].axes.yaxis.set_visible(False)
                    #ax[i].contour(epsr_opt.T, 1, colors='k', alpha=0.5)
                elif self.sources[src_names[i]][2] == 'ez':
                    simulation = fdfd_ez(self.sources[src_names[i]][1], self.dl, self.epsr,\
                                [self.Npml, self.Npml])
                    Hx, Hy, Ez = simulation.solve(self.sources[src_names[i]][0])
                    S = Sp_u(Sp_ez(Hx, Hy, Ez), u)
                    #print(np.real(S)[140,50])
                    #print(self.sources[src_names[i]])
                    cbar = plt.colorbar(ax[i].imshow(np.real(S.T), cmap='RdBu'), ax=ax[i])
                    cbar.set_ticks([])
                    cbar.ax.set_ylabel('Poynting Flux', fontsize=font)
                    ax[i].axes.xaxis.set_visible(False)
                    ax[i].axes.yaxis.set_visible(False)
                    #ax[i].contour(epsr_opt.T, 1, colors='k', alpha=0.5)
                else:
                    raise RuntimeError('The polarization associated with this source is\
                                        not valid.')
                
            cbar = plt.colorbar(ax[len(src_names)].imshow(self.epsr.T, cmap='RdGy',\
                            vmin = np.min(self.epsr), vmax = np.max(self.epsr)),\
                            ax=ax[len(src_names)])
            cbar.ax.set_ylabel('Relative Permittivity', fontsize=font)
            ax[len(src_names)].axes.xaxis.set_visible(False)
            ax[len(src_names)].axes.yaxis.set_visible(False)
            plt.savefig(savepath, dpi=1500)
            plt.show()

        else:
            fig, ax = plt.subplots(1, len(src_names), constrained_layout=False,\
                                   figsize=(4.5*len(src_names),3))

            if self.sources[src_names[0]][2] == 'hz':
                simulation = fdfd_hz(self.sources[src_names[0]][1], self.dl, self.epsr,\
                            [self.Npml, self.Npml])
                Ex, Ey, Hz = simulation.solve(self.sources[src_names[0]][0])
                S = Sp_u(Sp_hz(Ex, Ey, Hz), u)
                cbar = plt.colorbar(ax.imshow(np.real(S.T), cmap='RdBu'), ax=ax)
                cbar.set_ticks([])
                cbar.ax.set_ylabel('Poynting Flux', fontsize=font)
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
                #ax[i].contour(epsr_opt.T, 1, colors='k', alpha=0.5)
            elif self.sources[src_names[0]][2] == 'ez':
                simulation = fdfd_ez(self.sources[src_names[0]][1], self.dl, self.epsr,\
                            [self.Npml, self.Npml])
                Hx, Hy, Ez = simulation.solve(self.sources[src_names[0]][0])
                S = Sp_u(Sp_ez(Hx, Hy, Ez), u)
                cbar = plt.colorbar(ax.imshow(np.real(S.T), cmap='RdBu'), ax=ax)
                cbar.set_ticks([])
                cbar.ax.set_ylabel('Poynting Flux', fontsize=font)
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
                #ax[i].contour(epsr_opt.T, 1, colors='k', alpha=0.5)
            else:
                raise RuntimeError('The polarization associated with this source is\
                                        not valid.')

            plt.savefig(savepath, dpi=1500)
            plt.show()



    def Viz_Sim_Poynting_opt(self, rho, u, src_names, savepath, bounds = [], plasma = False,\
                           show = True, mult = False, wp_max = 0, gamma = 0, uniform = True,\
                           perturb = 0):
        """
        Solve and visualize an optimized simulation with certain sources active
        
        Args:
            rho: optimal parameters
            src_names: list of strings that indicate which sources you'd like to simulate
            savepath = save path
            bounds: Upper and lower bounds for parameters
            plasma: bool specifying if params map to wp
            show: bool determining if the plot is shown
            mult: bool determining if multiple sources are activated at once
            perturb: sigma value for gaussian perturbation of rods. 0.05 would
                     result in random perturbations of each element of ~5%.
        """
        fig, ax = plt.subplots(1, len(src_names)+1, constrained_layout=False,\
                               figsize=(9*len(src_names),4))
        if perturb > 0:
            pmat = self.Pmat(rho, perturb)
        else:
            pmat = np.empty(1)
        for i in range(len(src_names)):
            if i > 0:
                if src_names[i] != src_names[i-1]:
                    self.Clear_fields()
            if mult:
                w_src = self.sources[src_names[i][0]][1]*self.a/2/np.pi/c
                pol = self.sources[src_names[i][0]][2]
                w = self.sources[src_names[i][0]][1]
                src = self.sources[src_names[i][0]][0]
                for j in range(len(src_names[i])-1):
                    src = src + self.sources[src_names[i][j+1]][0]
            else:
                w_src = self.sources[src_names[i]][1]*self.a/2/np.pi/c
                pol = self.sources[src_names[i]][2]
                w = self.sources[src_names[i]][1]
                src = self.sources[src_names[i]][0]

            if plasma:
                epsr_opt = self.Rho_Parameterization_wp(rho, w_src, wp_max, gamma,\
                                                        uniform, pmat = pmat)
            else:
                epsr_opt = self.Rho_Parameterization(rho, bounds, pmat = pmat)

            if pol == 'hz':
                if self.fields == []:
                    simulation = fdfd_hz(w, self.dl, epsr_opt, [self.Npml, self.Npml])
                    Ex, Ey, Hz = simulation.solve(src)
                    self.fields = [Ex, Ey, Hz]

                S = Sp_u(Sp_hz(self.fields[0], self.fields[1], self.fields[2]), u)
                cbar = plt.colorbar(ax[i].imshow(np.real(S).T, cmap='RdBu'), ax=ax[i])
                cbar.set_ticks([])
                cbar.ax.set_ylabel('Poynting Flux', fontsize=font)
                ax[i].axes.xaxis.set_visible(False)
                ax[i].axes.yaxis.set_visible(False)
                #ax[i].contour(epsr_opt.T, 1, colors='k', alpha=0.5)
            elif pol == 'ez':
                if self.fields == []:
                    simulation = fdfd_ez(w, self.dl, epsr_opt, [self.Npml, self.Npml])
                    Hx, Hy, Ez = simulation.solve(src)
                    self.fields = [Hx, Hy, Ez]
 
                S = Sp_u(Sp_hz(self.fields[0], self.fields[1], self.fields[2]), u)
                cbar = plt.colorbar(ax[i].imshow(np.real(S).T, cmap='RdBu'), ax=ax[i])
                cbar.set_ticks([])
                cbar.ax.set_ylabel('Poynting Flux', fontsize=font)
                ax[i].axes.xaxis.set_visible(False)
                ax[i].axes.yaxis.set_visible(False)
                #ax[i].contour(epsr_opt.T, 1, colors='k', alpha=0.5)
            else:
                raise RuntimeError('The polarization associated with this source is\
                                    not valid.')

        cbar = plt.colorbar(ax[len(src_names)].imshow(np.real(epsr_opt).T, cmap='RdGy',\
                            vmin = np.min(self.design_region*np.real(epsr_opt)),\
                            vmax = np.max(np.real(epsr_opt))), ax=ax[len(src_names)])
        cbar.ax.set_ylabel('Relative Permittivity', fontsize=font)
        ax[len(src_names)].axes.xaxis.set_visible(False)
        ax[len(src_names)].axes.yaxis.set_visible(False)
        plt.savefig(savepath, dpi=1500)

        if show:
            plt.show()


    def Viz_Sim_fields_opt(self, rho, src_names, savepath, bounds = [], plasma = False,\
                           show = True, mult = False, wp_max = 0, gamma = 0, uniform = True,\
                           perturb = 0):
        """
        Solve and visualize an optimized simulation with certain sources active
        
        Args:
            rho: optimal parameters
            src_names: list of strings that indicate which sources you'd like to simulate
            savepath = save path
            bounds: Upper and lower bounds for parameters
            plasma: bool specifying if params map to wp
            show: bool determining if the plot is shown
            mult: bool determining if multiple sources are activated at once
            perturb: sigma value for gaussian perturbation of rods. 0.05 would
                     result in random perturbations of each element of ~5%.
        """
        fig, ax = plt.subplots(1, len(src_names)+1, constrained_layout=False,\
                               figsize=(4.5*(len(src_names)+1),4))
        if perturb > 0:
            pmat = self.Pmat(rho, perturb)
        else:
            pmat = np.empty(1)
        for i in range(len(src_names)):
            if i > 0:
                if src_names[i] != src_names[i-1]:
                    self.Clear_fields()
            if mult:
                w_src = self.sources[src_names[i][0]][1]*self.a/2/np.pi/c
                pol = self.sources[src_names[i][0]][2]
                w = self.sources[src_names[i][0]][1]
                src = self.sources[src_names[i][0]][0]
                for j in range(len(src_names[i])-1):
                    src = src + self.sources[src_names[i][j+1]][0]
            else:
                w_src = self.sources[src_names[i]][1]*self.a/2/np.pi/c
                pol = self.sources[src_names[i]][2]
                w = self.sources[src_names[i]][1]
                src = self.sources[src_names[i]][0]

            if plasma:
                epsr_opt = self.Rho_Parameterization_wp(rho, w_src, wp_max, gamma,\
                                                        uniform, pmat = pmat)
            else:
                epsr_opt = self.Rho_Parameterization(rho, bounds, pmat = pmat)

            if pol == 'hz':
                if self.fields == []:
                    simulation = fdfd_hz(w, self.dl, epsr_opt, [self.Npml, self.Npml])
                    Ex, Ey, Hz = simulation.solve(src)
                    self.fields = [Ex, Ey, Hz]
                cbar = plt.colorbar(ax[i].imshow(np.real(self.fields[2]).T, cmap='RdBu'), ax=ax[i])
                cbar.set_ticks([])
                cbar.ax.set_ylabel('H-Field', fontsize=font)
                ax[i].axes.xaxis.set_visible(False)
                ax[i].axes.yaxis.set_visible(False)
                #ax[i].contour(epsr_opt.T, 1, colors='k', alpha=0.5)
            elif pol == 'ez':
                if self.fields == []:
                    simulation = fdfd_ez(w, self.dl, epsr_opt, [self.Npml, self.Npml])
                    Hx, Hy, Ez = simulation.solve(src)
                    self.fields = [Hx, Hy, Ez]
                cbar = plt.colorbar(ax[i].imshow(np.real(self.fields[2]).T, cmap='RdBu'), ax=ax[i])
                cbar.set_ticks([])
                cbar.ax.set_ylabel('E-Field', fontsize=font)
                ax[i].axes.xaxis.set_visible(False)
                ax[i].axes.yaxis.set_visible(False)
                #ax[i].contour(epsr_opt.T, 1, colors='k', alpha=0.5)
            else:
                raise RuntimeError('The polarization associated with this source is\
                                    not valid.')

        cbar = plt.colorbar(ax[len(src_names)].imshow(np.real(epsr_opt).T, cmap='RdGy',\
                            vmin = np.min(self.design_region*np.real(epsr_opt)),\
                            vmax = np.max(np.real(epsr_opt))), ax=ax[len(src_names)])
        cbar.ax.set_ylabel('Relative Permittivity', fontsize=font)
        ax[len(src_names)].axes.xaxis.set_visible(False)
        ax[len(src_names)].axes.yaxis.set_visible(False)
        plt.savefig(savepath, dpi=1500)

        if show:
            plt.show()

        return (simulation, ax)


    def Viz_Obj(self, obj, savepath, show = True):
        """
        Plot evolution of objective function throughout the training

        Args:
            obj: Objective array returned by the Adam Optimizer
            savepath: Save path (string)
        """
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(obj, linewidth=3.0)
        ax.set_xlabel('Training Epoch', fontsize = font)
        ax.set_ylabel('Objective', fontsize = font)
        plt.savefig(savepath)
        if show:
            plt.show()

        return ax


    def Transmission_Spect(self, src_begin, src_end, u_s, prb_begin, prb_end,\
                           u_p, pol, w, savepath, dB = True, show = True):
        """
        Calculate a transmission spectrum for the static domain

        Args:
            src_begin/end: Location start and end of source
            u_s: normal unit vector to src plane
            prb: Name of a probe object, again with correct location and pol
            u_p: normal unit vector to prb plane
            w: 1-D numpy array of non-dimensionalized frequencies at which 
               to calculate transmission.
            savepath: save path
            dB: bool, True if transmission is to be given in dB
            show: bool, True to show plot
        """
        Trans = np.zeros((w.shape[0],))
        idx_miss = []
        for i in range(w.shape[0]):
            print("solving w =",w[i])
            try:
                self.Add_Source(src_begin, src_end, w[i], 'sim', pol)
                self.Add_Source(src_begin + 2*u_s[0:2]/self.res,\
                            src_end + 2*u_s[0:2]/self.res, w[i], 'mask', pol)
                self.Add_Probe(prb_begin, prb_end, w[i], 'mask', pol)
                if pol == 'ez':
                    sim = fdfd_ez(self.sources['sim'][1], self.dl, self.epsr,\
                                  [self.Npml, self.Npml])
                elif pol == 'hz':
                    sim = fdfd_hz(self.sources['sim'][1], self.dl, self.epsr,\
                                  [self.Npml, self.Npml])
                else:
                    RuntimeError("Invalid polarization.")
                #source_popped = self.sources.pop('sim')
                #print(source_popped)
                Fx, Fy, Fz = sim.solve(self.sources.pop('sim')[0])
                Trans[i] = 10*np.log10(Calc_Transmission(Fx, Fy, Fz, pol,\
                                   self.sources.pop('mask')[3], u_s,\
                                   self.probes.pop('mask')[3], u_p))
                #print(Trans[i])
            except RuntimeError:
                if i == w.shape[0]-1:
                    Trans[i]  = Trans[i-1]
                else: 
                    idx_miss.append(i)
                print("Something went wrong with w =", w[i])
                print("Interpolating if possible, otherwise, using last calculated value.")

        for i in idx_miss:
            if i >= 1:
                Trans[i] = (Trans[i-1]+Trans[i+1])/2
            else:
                Trans[i] = Trans[i+1]

        fig = plt.figure()
        ax = plt.axes()
        ax.plot(w*c/self.a, Trans, linewidth=3.0)
        ax.set_xlabel('Frequency (Hz)', fontsize=font)
        ax.set_ylabel('Transmission (dB)', fontsize=font)
        plt.savefig(savepath)
        if show:
            plt.show()

        return ax
    
    
    ###########################################################################
    ## Simulation Tools
    ###########################################################################
    def Clear_fields(self):
        """
        Clear calculated fields
        """
        self.fields = []
        
        
    def FDTD_fields_Opt(self, rho, T, src_names, savepath, plasma = False,\
                        plot = True, wp_max = 0, gamma = 0, uniform = True,\
                        bounds = []):
        if plot:
            fig, ax = plt.subplots(1,len(src_names)+1,constrained_layout=False,\
                                   figsize=(4.5*(len(src_names)+1),4))
            
        for i in range(len(src_names)):
            w_src = self.sources[src_names[i]][1]*self.a/2/np.pi/c
            pol = self.sources[src_names[i]][2]
            w = self.sources[src_names[i]][1]
            src = self.sources[src_names[i]][0]

            if plasma:
                epsr_opt = self.Rho_Parameterization_wp(rho, w_src, wp_max, gamma,\
                                                        uniform)
            else:
                epsr_opt = self.Rho_Parameterization(rho, bounds)

            if pol == 'hz':
                if self.fields == []:
                    simulation = fdfd_hz(w, self.dl, epsr_opt, [self.Npml, self.Npml])
                    Ex, Ey, Hz = simulation.solve(src)
                    self.fields = [Ex, Ey, Hz]
                S.append(10*np.log10(Calc_Transmission(self.fields[0], self.fields[1],\
                                   self.fields[2], pol, self.sources.pop('mask')[3], u_s,\
                                   self.probes.pop('mask')[3], u_p)))
                if plot:
                    cbar = plt.colorbar(ax[i].imshow(np.real(self.fields[2]).T, cmap='magma'), ax=ax[i])
                    cbar.set_ticks([])
                    cbar.ax.set_ylabel('H-Field', fontsize=font)
                    ax[i].axes.xaxis.set_visible(False)
                    ax[i].axes.yaxis.set_visible(False)
            elif pol == 'ez':
                if self.fields == []:
                    simulation = fdfd_ez(w, self.dl, epsr_opt, [self.Npml, self.Npml])
                    Hx, Hy, Ez = simulation.solve(src)
                    self.fields = [Hx, Hy, Ez]
                S.append(10*np.log10(Calc_Transmission(self.fields[0], self.fields[1],\
                                   self.fields[2], pol, self.sources.pop('mask')[3], u_s,\
                                   self.probes.pop('mask')[3], u_p)))
                if plot:
                    cbar = plt.colorbar(ax[i].imshow(np.real(self.fields[2]).T, cmap='magma'), ax=ax[i])
                    cbar.set_ticks([])
                    cbar.ax.set_ylabel('E-Field', fontsize=font)
                    ax[i].axes.xaxis.set_visible(False)
                    ax[i].axes.yaxis.set_visible(False)
            else:
                raise RuntimeError('The polarization associated with this source is\
                               not valid.')
            
            print("The requested transmission for source "+str(i)+" is:", S[i])
                               
        if plot:
            cbar = plt.colorbar(ax[1].imshow(np.real(epsr_opt).T, cmap='RdGy',\
                            vmin = np.min(self.design_region*np.real(epsr_opt)),\
                            vmax = np.max(np.real(epsr_opt))), ax=ax[len(src_names)])
            cbar.ax.set_ylabel('Relative Permittivity', fontsize=font)
            ax[len(src_names)].axes.xaxis.set_visible(False)
            ax[len(src_names)].axes.yaxis.set_visible(False)
            plt.savefig(savepath, dpi=1500)
            plt.show()

        
        return S
        
        
    ###########################################################################
    ## Device Evaluation
    ###########################################################################    
    def Calc_SParams_opt(self, rho, src_names, denoms, s11, s21, savepath,\
                    plasma = False, plot = True, wp_max = 0, gamma = 0,\
                    uniform = True, bounds = []):
        """
        Simulate device given optimal parameters and evaluate the S-parameters
        between two given source slices.
        
        Args:
            rho: optimal parameters
            src_names: names of sources in sources dictionary that will be active
                      in the simulation.
            denoms: [denominator for S-params]
            s11: [[np.array([[s11_loc_begin],[s11_loc_end]]), u_s11]]
                u_s11: unit vector in direction of source propagation
            s21: [[[np.array([[s21_loc_begin],[s21_loc_end]]), u_s21],...]]
            savepath = save path
            bounds: Upper and lower bounds for parameters
            plasma: bool specifying if params map to wp
            plot: bool determining if the plot is shown
        """
        if plot:
            fig, ax = plt.subplots(1,len(src_names)+1,constrained_layout=False,\
                                   figsize=(4.5*(len(src_names)+1),4))
            
        S = []
        for i in range(len(src_names)):
            if i > 0:
                if src_names[i] != src_names[i-1]:
                    self.Clear_fields()
            w_src = self.sources[src_names[i]][1]*self.a/2/np.pi/c
            pol = self.sources[src_names[i]][2]
            w = self.sources[src_names[i]][1]
            src = self.sources[src_names[i]][0]

            if plasma:
                epsr_opt = self.Rho_Parameterization_wp(rho, w_src, wp_max, gamma,\
                                                        uniform)
            else:
                epsr_opt = self.Rho_Parameterization(rho, bounds)

            if pol == 'hz':
                #Calculate fields if needed
                if self.fields == []:
                    simulation = fdfd_hz(w, self.dl, epsr_opt, [self.Npml, self.Npml])
                    Ex, Ey, Hz = simulation.solve(src)
                    self.fields = [Ex, Ey, Hz]
                
                #Plot magnitude of Poynting Vector if plot
                if plot:
                    Sp = Sp_mag(Sp_hz(self.fields[0], self.fields[1], self.fields[2]))
                    cbar = plt.colorbar(ax[i].imshow(Sp.T, cmap='magma'), ax=ax[i])
                    cbar.set_ticks([])
                    cbar.ax.set_ylabel('Poynting Flux', fontsize=font)
                    ax[i].axes.xaxis.set_visible(False)
                    ax[i].axes.yaxis.set_visible(False)
                    
            elif pol == 'ez':
                #Calculate fields if needed
                if self.fields == []:
                    simulation = fdfd_ez(w, self.dl, epsr_opt, [self.Npml, self.Npml])
                    Hx, Hy, Ez = simulation.solve(src)
                    self.fields = [Hx, Hy, Ez]
                    
                #Plot magnitude of Poynting Vector if plot
                if plot:
                    Sp = Sp_mag(Sp_ez(self.fields[0], self.fields[1], self.fields[2]))
                    cbar = plt.colorbar(ax[i].imshow(Sp.T, cmap='magma'), ax=ax[i])
                    cbar.set_ticks([])
                    cbar.ax.set_ylabel('Poynting Flux', fontsize=font)
                    ax[i].axes.xaxis.set_visible(False)
                    ax[i].axes.yaxis.set_visible(False)
            else:
                raise RuntimeError('The polarization associated with this source is\
                               not valid.')
            
            #Calculate Sparams
            S_src = []
            self.Add_Probe(s11[i][0][0,:], s11[i][0][1,:], w_src, 'mask', pol)
            Flux = Calc_Transmission(self.fields[0], self.fields[1],\
                    self.fields[2], pol, 1, self.probes.pop('mask')[3], s11[i][1])
            print('denom:',denoms[i])
            print('S11 Flux:',Flux)
            S_src.append(10*np.log10(np.abs(denoms[i] - Flux)/denoms[i]))
            for s21_prb in (s21[i]):
                print('probe:',s21_prb)
                self.Add_Probe(s21_prb[0][0,:], s21_prb[0][1,:], w_src, 'mask', pol)
                Flux = Calc_Transmission(self.fields[0], self.fields[1],\
                    self.fields[2], pol, 1, self.probes.pop('mask')[3], s21_prb[1])
                print('flux:',Flux)
                S_src.append(10*np.log10(Flux/denoms[i]))
            S.append(S_src)
            print("The requested S-params for source "+str(i)+" are:", S[i])
                               
        if plot:
            cbar = plt.colorbar(ax[len(src_names)].imshow(np.real(epsr_opt).T, cmap='RdGy',\
                            vmin = np.min(self.design_region*np.real(epsr_opt)),\
                            vmax = np.max(np.real(epsr_opt))), ax=ax[len(src_names)])
            cbar.ax.set_ylabel('Relative Permittivity', fontsize=font)
            ax[len(src_names)].axes.xaxis.set_visible(False)
            ax[len(src_names)].axes.yaxis.set_visible(False)
            plt.savefig(savepath, dpi=1500)
            plt.show()
        
        return S
    
    
    def Get_Total_Flux(self, src_names, prbs, opt = False, rho = None, savepath = 'none.pdf',
                       plasma = False, plot = True, wp_max = 0, gamma = 0,\
                       uniform = True, bounds = []):
        """
        Get total energy flux out of a source.
        
        Args:
            rho: optimal parameters
            src_names: names of sources in sources dictionary that will be active
                      in the simulation.
            denom: denominator for S-params
            prbs: [[[np.array([[prb_loc_begin],[prb_loc_end]]), u_prb]]]
                u_prb: unit vector in direction of source propagation
            savepath = save path
            bounds: Upper and lower bounds for parameters
            plasma: bool specifying if params map to wp
            plot: bool determining if the plot is shown
        """
        if plot:
            fig, ax = plt.subplots(1,len(src_names)+1,constrained_layout=False,\
                                   figsize=(4.5*(len(src_names)+1),4))
            
        flux_out = []
        for i in range(len(src_names)):
            if i > 0:
                if src_names[i] != src_names[i-1]:
                    self.Clear_fields()
            w_src = self.sources[src_names[i]][1]*self.a/2/np.pi/c
            pol = self.sources[src_names[i]][2]
            w = self.sources[src_names[i]][1]
            src = self.sources[src_names[i]][0]
            
            if not opt:
                epsr_opt = self.epsr
            else:
                if plasma:
                    epsr_opt = self.Rho_Parameterization_wp(rho, w_src, wp_max, gamma,\
                                                        uniform)
                else:
                    epsr_opt = self.Rho_Parameterization(rho, bounds)
                
            if pol == 'hz':
                #Calculate fields if needed
                if self.fields == []:
                    simulation = fdfd_hz(w, self.dl, epsr_opt, [self.Npml, self.Npml])
                    Ex, Ey, Hz = simulation.solve(src)
                    self.fields = [Ex, Ey, Hz]
                
                #Plot magnitude of Poynting Vector if plot
                if plot:
                    Sp = Sp_mag(Sp_hz(self.fields[0], self.fields[1], self.fields[2]))
                    
            elif pol == 'ez':
                #Calculate fields if needed
                if self.fields == []:
                    simulation = fdfd_ez(w, self.dl, epsr_opt, [self.Npml, self.Npml])
                    Hx, Hy, Ez = simulation.solve(src)
                    self.fields = [Hx, Hy, Ez]
                    
                    #Plot magnitude of Poynting Vector if plot
                if plot:
                    Sp = Sp_mag(Sp_ez(self.fields[0], self.fields[1], self.fields[2]))
                    
            else:
                raise RuntimeError('The polarization associated with this source is\
                               not valid.')
                    
            if plot:
                cbar = plt.colorbar(ax[i].imshow(Sp.T, cmap='magma'), ax=ax[i])
                cbar.set_ticks([])
                cbar.ax.set_ylabel('Poynting Flux', fontsize=font)
                ax[i].axes.xaxis.set_visible(False)
                ax[i].axes.yaxis.set_visible(False)
            
            flux_tot = 0
            for prb_ in prbs[i]:
                self.Add_Probe(prb_[0][0,:], prb_[0][1,:], w_src, 'mask', pol)
                flux_side = Calc_Transmission(self.fields[0], self.fields[1],\
                    self.fields[2], pol, 1, self.probes.pop('mask')[3], prb_[1])
                print('probe:', prb_)
                print('flux:', flux_side)
                flux_tot += flux_side
            flux_out.append(flux_tot)
                
        if plot:
            cbar = plt.colorbar(ax[len(src_names)].imshow(np.real(epsr_opt).T, cmap='RdGy',\
                            vmin = np.min(self.design_region*np.real(epsr_opt)),\
                            vmax = np.max(np.real(epsr_opt))), ax=ax[len(src_names)])
            cbar.ax.set_ylabel('Relative Permittivity', fontsize=font)
            ax[len(src_names)].axes.xaxis.set_visible(False)
            ax[len(src_names)].axes.yaxis.set_visible(False)
            plt.savefig(savepath, dpi=1500)
            plt.show()
        
        return flux_out
    
    
    def Get_Trans_Denom(self, src_names, prb, opt = False, rho = None, savepath = 'none.pdf',\
                    plasma = False, plot = True, wp_max = 0, gamma = 0,\
                    uniform = True, bounds = []):
        """
        Simulate device given optimal parameters and evaluate the S-parameters
        between two given source slices.
        
        Args:
            rho: optimal parameters
            src_names: names of sources in sources dictionary that will be active
                      in the simulation.
            denom: denominator for S-params
            prb: [[np.array([[prb_loc_begin],[prb_loc_end]]), u_prb]]
                u_prb: unit vector in direction of source propagation
            savepath = save path
            bounds: Upper and lower bounds for parameters
            plasma: bool specifying if params map to wp
            plot: bool determining if the plot is shown
        """
        if plot:
            fig, ax = plt.subplots(1,len(src_names)+1,constrained_layout=False,\
                                   figsize=(4.5*(len(src_names)+1),4))
            
        denoms = []
        for i in range(len(src_names)):
            if i > 0:
                if src_names[i] != src_names[i-1]:
                    self.Clear_fields()
            w_src = self.sources[src_names[i]][1]*self.a/2/np.pi/c
            pol = self.sources[src_names[i]][2]
            w = self.sources[src_names[i]][1]
            src = self.sources[src_names[i]][0]
            
            if not opt:
                epsr_opt = self.epsr
            else:
                if plasma:
                    epsr_opt = self.Rho_Parameterization_wp(rho, w_src, wp_max, gamma,\
                                                        uniform)
                else:
                    epsr_opt = self.Rho_Parameterization(rho, bounds)

            if pol == 'hz':
                #Calculate fields if needed
                if self.fields == []:
                    simulation = fdfd_hz(w, self.dl, epsr_opt, [self.Npml, self.Npml])
                    Ex, Ey, Hz = simulation.solve(src)
                    self.fields = [Ex, Ey, Hz]
                
                #Plot magnitude of Poynting Vector if plot
                if plot:
                    Sp = Sp_mag(Sp_hz(self.fields[0], self.fields[1], self.fields[2]))
                    cbar = plt.colorbar(ax[i].imshow(Sp.T, cmap='magma'), ax=ax[i])
                    cbar.set_ticks([])
                    cbar.ax.set_ylabel('Poynting Flux', fontsize=font)
                    ax[i].axes.xaxis.set_visible(False)
                    ax[i].axes.yaxis.set_visible(False)
                    
            elif pol == 'ez':
                #Calculate fields if needed
                if self.fields == []:
                    simulation = fdfd_ez(w, self.dl, epsr_opt, [self.Npml, self.Npml])
                    Hx, Hy, Ez = simulation.solve(src)
                    self.fields = [Hx, Hy, Ez]
                    
                    #Plot magnitude of Poynting Vector if plot
                if plot:
                    Sp = Sp_mag(Sp_ez(self.fields[0], self.fields[1], self.fields[2]))
                    cbar = plt.colorbar(ax[i].imshow(Sp.T, cmap='magma'), ax=ax[i])
                    cbar.set_ticks([])
                    cbar.ax.set_ylabel('Poynting Flux', fontsize=font)
                    ax[i].axes.xaxis.set_visible(False)
                    ax[i].axes.yaxis.set_visible(False)
            else:
                raise RuntimeError('The polarization associated with this source is\
                               not valid.')
                    
            self.Add_Probe(prb[i][0][0,:], prb[i][0][1,:], w_src, 'mask', pol)
            denoms.append(Calc_Transmission(self.fields[0], self.fields[1],\
                    self.fields[2], pol, 1, self.probes.pop('mask')[3], prb[i][1]))
                
        if plot:
            cbar = plt.colorbar(ax[len(src_names)].imshow(np.real(epsr_opt).T, cmap='RdGy',\
                            vmin = np.min(self.design_region*np.real(epsr_opt)),\
                            vmax = np.max(np.real(epsr_opt))), ax=ax[len(src_names)])
            cbar.ax.set_ylabel('Relative Permittivity', fontsize=font)
            ax[len(src_names)].axes.xaxis.set_visible(False)
            ax[len(src_names)].axes.yaxis.set_visible(False)
            plt.savefig(savepath, dpi=1500)
            plt.show()
        
        return denoms
    

    ###########################################################################
    ## Parameterization Functions
    ###########################################################################
    def Mask_Combine_Rho(self, train_epsr, elem_locs, bounds, eps_bg_des, Complex = False):
        """
        Utility function for combining the design region with its trainable 
        elements and the static region

        Args:
            train_epsr: An np array that is the size of the domain and contains
            zeros everywhere except the location of the trainable elements where
            their scaled epsilon values are present.
            elem_locs: similar to train_epsr but just an element selector
            bounds: Array containing the upper and lower bounds of the perm of
            the trainable elements
            eps_bg_des: Float, permittivity of the background in the design region
        """
        if Complex:
            train = (bounds[1] - bounds[0])*train_epsr*(elem_locs!=0).astype(np.complex128)\
                    + bounds[0]*(elem_locs!=0).astype(np.complex128)
            design = eps_bg_des*self.design_region*(elem_locs==0).astype(np.complex128)
            bckgd = self.static_elems*(self.design_region==0).astype(np.complex128)
        else:
            train = (bounds[1] - bounds[0])*train_epsr*(elem_locs!=0).astype(np.float)\
                    + bounds[0]*(elem_locs!=0).astype(np.float)
            design = eps_bg_des*self.design_region*(elem_locs==0).astype(np.float)
            bckgd = self.static_elems*(self.design_region==0).astype(np.float)

        return train + design + bckgd


    def Mask_Combine_Rho_wp(self, train_epsr, elem_locs, eps_bg_des, Complex = False):
        """
        Utility function for combining the design region with its trainable 
        elements and the static region when mapping directly from wp values

        Args:
            train_epsr: An np array that is the size of the domain and contains
            zeros everywhere except the location of the trainable elements where
            their scaled epsilon values are present.
            elem_locs: similar to train_epsr but just an element selector
            w_src: Source frequency
            eps_bg_des: Float, permittivity of the background in the design region
        """
        if Complex:
            train = (train_epsr)*(elem_locs!=0).astype(np.complex128)
            design = eps_bg_des*self.design_region*(elem_locs==0).astype(np.complex128)
            bckgd = self.static_elems*(self.design_region==0).astype(np.complex128)
        else:
            train = (train_epsr)*(elem_locs!=0).astype(np.float)
            design = eps_bg_des*self.design_region*(elem_locs==0).astype(np.float)
            bckgd = self.static_elems*(self.design_region==0).astype(np.float)

        return train + design + bckgd


    def Scale_Rho(self, rho):
        """
        Scales parameters to the space (0,1). NOTE: Runs into vanishing gradient
        issues near the edges of the allowed values for eps

        Args:
            rho: Parameters being optimized
        """
        rho = rho.flatten()
        rho = npa.arctan(rho) / np.pi + 0.5
        train_elem = self.train_elems[0].toarray()
        train_epsr = np.zeros(train_elem.shape)
        elem_locations = np.zeros(train_elem.shape)
        for i in range(len(rho)):
            train_elem = self.train_elems[i].toarray()
            train_epsr += rho[i]*train_elem
            elem_locations += train_elem
        
        return train_epsr, elem_locations


    def Scale_Rho_wp(self, rho, w_src, wp_max = 0, gamma = 0, pmat = np.empty(0)):
        """
        Uses the Drude dispersion along with an arctan barrier to map the
        parameters to relative permittivity

        Args:
            rho: Parameters being optimized
            w_src: Non-dimensionalized operating frequency
            wp_max: Approximate maximum non-dimensionalized plasma frequency
            gamma: Non-dimensionalized collision frequency
            pmat: perturbation matrix
        """
        rho = rho.flatten()
        pmat = pmat.flatten()
        denom = w_src**2 - 1j*gamma*w_src
        if wp_max > 0:
            rho = (wp_max/1.5)*npa.arctan(rho/(wp_max/7.5))
        if pmat.shape == rho.shape:
            wp2 = npa.power(npa.abs(rho), 2)
            wp2p = npa.abs(wp2 + npa.multiply(wp2, pmat))
            rho = npa.subtract(1, npa.divide(wp2p, denom))
        else:
            rho = npa.subtract(1, npa.divide(npa.power(npa.abs(rho), 2), denom))
        train_elem = self.train_elems[0].toarray()
        train_epsr = np.zeros(train_elem.shape)
        elem_locations = np.zeros(train_elem.shape)
        for i in range(len(rho)):
            train_elem = self.train_elems[i].toarray()
            train_epsr = train_epsr + rho[i]*train_elem
            elem_locations += train_elem
        
        return train_epsr, elem_locations


    def Scale_Rho_wp_polynomial(self, rho, w_src, wp_max = 0, gamma = 0,\
                                pmat = np.empty(0)):
        """
        Uses the Drude dispersion along with an arctan barrier to map the
        parameters to relative permittivity according to a 6th order polynomial
        density profile

        Args:
            rho: Parameters being optimized. In this case, these parameters are
                 the non-dimensionalizxed average plasma frequencies of the
                 columns.
            w_src: Non-dimensionalized operating frequency
            wp_max: Approximate maximum non-dimensionalized plasma frequency
            gamma: Non-dimensionalized collision frequency
            pmat: perturbation matrix
        """
        rho = rho.flatten()
        pmat = pmat.flatten()
        denom = w_src**2 - 1j*gamma*w_src
        train_elem = self.train_elems[0].toarray()
        train_epsr = np.zeros(train_elem.shape)
        elem_locations = np.zeros(train_elem.shape)
        if wp_max > 0:
            rho = (wp_max/1.5)*npa.arctan(rho/(wp_max/7.5))
        if pmat.shape == rho.shape:
            wp2 = npa.power(npa.abs(rho), 2)
            wp2p = npa.abs(wp2 + npa.multiply(wp2, pmat))
            for r in range(self.rod_shells):
                rho_shell = npa.subtract(1, npa.divide(npa.multiply(wp2p,\
                            4*((4.6/6.5)**2)*(1-r**6/(self.rod_shells-1)**6)/3), denom))
                for i in range(len(rho_shell)):
                    train_elem = self.train_elems[i*self.rod_shells + r].toarray()
                    train_epsr = train_epsr + rho_shell[i]*train_elem
                    elem_locations += train_elem
        else:
            for r in range(self.rod_shells):
                rho_shell = npa.subtract(1, npa.divide(npa.multiply(npa.power(npa.abs(rho), 2),\
                            4*((4.6/6.5)**2)*(1-r**6/(self.rod_shells-1)**6)/3), denom))
                for i in range(len(rho_shell)):
                    train_elem = self.train_elems[i*self.rod_shells + r].toarray()
                    train_epsr = train_epsr + rho_shell[i]*train_elem
                    elem_locations += train_elem
        
        return train_epsr, elem_locations


    def Eps_to_Rho(self, epsr, bounds=[], plasma = False, w_src = 1, wp_max = 0):
        """
        Returns parameters associated with array of values of epsr

        Args:
            epsr: array of relative permittivity values
            bounds: Max and min perm values if directly training eps
            plasma: boolean determining whether or not you're directly
                    optimizing wp.
            w_src: source frequency
        """
        if plasma:
            return self.Eps_to_Rho_wp(epsr, w_src, wp_max)
        else:
            return npa.tan(((epsr-bounds[0])/(bounds[1]-bounds[0])-0.5)*np.pi)

    
    def Rho_to_Eps(self, rho, bounds=[], plasma = False, w_src = 1, wp_max = 0, gamma = 0):
        """
        Returns permittivity values associated with a parameter matrix

        Args:
            rho: array of optimization parameters
            bounds: Min and max eps values for training if directly optimizing
                    eps
            plasma: bool specifying if params map to wp
            w_src: source frequency
        """
        if plasma:
            return self.Rho_to_Eps_wp(rho, w_src, wp_max)
        else:
            return (bounds[1]-bounds[0])*(npa.arctan(rho)/np.pi+0.5)+bounds[0]


    def Eps_to_Rho_wp(self, epsr, w_src, wp_max = 0):
        """
        Returns parameters associated with array of *real* values of epsr

        Args:
            epsr: array of relative permittivity values
            w_src: source frequency
        """
        if np.max(epsr) >= 1:
            raise RuntimeError("One or more of the permittivity values is invalid.\
                    Choose relative permittivity values less than 1.")  
        if wp_max > 0:
            return npa.tan((w_src**2*(1-epsr))**0.5*(1.5/wp_max))*(wp_max/7.5)
        else:
            return (w_src**2*(1-epsr))**0.5


    
    def Rho_to_Eps_wp(self, rho, w_src, wp_max = 0, gamma = 0):
        """
        Returns permittivity values associated with a parameter matrix

        Args:
            rho: array of optimization parameters
            w_src: source frequency
        """
        if wp_max > 0:
            if gamma > 0:
                denom = w_src**2 + 1j*gamma*w_src
                return 1-((wp_max/1.5)*npa.arctan(rho/(wp_max/7.5)))**2/denom
            else:
                return 1-((wp_max/1.5)*npa.arctan(rho/(wp_max/7.5)))**2/w_src**2
        else:
            return 1-(rho)**2/w_src**2
  

    def Rho_Parameterization(self, rho, bounds, eps_bg_des = 1):
        """
        Apply activation/parameterization and create a permittivity matrix

        Args:
            rho: parameters to be optimized
            bounds: upper and lower limits for the elements of rho
            eps_bg_des: background epsilon for the design/optimization region
        """
        train_epsr, elem_locs = self.Scale_Rho(rho)

        return self.Mask_Combine_Rho(train_epsr, elem_locs, bounds, eps_bg_des)


    def Rho_Parameterization_wp(self, rho, w_src, wp_max = 0, gamma = 0,\
                                uniform = True, eps_bg_des = 1, pmat = np.empty(0)):
        """
        Apply scaling/parameterization and create a permittivity matrix when
        mapping plasma frequency to permittivity

        Args:
            rho: parameters to be optimized
            w_src: Source frequency, non-dimensionalized
            eps_bg_des: background epsilon for the design/optimization region
            pmat: perturbation matrix
        """
        Complex = False
        if gamma > 0:
            Complex = True
        if uniform:
            train_epsr, elem_locs = self.Scale_Rho_wp(rho, w_src, wp_max, gamma,\
                                                      pmat)
        else:
            train_epsr, elem_locs = self.Scale_Rho_wp_polynomial(rho, w_src, wp_max,\
                                                                gamma, pmat)

        return self.Mask_Combine_Rho_wp(train_epsr, elem_locs, eps_bg_des, Complex = Complex)


    def Pmat(self, rho, perturb):
        """
        Create perturbation matrix which is composed of mean-0 gaussian RV
        with std. dev = perturb

        Args:
            rho: array that is same size as the array that determines element
                 permittivities.
            perturb: sigma value for gaussian perturbation of rods. 0.05 would
                     result in random perturbations of each element of ~5%.
        """
        return np.random.normal(loc = 0, scale = perturb*np.ones_like(rho))


    def gamma(self, gamma_Hz):
        """
        Convert dimensionalized collision frequency to non-dim freq using a
        """
        return gamma_Hz/(c/self.a)
    
    
    def f_Hz(self, F_a):
        """
        Convert non-dimensionalized frequency to dim freq using a
        """
        return F_a*(c/self.a)


    def Get_MP_Norms(self, Rho, src_1, src_2, prb_1, prb_2, bounds = [],\
                     plasma = False, wp_max = 0, gamma = 0, uniform = True):
        """
        Get norms for a given set of starting parameters; useful for cases
        where runs are cut off prematurely

        Args:
            Rho: Initial parameters
            src_1: Key for source 1 in the sources dict.
            src_2: Key for source 2 in the sources dict.
            prb_1: Key for probe 1 in the probes dict.
            prb_2: Key for probe 2 in the probes dict.
            bounds: Lower and upper limits to permittivity values (e.g. [-6,1])
            plasma: bool specifying if params map to wp
            wp_max: Max plasma frequency in a units
            gamma: Damping frequency in a units
            uniform: bool, specifies whether plasma density profile is uniform
        """
        if plasma:
            epsr_init1 = self.Rho_Parameterization_wp(Rho,\
                    self.sources[src_1][1]*self.a/2/np.pi/c, wp_max, gamma,\
                    uniform)
            epsr_init2 = self.Rho_Parameterization_wp(Rho,\
                    self.sources[src_2][1]*self.a/2/np.pi/c, wp_max, gamma,\
                    uniform)
        else:
            epsr_init1 = self.Rho_Parameterization(Rho, bounds)
            epsr_init2 = self.Rho_Parameterization(Rho, bounds)

        if self.sources[src_1][2] == 'hz' and self.sources[src_2][2] == 'hz':
            sim1 = fdfd_hz(self.sources[src_1][1], self.dl, epsr_init1,\
                           [self.Npml, self.Npml])
            sim2 = fdfd_hz(self.sources[src_2][1], self.dl, epsr_init2,\
                           [self.Npml, self.Npml])
            E1, _, _ = sim1.solve(self.sources[src_1][0])
            E2, _, _ = sim2.solve(self.sources[src_2][0])
        elif self.sources[src_1][2] == 'ez' and self.sources[src_2][2] == 'ez':
            sim1 = fdfd_ez(self.sources[src_1][1], self.dl, epsr_init1,\
                           [self.Npml, self.Npml])
            sim2 = fdfd_ez(self.sources[src_2][1], self.dl, epsr_init2,\
                           [self.Npml, self.Npml])
            _, _, E1 = sim1.solve(self.sources[src_1][0])
            _, _, E2 = sim2.solve(self.sources[src_2][0])
        else:
            raise RuntimeError("The two sources must have the same polarization.")

        E01 = mode_overlap(E1, self.probes[prb_1][0])
        E02 = mode_overlap(E2, self.probes[prb_2][0])
        E01l = field_mag_int(E1, self.probes[prb_2][3])
        E02l = field_mag_int(E2, self.probes[prb_1][3])

        return np.array([E01, E02, E01l, E02l])


    def Get_WG_Norms(self, Rho, src, prb, prbl, bounds = [], plasma = False,\
                     wp_max = 0, gamma = 0, uniform = True, pen = True):
        """
        Get norms for a given set of starting parameters; useful for cases
        where runs are cut off prematurely

        Args:
            Rho: Initial parameters
            src_1: Key for source 1 in the sources dict.
            src_2: Key for source 2 in the sources dict.
            prb_1: Key for probe 1 in the probes dict.
            prb_2: Key for probe 2 in the probes dict.
            bounds: Lower and upper limits to permittivity values (e.g. [-6,1])
            plasma: bool specifying if params map to wp
            wp_max: Max plasma frequency in a units
            gamma: Damping frequency in a units
            uniform: bool, specifies whether plasma density profile is uniform
            pen: bool, is there a penalty?
        """
        if plasma:
            epsr_init = self.Rho_Parameterization_wp(Rho,\
                    self.sources[src][1]*self.a/2/np.pi/c, wp_max, gamma,\
                    uniform)
        else:
            epsr_init = self.Rho_Parameterization(Rho, bounds)

        if self.sources[src][2] == 'hz':
            sim = fdfd_hz(self.sources[src][1], self.dl, epsr_init,\
                           [self.Npml, self.Npml])
            E, _, _ = sim.solve(self.sources[src][0])
        elif self.sources[src][2] == 'ez':
            sim = fdfd_ez(self.sources[src][1], self.dl, epsr_init,\
                           [self.Npml, self.Npml])
            _, _, E = sim.solve(self.sources[src][0])
        else:
            raise RuntimeError("The source polarization is not valid.")

        E0 = mode_overlap(E, self.probes[prb][0])
        E0l = field_mag_int(E, self.probes[prbl][3])

        if pen:
            return np.array([E0, E0l])
        else:
            return E0


    ###########################################################################
    ## Optimizers
    ###########################################################################
    def Optimize_Waveguide(self, Rho, src, prb, alpha, nepochs, bounds = [],\
            plasma = False, wp_max = 0, gamma = 0, uniform = True,\
            param_evolution = False, param_out = None, E0 = None):
        """
        Optimize a waveguide PMM

        Args:
            Rho: Initial parameters
            src: Key for the source in the sources dict.
            prb: Key for probe (slice in desired output waveguide) in the probes 
                 dict.
            alpha: Adam learning rate.
            nepochs: Number of training epochs.
            bounds: Lower and upper limits to permittivity values (e.g. [-6,1])
            plasma: bool specifying if params map to wp
            wp_max: Max plasma frequency in a units
            gamma: Damping frequency in a units
            uniform: bool, specifies whether plasma density profile is uniform
            param_evolution: bool, specifies whether or not to output params at
                             each iteration
            param_out: str, param output directory
            E0: Objective normalization constant, allows objective to continue
                to be tracked over several runs
        """
        #Begin by running sim with initial params to get normalization consts
        if plasma:
            epsr_init = self.Rho_Parameterization_wp(Rho,\
                    self.sources[src][1]*self.a/2/np.pi/c, wp_max, gamma,\
                    uniform)
        else:
            epsr_init = self.Rho_Parameterization(Rho, bounds)

        if self.sources[src][2] == 'hz':
            sim = fdfd_hz(self.sources[src][1], self.dl, epsr_init,\
                           [self.Npml, self.Npml])
            E, _, _ = sim.solve(self.sources[src][0])
        elif self.sources[src][2] == 'ez':
            sim = fdfd_ez(self.sources[src][1], self.dl, epsr_init,\
                           [self.Npml, self.Npml])
            _, _, E = sim.solve(self.sources[src][0])
        else:
            raise RuntimeError("The source polarization is not valid.")

        if E0 == None:
            E0 = mode_overlap(E, self.probes[prb][0])
        
        #Define objective
        def objective(rho):
            """
            Objective function called by optimizer

            1) Takes the density distribution as input
            2) Constructs epsr
            3) Runs the simulation
            4) Returns the overlap integral between the output wg field
            and the desired mode field
            """
            rho = rho.reshape(Rho.shape)
            if plasma:
                epsr = self.Rho_Parameterization_wp(rho,\
                        self.sources[src][1]*self.a/2/np.pi/c, wp_max,\
                        gamma, uniform)
            else:
                epsr = self.Rho_Parameterization(rho, bounds)
            sim.eps_r = epsr

            if self.sources[src][2] == 'hz':
                E, _, _ = sim.solve(self.sources[src][0])
            elif self.sources[src][2] == 'ez':
                _, _, E = sim.solve(self.sources[src][0])
            else:
                raise RuntimeError("The source polarization is not valid.")

            return mode_overlap(E, self.probes[prb][0])/E0

        # Compute the gradient of the objective function
        objective_jac = jacobian(objective, mode='reverse')

        # Maximize the objective function using an ADAM optimizer
        if param_evolution:
            rho_optimum, obj = adam_optimize(objective, Rho.flatten(),\
                                    objective_jac, Nsteps = nepochs,\
                                    direction = 'max', step_size = alpha,\
                                    callback = partial(callback_params,\
                                    dir = param_out))
        else:
            rho_optimum, obj = adam_optimize(objective, Rho.flatten(),\
                                    objective_jac, Nsteps = nepochs,\
                                    direction = 'max', step_size = alpha)

        return rho_optimum.reshape(Rho.shape), obj, E0


    def Optimize_Waveguide_Penalize(self, Rho, src, prb, prbl, alpha, nepochs,\
            bounds = [], plasma = False, wp_max = 0, gamma = 0, uniform = True,\
            param_evolution = False, param_out = None, E0 = None, E0l = None):
        """
        Optimize a waveguide PMM

        Args:
            Rho: Initial parameters
            src: Key for the source in the sources dict.
            prb: Key for probe (slice in desired output waveguide) in the probes dict.
            prbl: Key for probe at slice that you want to penalize.
            alpha: Adam learning rate.
            nepochs: Number of training epochs.
            bounds: Lower and upper limits to permittivity values (e.g. [-6,1])
            plasma: bool specifying if params map to wp
            wp_max: Max plasma frequency in a units
            gamma: Damping frequency in a units
            uniform: bool, specifies whether plasma density profile is uniform
            param_evolution: bool, specifies whether or not to output params at
                             each iteration
            param_out: str, param output directory
            E0,E0l: Field normalization values
        """
        #Begin by running sim with initial params to get normalization consts
        if plasma:
            epsr_init = self.Rho_Parameterization_wp(Rho,\
                    self.sources[src][1]*self.a/2/np.pi/c, wp_max, gamma,\
                    uniform)
        else:
            epsr_init = self.Rho_Parameterization(Rho, bounds)

        if self.sources[src][2] == 'hz':
            sim = fdfd_hz(self.sources[src][1], self.dl, epsr_init,\
                           [self.Npml, self.Npml])
            E, _, _ = sim.solve(self.sources[src][0])
        elif self.sources[src][2] == 'ez':
            sim = fdfd_ez(self.sources[src][1], self.dl, epsr_init,\
                           [self.Npml, self.Npml])
            _, _, E = sim.solve(self.sources[src][0])
        else:
            raise RuntimeError("The source polarization is not valid.")

        if E0 == None:
            E0 = mode_overlap(E, self.probes[prb][0])
        if E0l == None:
            E0l = field_mag_int(E, self.probes[prbl][3])
            
        #Define objective
        def objective(rho):
            """
            Objective function called by optimizer

            1) Takes the density distribution as input
            2) Constructs epsr
            3) Runs the simulation
            4) Returns the overlap integral between the output wg field
            and the desired mode field
            """
            rho = rho.reshape(Rho.shape)
            if plasma:
                epsr = self.Rho_Parameterization_wp(rho,\
                        self.sources[src][1]*self.a/2/np.pi/c, wp_max, gamma,\
                        uniform)
            else:
                epsr = self.Rho_Parameterization(rho, bounds)
            sim.eps_r = epsr

            if self.sources[src][2] == 'hz':
                E, _, _ = sim.solve(self.sources[src][0])
            elif self.sources[src][2] == 'ez':
                _, _, E = sim.solve(self.sources[src][0])
            else:
                raise RuntimeError("The source polarization is not valid.")

            return mode_overlap(E, self.probes[prb][0])/E0-\
                    field_mag_int(E, self.probes[prbl][3])/E0l

        # Compute the gradient of the objective function
        objective_jac = jacobian(objective, mode='reverse')

        # Maximize the objective function using an ADAM optimizer
        if param_evolution:
            rho_optimum, obj = adam_optimize(objective, Rho.flatten(),\
                                objective_jac, Nsteps = nepochs,\
                                direction = 'max', step_size = alpha,\
                                callback = partial(callback_params,\
                                dir = param_out))
        else:
            rho_optimum, obj = adam_optimize(objective, Rho.flatten(),\
                                objective_jac, Nsteps = nepochs,\
                                direction = 'max', step_size = alpha)

        return rho_optimum.reshape(Rho.shape), obj, E0, E0l


    def Optimize_Multiplexer(self, Rho, src_1, src_2, prb_1, prb_2,\
                             alpha, nepochs, bounds = [], plasma = False,\
                             wp_max = 0, gamma = 0, uniform = True,\
                             param_evolution = False, param_out = None,\
                             E01 = None, E02 = None):
        """
        Optimize a multiplexer PMM

        Args:
            Rho: Initial parameters
            src_1: Key for source 1 in the sources dict.
            src_2: Key for source 1 in the sources dict.
            prb_1: Key for probe 1 in the probes dict.
            prb_2: Key for probe 2 in the probes dict.
            alpha: Adam learning rate.
            nepochs: Number of training epochs.
            bounds: Lower and upper limits to permittivity values (e.g. [-6,1])
            plasma: bool specifying if params map to wp
            wp_max: Max plasma frequency in a units
            gamma: Damping frequency in a units
            uniform: bool, specifies whether plasma density profile is uniform
            param_evolution: bool, specifies whether or not to output params at
                             each iteration
            param_out: str, param output directory
            E01,E02: Field normalization values
        """
        #Begin by running sim with initial params to get normalization consts
        if plasma:
            epsr_init1 = self.Rho_Parameterization_wp(Rho,\
                        self.sources[src_1][1]*self.a/2/np.pi/c, wp_max, gamma,\
                        uniform)
            epsr_init2 = self.Rho_Parameterization_wp(Rho,\
                        self.sources[src_2][1]*self.a/2/np.pi/c, wp_max, gamma,\
                        uniform)
        else:
            epsr_init1 = self.Rho_Parameterization(Rho, bounds)
            epsr_init2 = self.Rho_Parameterization(Rho, bounds)

        if self.sources[src_1][2] == 'hz' and self.sources[src_2][2] == 'hz':
            sim1 = fdfd_hz(self.sources[src_1][1], self.dl, epsr_init1,\
                           [self.Npml, self.Npml])
            sim2 = fdfd_hz(self.sources[src_2][1], self.dl, epsr_init2,\
                           [self.Npml, self.Npml])
            E1, _, _ = sim1.solve(self.sources[src_1][0])
            E2, _, _ = sim2.solve(self.sources[src_2][0])
        elif self.sources[src_1][2] == 'ez' and self.sources[src_2][2] == 'ez':
            sim1 = fdfd_ez(self.sources[src_1][1], self.dl, epsr_init1,\
                           [self.Npml, self.Npml])
            sim2 = fdfd_ez(self.sources[src_2][1], self.dl, epsr_init2,\
                           [self.Npml, self.Npml])
            _, _, E1 = sim1.solve(self.sources[src_1][0])
            _, _, E2 = sim2.solve(self.sources[src_2][0])
        else:
            raise RuntimeError("The two sources must have the same polarization.")

        if E01 == None:
            E01 = mode_overlap(E1, self.probes[prb_1][0])
        if E02 == None:
            E02 = mode_overlap(E2, self.probes[prb_2][0])
            
        #Define objective
        def objective(rho):
            """
            Objective function called by optimizer

            1) Takes the density distribution as input
            2) Constructs epsr
            3) Runs the simulation
            4) Returns the overlap integral between the output wg field
            and the desired mode field
            """
            rho = rho.reshape(Rho.shape)
            if plasma:
                epsr1 = self.Rho_Parameterization_wp(rho,\
                        self.sources[src_1][1]*self.a/2/np.pi/c, wp_max,\
                        gamma, uniform)
                epsr2 = self.Rho_Parameterization_wp(rho,\
                        self.sources[src_2][1]*self.a/2/np.pi/c, wp_max,\
                        gamma, uniform)
            else:
                epsr1 = self.Rho_Parameterization(rho, bounds)
                epsr2 = self.Rho_Parameterization(rho, bounds)
            sim1.eps_r = epsr1
            sim2.eps_r = epsr2

            if self.sources[src_1][2] == 'hz' and self.sources[src_2][2] == 'hz':
                E1, _, _ = sim1.solve(self.sources[src_1][0])
                E2, _, _ = sim2.solve(self.sources[src_2][0])
            elif self.sources[src_1][2] == 'ez' and self.sources[src_2][2] == 'ez':
                _, _, E1 = sim1.solve(self.sources[src_1][0])
                _, _, E2 = sim2.solve(self.sources[src_2][0])
            else:
                raise RuntimeError("The two sources must have the same polarization.")

            return (mode_overlap(E1, self.probes[prb_1][0])/E01)*\
                    (mode_overlap(E2, self.probes[prb_2][0])/E02)

        # Compute the gradient of the objective function
        objective_jac = jacobian(objective, mode='reverse')

        # Maximize the objective function using an ADAM optimizer
        if param_evolution:
            rho_optimum, obj = adam_optimize(objective, Rho.flatten(),\
                                objective_jac, Nsteps = nepochs,\
                                direction = 'max', step_size = alpha,\
                                callback = partial(callback_params,\
                                dir = param_out))
        else:
            rho_optimum, obj = adam_optimize(objective, Rho.flatten(),\
                                objective_jac, Nsteps = nepochs,\
                                direction = 'max', step_size = alpha)

        return rho_optimum.reshape(Rho.shape), obj, E01, E02


    def Optimize_Multiplexer_Penalize(self, Rho, src_1, src_2, prb_1, prb_2,\
                             alpha, nepochs, bounds = [], plasma = False,\
                             wp_max = 0, gamma = 0, uniform = True,\
                             param_evolution = False, param_out = None,\
                             E01 = None, E02 = None, E01l = None, E02l = None):
        """
        Optimize a multiplexer PMM with leak into opposite gate penalized.

        Args:
            Rho: Initial parameters
            src_1: Key for source 1 in the sources dict.
            src_2: Key for source 1 in the sources dict.
            prb_1: Key for probe 1 in the probes dict.
            prb_2: Key for probe 2 in the probes dict.
            alpha: Adam learning rate.
            nepochs: Number of training epochs.
            bounds: Lower and upper limits to permittivity values (e.g. [-6,1])
            plasma: bool specifying if params map to wp
            wp_max: Max plasma frequency in a units
            gamma: Damping frequency in a units
            uniform: bool, specifies whether plasma density profile is uniform
            param_evolution: bool, specifies whether or not to output params at
                             each iteration
            param_out: str, param output directory
            E01,E02,E01l,E02l: Field normalization values
        """
        #Begin by running sim with initial params to get normalization consts
        if plasma:
            epsr_init1 = self.Rho_Parameterization_wp(Rho,\
                    self.sources[src_1][1]*self.a/2/np.pi/c, wp_max, gamma,\
                    uniform)
            epsr_init2 = self.Rho_Parameterization_wp(Rho,\
                    self.sources[src_2][1]*self.a/2/np.pi/c, wp_max, gamma,\
                    uniform)
        else:
            epsr_init1 = self.Rho_Parameterization(Rho, bounds)
            epsr_init2 = self.Rho_Parameterization(Rho, bounds)

        if self.sources[src_1][2] == 'hz' and self.sources[src_2][2] == 'hz':
            sim1 = fdfd_hz(self.sources[src_1][1], self.dl, epsr_init1,\
                           [self.Npml, self.Npml])
            sim2 = fdfd_hz(self.sources[src_2][1], self.dl, epsr_init2,\
                           [self.Npml, self.Npml])
            E1, _, _ = sim1.solve(self.sources[src_1][0])
            E2, _, _ = sim2.solve(self.sources[src_2][0])
        elif self.sources[src_1][2] == 'ez' and self.sources[src_2][2] == 'ez':
            sim1 = fdfd_ez(self.sources[src_1][1], self.dl, epsr_init1,\
                           [self.Npml, self.Npml])
            sim2 = fdfd_ez(self.sources[src_2][1], self.dl, epsr_init2,\
                           [self.Npml, self.Npml])
            _, _, E1 = sim1.solve(self.sources[src_1][0])
            _, _, E2 = sim2.solve(self.sources[src_2][0])
        else:
            raise RuntimeError("The two sources must have the same polarization.")

        if E01 == None:
            E01 = mode_overlap(E1, self.probes[prb_1][0])
        if E02 == None:
            E02 = mode_overlap(E2, self.probes[prb_2][0])
        if E01l == None:
            E01l = field_mag_int(E1, self.probes[prb_2][3])
        if E02l == None:
            E02l = field_mag_int(E2, self.probes[prb_1][3])
            
        #Define objective
        def objective(rho):
            """
            Objective function called by optimizer

            1) Takes the density distribution as input
            2) Constructs epsr
            3) Runs the simulation
            4) Returns the overlap integral between the output wg field
            and the desired mode field
            """
            rho = rho.reshape(Rho.shape)
            if plasma:
                epsr1 = self.Rho_Parameterization_wp(rho,\
                        self.sources[src_1][1]*self.a/2/np.pi/c, wp_max,\
                        gamma, uniform)
                epsr2 = self.Rho_Parameterization_wp(rho,\
                        self.sources[src_2][1]*self.a/2/np.pi/c, wp_max,\
                        gamma, uniform)
            else:
                epsr1 = self.Rho_Parameterization(rho, bounds)
                epsr2 = self.Rho_Parameterization(rho, bounds)
            sim1.eps_r = epsr1
            sim2.eps_r = epsr2

            if self.sources[src_1][2] == 'hz' and self.sources[src_2][2] == 'hz':
                E1, _, _ = sim1.solve(self.sources[src_1][0])
                E2, _, _ = sim2.solve(self.sources[src_2][0])
            elif self.sources[src_1][2] == 'ez' and self.sources[src_2][2] == 'ez':
                _, _, E1 = sim1.solve(self.sources[src_1][0])
                _, _, E2 = sim2.solve(self.sources[src_2][0])
            else:
                raise RuntimeError("The two sources must have the same polarization.")

            return (mode_overlap(E1, self.probes[prb_1][0])/E01)*\
                    (mode_overlap(E2, self.probes[prb_2][0])/E02)-\
                    (field_mag_int(E1, self.probes[prb_2][3])/E01l)-\
                    (field_mag_int(E2, self.probes[prb_1][3])/E02l)

        # Compute the gradient of the objective function
        objective_jac = jacobian(objective, mode='reverse')

        # Maximize the objective function using an ADAM optimizer
        if param_evolution:
            rho_optimum, obj = adam_optimize(objective, Rho.flatten(),\
                                objective_jac, Nsteps = nepochs,\
                                direction = 'max', step_size = alpha,\
                                callback = partial(callback_params,\
                                dir = param_out))
        else:
            rho_optimum, obj = adam_optimize(objective, Rho.flatten(),\
                                objective_jac, Nsteps = nepochs,\
                                direction = 'max', step_size = alpha)

        return rho_optimum.reshape(Rho.shape), obj, E01, E02, E01l, E02l


    def Optimize_Logic_Gate(self, Rho, src_1, src_2, src_c, prb_n, prb_t, alpha,\
            nepochs, logic, bounds = [], plasma = False, wp_max = 0, gamma = 0,\
            uniform = True, param_evolution = False, param_out = None,\
            Ec0n = None, E10n = None, E20n = None, E10t = None, E20t = None,\
            E120t = None, E10ln = None, E20ln = None, E120ln = None,\
            Ec0lt = None, E10lt = None, E20lt = None):
        """
        Optimize a logic gate PMM

        Args:
            Rho: Initial parameters
            src_1: Key for source 1 in the sources dict.
            src_2: Key for source 1 in the sources dict.
            src_c: Key for the constant source in the sources dict.
            prb_n: Key for the not true probe in the probes dict
            prb_t: Key for the true probe in the probes dict.
            alpha: Adam learning rate.
            nepochs: Number of training epochs.
            logic: string specifying what kind of logic you're interested in
            bounds: Lower and upper limits to permittivity values (e.g. [-6,1])
            plasma: bool specifying if params map to wp
            wp_max: Max plasma frequency in a units
            gamma: Damping frequency in a units
            uniform: bool, specifies whether plasma density profile is uniform
            param_evolution: bool, specifies whether or not to output params at
                             each iteration
            param_out: str, param output directory
            Ec0n, E10n, etc.: Field normalization values
        """
        #Begin by running sim with initial parameters to get normalization consts
        if plasma:
            epsr_init = self.Rho_Parameterization_wp(Rho,\
                        self.sources[src_1][1]*self.a/2/np.pi/c, wp_max, gamma,\
                        uniform)
        else:
            epsr_init = self.Rho_Parameterization(Rho, bounds)
            
        if self.sources[src_1][2] == 'hz' and self.sources[src_2][2] == 'hz':
            sim = fdfd_hz(self.sources[src_1][1], self.dl, epsr_init,\
                           [self.Npml, self.Npml])
            Ec, _, _ = sim.solve(self.sources[src_c][0])
            E1, _, _ = sim.solve(self.sources[src_c][0]+self.sources[src_1][0])
            E2, _, _ = sim.solve(self.sources[src_c][0]+self.sources[src_2][0])
            E12, _, _ = sim.solve(self.sources[src_c][0]+self.sources[src_1][0]+\
                                  self.sources[src_2][0])
        elif self.sources[src_1][2] == 'ez' and self.sources[src_2][2] == 'ez':
            sim = fdfd_ez(self.sources[src_1][1], self.dl, epsr_init,\
                           [self.Npml, self.Npml])
            _, _, Ec = sim.solve(self.sources[src_c][0])
            _, _, E1 = sim.solve(self.sources[src_c][0]+self.sources[src_1][0])
            _, _, E2 = sim.solve(self.sources[src_c][0]+self.sources[src_2][0])
            _, _, E12 = sim.solve(self.sources[src_c][0]+self.sources[src_1][0]+\
                                  self.sources[src_2][0])
        else:
            raise RuntimeError("The two sources must have the same polarization.")

        if Ec0n == None:
            Ec0n = mode_overlap(Ec, self.probes[prb_n][0])
        if E10n == None:
            E10n = mode_overlap(E1, self.probes[prb_n][0])
        if E20n == None:
            E20n = mode_overlap(E2, self.probes[prb_n][0])
        if E10t == None:
            E10t = mode_overlap(E1, self.probes[prb_t][0])
        if E20t == None:
            E20t = mode_overlap(E2, self.probes[prb_t][0])
        if E120t == None:
            E120t = mode_overlap(E12, self.probes[prb_t][0])
        if E10ln == None:
            E10ln = field_mag_int(E1, self.probes[prb_n][3])
        if E20ln == None:
            E20ln = field_mag_int(E2, self.probes[prb_n][3])
        if E120ln == None:
            E120ln = field_mag_int(E12, self.probes[prb_n][3])
        if Ec0lt == None:
            Ec0lt = field_mag_int(Ec, self.probes[prb_t][3])
        if E10lt == None:
            E10lt = field_mag_int(E1, self.probes[prb_t][3])
        if E20lt == None:
            E20lt = field_mag_int(E2, self.probes[prb_t][3])
           
        #Define objective
        def objective(rho):
            """
            Objective function called by optimizer

            1) Takes the density distribution as input
            2) Constructs epsr
            3) Runs the simulation
            4) Returns the overlap integral between the output wg field
            and the desired mode field
            """
            rho = rho.reshape(Rho.shape)
            if plasma:
                epsr = self.Rho_Parameterization_wp(rho,\
                        self.sources[src_1][1]*self.a/2/np.pi/c, wp_max,\
                        gamma, uniform)
            else:
                epsr = self.Rho_Parameterization(rho, bounds)
            sim.eps_r = epsr

            if self.sources[src_1][2] == 'hz' and self.sources[src_2][2] == 'hz':
                Exc, _, _ = sim.solve(self.sources[src_c][0])
                Ex1, _, _ = sim.solve(self.sources[src_c][0]+self.sources[src_1][0])
                Ex2, _, _ = sim.solve(self.sources[src_c][0]+self.sources[src_2][0])
                Ex12, _, _ = sim.solve(self.sources[src_c][0]+self.sources[src_1][0]+\
                                       self.sources[src_2][0])
            elif self.sources[src_1][2] == 'ez' and self.sources[src_2][2] == 'ez':
                _, _, Ec = sim.solve(self.sources[src_c][0])
                _, _, E1 = sim.solve(self.sources[src_c][0]+self.sources[src_1][0])
                _, _, E2 = sim.solve(self.sources[src_c][0]+self.sources[src_2][0])
                _, _, E12 = sim.solve(self.sources[src_c][0]+self.sources[src_1][0]+\
                                      self.sources[src_2][0])
            else:
                raise RuntimeError("The two sources must have the same polarization.")

            if logic == 'and':
                off = mode_overlap(Ec, self.probes[prb_n][0])/Ec0n -\
                    field_mag_int(Ec, self.probes[prb_t][3])/Ec0lt
                one = mode_overlap(E1, self.probes[prb_n][0])/E10n -\
                    field_mag_int(E1, self.probes[prb_t][3])/E10lt
                two = mode_overlap(E2, self.probes[prb_n][0])/E20n -\
                    field_mag_int(E2, self.probes[prb_t][3])/E20lt
                both = 3*mode_overlap(E12, self.probes[prb_t][0])/E120t -\
                    3*field_mag_int(E12, self.probes[prb_n][3])/E120ln
                            
            elif logic == 'or':
                off = 3*mode_overlap(Ec, self.probes[prb_n][0])/Ec0n -\
                    3*field_mag_int(Ec, self.probes[prb_t][3])/Ec0lt
                one = mode_overlap(E1, self.probes[prb_t][0])/E10t -\
                    field_mag_int(E1, self.probes[prb_n][3])/E10ln
                two = mode_overlap(E2, self.probes[prb_t][0])/E20t -\
                    field_mag_int(E2, self.probes[prb_n][3])/E20ln
                both = mode_overlap(E12, self.probes[prb_t][0])/E120t -\
                    field_mag_int(E12, self.probes[prb_n][3])/E120ln
            else:
                raise RuntimeError("Logic not implemented yet")

            return off + one + two + both

        # Compute the gradient of the objective function
        objective_jac = jacobian(objective, mode='reverse')

        # Maximize the objective function using an ADAM optimizer
        if param_evolution:
            rho_optimum, obj = adam_optimize(objective, Rho.flatten(),\
                                objective_jac, Nsteps = nepochs,\
                                direction = 'max', step_size = alpha,\
                                callback = partial(callback_params,
                                    dir = param_out))
        else:
            rho_optimum, obj = adam_optimize(objective, Rho.flatten(),\
                                objective_jac, Nsteps = nepochs,\
                                direction = 'max', step_size = alpha)

        return rho_optimum.reshape(Rho.shape), obj


    ###########################################################################
    ## Params i/o
    ###########################################################################
    def Params_to_Exp(self, rho, src, bounds = [], plasma = False, wp_max = 0):
        """
        Output experimental data needed to rebuild a certain design

        Args:
            rho: parameters for trainable element permittivity values
            src: key for active source in sources dict
            bounds: max and min perm values for training
            plasma: bool specifying if params map to wp
            gamma: collision frequency (loss) in c/a units
        """
        if plasma:
            self.Params_to_Exp_wp(rho, src, wp_max)
        else:
            print("The lattice frequency is: ", c/self.a/(10**9)," GHz")
            print("The source frequency is: ", self.sources[src][1]/2/np.pi/(10**9), " GHz")
            print("The plasma frequencies (GHz) necessary to achieve this design are:")
            print(np.sqrt((1-np.real(self.Rho_to_Eps(rho, bounds)))*\
                (self.sources[src][1]**2+(nu_col*2*np.pi*c/self.a)**2))/(10**9))


    def Params_to_Exp_wp(self, rho, src, wp_max = 0):
        """
        Output experimental data needed to rebuild a certain design

        Args:
            rho: parameters for trainable element permittivity values
            bounds: max and min perm values for training
            src: key for active source in sources dict
            nu_col: supposed collision frequency in c/a units
        """
        print("The lattice frequency is: ", c/self.a/(10**9)," GHz")
        print("The source frequency is: ", self.sources[src][1]/2/np.pi/(10**9), " GHz")
        print("The plasma frequencies (GHz, corresponding to average density in discharge) necessary to achieve this design are:")
        if wp_max > 0:
            print((wp_max/1.5)*npa.arctan(npa.abs(rho)/(wp_max/7.5))*c/self.a/(10**9))
        else:
            print(npa.abs(rho)*c/self.a/(10**9))


    def Save_Params(self, rho, savepath):
        """
        Saves optimization parameters. A wrapper for np.savetxt

        Args:
            rho: parameters to be saved
            savepath: save path. Must be csv.
        """
        np.savetxt(savepath, rho, delimiter=",")


    def Read_Params(self, readpath):
        """
        Reads optimization paramters.

        Args:
            readpath: read path. Must be csv.
        """
        return np.loadtxt(readpath, delimiter=",")
