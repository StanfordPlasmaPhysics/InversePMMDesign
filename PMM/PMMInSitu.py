"""
The module composed in this file is meant to facilitate both in-situ inverse
design and experimental control of plasma metamaterials composed of many
elements. The library is constructed for use with Longwei DC power supplies
which are connected on RS485 multidrop networks. When carrying out the
optimization fully in-situ, you must use a Rohde and Schwarz ZNB vector network
analyzer, or else you need to modify the vna commands and use a different VISA
library than RSInstrument. The VISA backend I use on my Macbook Pro is the 
National Instruments VISA. All that to say: this is NOT a general purpose
library and functions with a very specific experimental setup. For more
information, contact Jesse Rodriguez: jrodrig@stanford.edu
05/11/2023
"""

import minimalmodbus
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from RsInstrument import *
import serial
import glob
import sys
import time
import yaml

###############################################################################
## Utility functions and globals
###############################################################################
c = 299792458
e = 1.60217662*10**(-19)
epso = 8.8541878128*10**(-12)
muo = 4*np.pi*10**(-7)
me = 9.1093837015*10**(-31)

def serial_ports():
    """ Lists serial port names

        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of the serial ports available on the system
    """
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result


def Demult_Obj_Comp(freq, S21, S31, f1, f2, df = 0.25, norms = []):
    """
    Compute the demultiplexer objective that is analogous to the computational
    inverse design library. f1 corresponds to port 2, f2 corresponds to port 3.

    Args:
        freq: frequency np.array, in GHz
        S21: S21 np.array, in dB
        S31: S31 np.array, in dB
        f1: first operating frequency in GHz
        f2: second operating frequency in GHz
        df: bandwidth around operating frequencies in GHz
        norms: normalization factors
    """
    i1_l = np.searchsorted(freq, f1-df/2, side='left')
    i1_r = np.searchsorted(freq, f1+df/2, side='right')
    i2_l = np.searchsorted(freq, f2-df/2, side='left')
    i2_r = np.searchsorted(freq, f2+df/2, side='right')
    T21 = np.power(10*np.ones_like(S21), S21/10)
    T31 = np.power(10*np.ones_like(S31), S31/10)

    correct_1 = np.sum(T21[i1_l:i1_r])
    incorrect_1 = np.sum(T31[i1_l:i1_r])
    correct_2 = np.sum(T31[i2_l:i2_r])
    incorrect_2 = np.sum(T21[i2_l:i2_r])

    if len(norms) > 0:
        c_1 = correct_1/norms[0]
        i_1 = incorrect_1/norms[1]
        c_2 = correct_2/norms[2]
        i_2 = incorrect_2/norms[3]
        return c_1*c_2 - i_1 - i_2, norms
    else:
        new_norms = [np.abs(correct_1), np.abs(incorrect_1),\
                     np.abs(correct_2), np.abs(incorrect_2)]
        return -1, new_norms


def Demult_Obj_dB(freq, S21, S31, f1, f2, df = 0.25, norms = []):
    """
    Compute the demultiplexer objective that focuses more instead on dB
    transmission values. f1 corresponds to port 2, f2 corresponds to port 3.

    Args:
        freq: frequency np.array, in GHz
        S21: S21 np.array, in dB
        S31: S31 np.array, in dB
        f1: first operating frequency in GHz
        f2: second operating frequency in GHz
        df: bandwidth around operating frequencies in GHz
        norms: normalization factors
    """
    i1_l = np.searchsorted(freq, f1-df/2, side='left')
    i1_r = np.searchsorted(freq, f1+df/2, side='right')
    i2_l = np.searchsorted(freq, f2-df/2, side='left')
    i2_r = np.searchsorted(freq, f2+df/2, side='right')
    T21 = np.power(10*np.ones_like(S21), S21/10)
    T31 = np.power(10*np.ones_like(S31), S31/10)
    DdB = S21-S31

    correct_1 = np.sum(T21[i1_l:i1_r])
    correct_2 = np.sum(T31[i2_l:i2_r])
    isolation_1 = np.sum(DdB[i1_l:i1_r])
    isolation_2 = np.sum(-DdB[i2_l:i2_r])

    if len(norms) > 0:
        c_1 = correct_1/norms[0]
        c_2 = correct_2/norms[1]
        i_1 = isolation_1/norms[2]
        i_2 = isolation_2/norms[3]
        return c_1*c_2 + i_1 + i_2, norms
    else:
        new_norms = [np.abs(correct_1), np.abs(correct_2),\
                     np.abs(isolation_1), np.abs(isolation_2)]
        c_1 = correct_1/new_norms[0]
        c_2 = correct_2/new_norms[1]
        i_1 = isolation_1/new_norms[2]
        i_2 = isolation_2/new_norms[3]
        return c_1*c_2 + i_1 + i_2, new_norms


def Waveguide_Obj_Comp(freq, S21, S31, f, df = 0.25, norms = []):
    """
    Compute the waveguide objective that is analogous to the computational
    inverse design library. Port 2 has to be correct port, otherwise switch
    S21 and S31.

    Args:
        freq: frequency np.array, in GHz
        S21: S21 np.array, in dB
        S31: S31 np.array, in dB
        f: operating frequency in GHz
        df: bandwidth around operating frequencies in GHz
        norms: normalization factors
    """
    i_l = np.searchsorted(freq, f-df/2, side='left')
    i_r = np.searchsorted(freq, f+df/2, side='right')
    T21 = np.power(10*np.ones_like(S21), S21/10)
    T31 = np.power(10*np.ones_like(S31), S31/10)

    correct = np.sum(T21[i_l:i_r])
    incorrect = np.sum(T31[i_l:i_r])

    if len(norms) > 0:
        c = correct/norms[0]
        i = incorrect/norms[1]
        return c - i, norms
    else:
        new_norms = [np.abs(correct), np.abs(incorrect)]
        return 0, new_norms


def Waveguide_Obj_dB(freq, S21, S31, f, df = 0.25, norms = []):
    """
    Compute the waveguide objective that focuses more instead on dB
    isolation values. Port 2 has to be correct port, otherwise switch
    S21 and S31.
    
    Args:
        freq: frequency np.array, in GHz
        S21: S21 np.array, in dB
        S31: S31 np.array, in dB
        f: operating frequency in GHz
        df: bandwidth around operating frequencies in GHz
        norms: normalization factors
    """
    i_l = np.searchsorted(freq, f-df/2, side='left')
    i_r = np.searchsorted(freq, f+df/2, side='right')
    
    correct = np.sum(S21[i_l:i_r])
    incorrect = np.sum(S31[i_l:i_r])
    
    if len(norms) > 0:
        c = correct/norms[0]
        i = incorrect/norms[1]
        return c - i, norms
    else:
        new_norms = [np.abs(correct), np.abs(incorrect)]
        return 0, new_norms


###############################################################################
## In-situ inverse design class
###############################################################################
class PMMInSitu:
    def __init__(self, conf_file, conf_dir = './../confs/'):
        with open(conf_file, 'r') as conf:
            self.config = yaml.load(conf, Loader=yaml.SafeLoader)

        self.a = self.config['array-a']
        self.mu = self.config['mobility']
        self.L = self.config['bulb-length']
        self.VtoI = np.loadtxt(conf_dir+'VtoI.txt', delimiter = ',')
        self.bulbs = {'all': {}}
        self.VNA = self.config['VNA']
        ports = serial_ports()
        for port in self.config['serial_ports']:
            if port not in ports:
                raise RuntimeError("One or more of the ports in the config file\
                                    is not connected")
                
            print("checking port", port)
            # Create 'all' pathway
            self.bulbs['all'][port] = minimalmodbus.Instrument(port = port,\
                                      slaveaddress = 0,\
                                      mode = minimalmodbus.MODE_RTU)
            self.bulbs['all'][port].serial.baudrate = 9600
            self.bulbs['all'][port].serial.bytesize = 8
            self.bulbs['all'][port].serial.parity = minimalmodbus.serial.PARITY_NONE
            self.bulbs['all'][port].serial.stopbits = 1
            self.bulbs['all'][port].serial.timeout = 1
            self.bulbs['all'][port].serial.close_port_after_each_call = True
            self.bulbs['all'][port].serial.clear_buffers_before_each_transaction = True

            # Create bulb entries in dict
            for bulb_addr in self.config['serial_ports'][port]:
                self.bulbs[bulb_addr] = {'I': 0.0, 'V': 0.0,\
                                    'Inst': minimalmodbus.Instrument(\
                                    port = port, slaveaddress = bulb_addr,\
                                    mode = minimalmodbus.MODE_RTU)}
                self.bulbs[bulb_addr]['Inst'].serial.baudrate = 9600
                self.bulbs[bulb_addr]['Inst'].serial.bytesize = 8
                self.bulbs[bulb_addr]['Inst'].serial.parity = minimalmodbus.serial.PARITY_NONE
                self.bulbs[bulb_addr]['Inst'].serial.stopbits = 1
                self.bulbs[bulb_addr]['Inst'].serial.timeout = 1
                self.bulbs[bulb_addr]['Inst'].serial.close_port_after_each_call = True
                self.bulbs[bulb_addr]['Inst'].serial.clear_buffers_before_each_transaction = True

                # Now ping each of the power supplies and make sure they are
                # connected to the correct RS-485 bus and aren't already running.
                try:
                    on = self.bulbs[bulb_addr]['Inst'].read_register(\
                            registeraddress=0x1004)
                    if on == 1:
                        print("The power supply associated with bulb "\
                              +str(bulb_addr)+" is putting out power, fixing "+\
                              "now. Check power supply.")
                        self.bulbs[bulb_addr]['Inst'].write_register(\
                                registeraddress=0x1006, value = 0, functioncode = 6)
                except:
                    raise RuntimeError("Bulb "+str(bulb_addr)+" is not connected "+\
                                       "to the correct RS-485 bus.")


    def Address(self, coords):
        """
        Takes array coordinates and returns a bulb address

        Args:
            coords: tuple/list; e.g. (i,j)
        """
        return (coords[0]+self.config['array-x']*coords[1])


    def Set_Bulb_VI(self, addr, V, I, verbose = True):
        """
        Set bulb current and voltage

        Args:
            addr: int, bulb address
            V: float, bulb voltage in [0,32] (volts)
            I: float, bulb current in [0,10] (amps)
        """
        if addr == 'all':
            if verbose:
                print("WARNING: Setting the current and voltage of all bulbs at "+\
                  "once invalidates the tracking of I and V in the bulb dict. "+\
                  "Do not query V or I until the bulbs are set individually "+\
                  "again.")
            for port in self.bulbs['all']:
                self.bulbs['all'][port].write_register(registeraddress = 0x1000,\
                value = V*100, functioncode = 6)
                time.sleep(0.005)
                self.bulbs['all'][port].write_register(registeraddress = 0x1001,\
                value = I*100, functioncode = 6)
        else:
            self.bulbs[addr]['V'] = V
            self.bulbs[addr]['I'] = I
            self.bulbs[addr]['Inst'].write_register(registeraddress = 0x1000,\
                    value = V*100, functioncode = 6)
            time.sleep(0.005)
            self.bulbs[addr]['Inst'].write_register(registeraddress = 0x1001,\
                    value = I*100, functioncode = 6)
        return
    
    
    def Run_Bulb_VI(self, addr, V, I, t = 0, verbose = True):
        """
        Sets and activates the bulb for t seconds using the proper procedure.
        
        Args:
            addr: int, bulb address
            V: float, bulb voltage in [0,32] (volts)
            I: float, bulb current in [0,10] (amps)
            t: float, time to stay activated (seconds). Default is to stay on
               indefinitely.
        """
        self.Set_Bulb_VI(addr, 28, 10, verbose)
        self.Activate_Bulb(addr)
        time.sleep(0.3)
        self.Set_Bulb_VI(addr, V, I, verbose)
        
        if t > 0.005:
            time.sleep(t)
            self.Deactivate_Bulb(addr)
        
        return


    def Config_Check(self):
        """
        Sets the current and voltage of every power supply to be equal to their
        bulb address and activates them to make sure each supply is turned on
        and the RS-485 bus is connected properly.
        """
        for addr in self.bulbs:
            if addr != 'all':
                self.Set_Bulb_VI(addr, addr/100, 0)

        self.Activate_Bulb('all')

        return


    def Config_Warmup(self, T = 10, ballasts = 'New', duty_cycle = 0.5):
        """
        Runs the standard warm-up procedure for the bulb array
        """
        if ballasts == 'New':
            activate = 20
        else:
            activate = 28
        for i in range(T):
            print("Warmup cycle", i+1)
            self.Set_Bulb_VI('all', activate, 10, verbose = False)
            time.sleep(0.5)
            self.Activate_Bulb('all')
            time.sleep(4)
            self.Set_Bulb_VI('all', activate-4, 10, verbose = False)
            time.sleep(10)
            self.Deactivate_Bulb('all')
            time.sleep(15/duty_cycle-15)

        return


    def Activate_Bulb(self, addr):
        """
        Activate bulb
        """
        if addr == 'all':
            for port in self.bulbs['all']:
                self.bulbs['all'][port].write_register(registeraddress = 0x1006,\
                        value = 1, functioncode = 6)
        else:
            self.bulbs[addr]['Inst'].write_register(registeraddress = 0x1006,\
                    value = 1, functioncode = 6)
        return


    def Deactivate_Bulb(self,addr):
        """
        Deactivate bulb
        """
        if addr == 'all':
            for port in self.bulbs['all']:
                self.bulbs['all'][port].write_register(registeraddress = 0x1006,\
                        value = 0, functioncode = 6)
        else:
            self.bulbs[addr]['Inst'].write_register(registeraddress = 0x1006,\
                    value = 0, functioncode = 6)
        return


    def Scale_Rho_ne(self, rho, wp_max):
        """
        Uses an arctan barrier to map optimal parameters from the computational 
        inverse design library to plasma density values (dimensionalized, m^-3)

        Args:
            rho: Parameters being optimized
            wp_max: Approximate maximum non-dimensionalized plasma frequency
        """
        
        wp = (wp_max/1.5)*npa.arctan(rho/(wp_max/7.5))
        wp_dim = wp*c/self.a*2*np.pi
        ne = wp_dim**2*me*epso/e**2

        return ne


    def Scale_Rho_fp(self, rho, wp_max):
        """
        Uses an arctan barrier to map optimal parameters from the computational 
        inverse design library to plasma frequency values (dimensionalized, GHz)

        Args:
            rho: Parameters being optimized
            wp_max: Approximate maximum non-dimensionalized plasma frequency
        """
        
        fp = (wp_max/1.5)*np.arctan(np.abs(rho)/(wp_max/7.5))
        fp_dim = fp*c/self.a/10**9

        return fp_dim


    def BulbSetting_BOLSIG(self, fp, knob = 0.5, scale = 1.0):
        """
        Maps plasma frequency value in GHz to a current and voltage setting for
        the DC power supplies

        Args:
            fp: plasma frequency in GHz (NOT rad/s)
            knob: constant to tune experimental fit to lower and upper range of
                  BOLSIG cases. knob = 0 is low end and knob = 1 is high end.
            scale: Parameter that scales the overall plasma frequency values.
        """
        k = knob
        S = scale

        if fp/S < 0.21:
            return (0,0)
        
        elif fp/S >= 0.21 and fp/S < 0.42:
            I = (0.42/S-(0.03*k-0.47))**(1/(0.8-0.01*k))/(3.5*k+8.7) #A
            return (30, I)

        elif fp/S >= 0.42 and fp/S < 2.2 + 0.81*k:
            I = (fp/S-(0.03*k-0.47))**(1/(0.8-0.01*k))/(3.5*k+8.7) #A
            return (30, I)
        
        elif fp/S >= 2.2 + 0.81*k and fp/S < 3.32 + 1.3*k:
            if np.abs(fp/S-(3.32 + 1.3*k)) >= np.abs(fp/S-(2.2 + 0.81*k)):
                I = (2.2 + 0.81*k-(0.03*k-0.47))**(1/(0.8-0.01*k))/(3.5*k+8.7)
                return (30, I)
            else:
                V = (3.32 + 1.3*k+(5.75+1.1*k))**(1/(0.44+0.04*k))/(20+1.5*k)-0.5
                return (V, 10)

        elif fp/S >= 3.32 + 1.3*k and fp/S <= 11 + 4.6*k:
            V = (fp/S+(5.75+1.1*k))**(1/(0.44+0.04*k))/(20+1.5*k)-0.5
            return (V, 10)

        elif fp/S > 11 + 4.6*k:
            return (30,10)
        

    def BulbSetting_BOLSIG_NewDC(self, fp, knob = 0.5, scale = 1.0):
        """
        Maps plasma frequency value in GHz to a current and voltage setting for
        the DC power supplies

        Args:
            fp: plasma frequency in GHz (NOT rad/s)
            knob: constant to tune experimental fit to lower and upper range of
                  BOLSIG cases. knob = 0 is low end and knob = 1 is high end.
            scale: Parameter that scales the overall plasma frequency values.
        """
        k = knob
        S = scale
        Min_curr = S*((16/13)*((8.7+k*3.5)*0.08)**(-k*0.01+0.8)+(-0.47+k*0.03))
        Max_curr = S*((16/13)*((8.7+k*3.5)*0.2)**(-k*0.01+0.8)+(-0.47+k*0.03))
        Min_volt = S*(((5.25-k*1.7)*(6+0.5))**(k*0.1+0.55)+(k*0.725-0.3475))
        Max_volt = S*(((5.25-k*1.7)*(30+0.5))**(k*0.1+0.55)+(k*0.725-0.3475))
        
        if fp < Min_curr/2:
            return (0,0)
        
        elif fp >= Min_curr/2 and fp < Min_curr:
            I = (13*Min_curr/16/S+0.47-0.03*k)**(1/(0.8-0.01*k))/(3.5*k+8.7) #A
            return (30, I)

        elif fp >= Min_curr and fp < Max_curr:
            I = (13*fp/16/S+0.47-0.03*k)**(1/(0.8-0.01*k))/(3.5*k+8.7) #A
            return (30, I)
        
        elif fp >= Max_curr and fp < Min_volt:
            if fp < Max_curr+(Min_volt-Max_curr)/2:
                I = (13*Max_curr/16/S+0.47-0.03*k)**(1/(0.8-0.01*k))/(3.5*k+8.7)
                return (30, I)
            else:
                V = (Min_volt/S+0.3475-0.725*k)**(1/(0.55+0.1*k))/(5.25-1.7*k)-0.5
                return (V, 10)

        elif fp >= Min_volt and fp <= Max_volt:
            V = (fp/S+0.3475-0.725*k)**(1/(0.55+0.1*k))/(5.25-1.7*k)-0.5
            return (V, 10)

        elif fp > Max_volt:
            return (30,10)


    def Rho_to_Bulb(self, rho, wp_max, knob = 0.5, scale = 1.0,\
                    ballast = 'New'):
        """
        Accepts optimal parameter array (MUST BE FLATTENED) and returns (V,I)
        for each bulb.

        Args:
            rho: optimal parameter array (flattened) from PMMInverse library
            wp_max: Approximate maximum non-dimensionalized plasma frequency
            knob: constant to tune experimental fit to lower and upper range of
                  BOLSIG cases. knob = 0 is low end and knob = 1 is high end.
            scale: Parameter that scales the overall plasma frequency values.
        """
        BulbSet = np.zeros((rho.shape[0],2))
        fp = self.Scale_Rho_fp(rho, wp_max)

        for i in range(rho.shape[0]):
            if ballast == 'New':
                BulbSet[i,:] = self.BulbSetting_BOLSIG_NewDC(fp[i], knob, scale)
            else:
                BulbSet[i,:] = self.BulbSetting_BOLSIG(fp[i], knob, scale)

        return BulbSet


    def ArraySet_Rho(self, rho, wp_max, knob = 0.5, scale = 1.0,\
                     ballast = 'New'):
        """
        Accepts optimal parameter array (MUST BE FLATTENED) and activates the
        bulb array accordingly.

        Args:
            rho: optimal parameter array (flattened) from PMMInverse library
            wp_max: Approximate maximum non-dimensionalized plasma frequency
            knob: constant to tune experimental fit to lower and upper range of
                  BOLSIG cases. knob = 0 is low end and knob = 1 is high end.
            scale: Parameter that scales the overall plasma frequency values.
        """
        BulbSet = self.Rho_to_Bulb(rho, wp_max, knob, scale, ballast)
        if ballast == 'New':
            activate = 20
        else:
            activate = 28

        self.Set_Bulb_VI('all', activate, 10, verbose = False)
        time.sleep(0.005)
        try:
            self.Activate_Bulb('all')
        except:
            print('Trouble activating bulbs, trying one more time')
            time.sleep(1)
            try:
                self.Activate_Bulb('all')
            except:
                raise RuntimeError("Failed to activate bulbs twice, check"+\
                        " config.")
        time.sleep(1)
        self.Set_Bulb_VI('all', activate-4, 10, verbose = False)

        for i in range(rho.shape[0]):
            not_set = True
            tries = 0
            while not_set and tries < 5:
                try:
                    self.Set_Bulb_VI(i+1, BulbSet[i,0], BulbSet[i,1])
                    time.sleep(0.005)
                    not_set = False
                except:
                    tries += 1
                    print('Trouble setting bulb '+str(i+1)+', trying again')
                    time.sleep(1)
            if not_set:
                try:
                    self.Set_Bulb_VI(i+1, BulbSet[i,0], BulbSet[i,1])
                    time.sleep(0.005)
                except:
                    self.Deactivate_Bulb('all')
                    time.sleep(3)
                    self.Deactivate_Bulb('all')
                    raise RuntimeError("Failed to set bulb "+str(i+1)+\
                                       " six times. Check config.")

        return


    def Get_S21_S31(self):
        """
        Gets the freq array, S21 and S31 from the R&S VNA. Make sure the VNA is
        in the measurement state you want PRIOR to running this function. In 
        our case, that is with our cal set, 10000 points, Avg. factor 10.

        Args:
        """
        instr = RsInstrument(self.VNA)

        instr.write_str('TRIGger1:SEQuence:SOURce IMM')
        time.sleep(7)
        S21 = np.array(list(map(str,\
                instr.query_str('CALC1:DATA:TRAC? "Trc1", FDAT').split(','))),\
                                dtype = float)
        S31 = np.array(list(map(str,\
                instr.query_str('CALC1:DATA:TRAC? "Trc2", FDAT').split(','))),\
                                dtype = float)
        freq = np.array(list(map(str,\
                instr.query_str('CALC1:DATA:STIM?').split(','))), dtype = float)
        instr.write_str('TRIGger1:SEQuence:SOURce MAN')
        instr.close()

        return freq, S21, S31


    def Optimize_Demultiplexer(self, epochs, rho, fpm, k, S, f1, f2, df = 0.25,\
                               alpha = 0.01, sample = 12, p = 0.1,\
                               objective = 'comp', optimizer = 'grad. asc.',\
                               wu = 10, progress_dir = '.', fwin = [],\
                               duty_cycle = 0.5, show = True):
        """
        Performs an in-situ optimization procedure to produce a demultiplexer that 
        differentiates between freqeuncies f1 and f2.

        Args:
            epochs: int, Number of epochs (1 epoch = all bulbs adjusted once)
            rho: np.array, Starting parameters
            fpm: float, Max plasma frequency in GHz
            k: float in [0,1], Bulb fit knob
            S: float in [0,1], Bulb fit scale factor
            f1: float, frequency 1 in GHz
            f2: float, frequency 2 in GHz
            df: float, bandwidth around operating frequencies in GHz
            alpha: float, learning rate
            sample: int, How many bulbs at a time are modified to compute 
                    gradient wrt subset of parameters
            p: float, std. dev. of noise added to parameters
            objective: str, chooses objective function
            wu: int, # of minutes to warm up the array
        """
        if os.path.isfile(progress_dir+'/rho_Demult_%.1fGHz_fpm_%.1fGHz.csv'\
                                          %(f, fpm)):
            obj = self.Read_Params(progress_dir+\
                    '/obj_Demult_%.1fGHz_fpm_%.1fGHz.csv'%(f, fpm)).tolist()
            norms = self.Read_Params(progress_dir+\
                    '/norms_Demult_%.1fGHz_fpm_%.1fGHz.csv'%(f, fpm)).tolist()
            rho_evolution = self.Read_Params(progress_dir+\
                    '/rho_Demult_%.1fGHz_fpm_%.1fGHz.csv'%(f, fpm))
            rho = np.copy(rho_evolution[np.argmax(obj),:])
            print('='*80)
            print('NOTE: Optimizer starting over from sample %.0f of previous'+\
                  ' run'%(np.argmax(obj)+1))
            print('='*80)
            continuation = True
        else:
            rho_evolution = np.zeros((1,rho.shape[0]))
            rho_evolution[0,:] = np.copy(rho)
            continuation = False

        num_bulbs = rho.shape[0]
        bulb_idx = np.array(list(range(num_bulbs)))

        if num_bulbs%sample != 0:
            per_epoch = num_bulbs//sample + 1
        else:
            per_epoch = num_bulbs//sample
        
        print("="*80)
        print("Initiating demultiplexer optimization. You have chosen to run "+\
              str(epochs)+" epochs with a\nsample factor of "+str(sample)+".")
        print("Since there are "+str(num_bulbs)+" bulbs, this means that each epoch"+\
              " will take %.1f minutes, for\na total runtime of"%(per_epoch*38/60)+\
              " about %.1f minutes."%(per_epoch*epochs+wu))
        print("="*80)
        print("\n")
        print("="*80)
        print("Running array warmup.")
        print("="*80)
        self.Config_Warmup(T = wu, ballasts = 'New', duty_cycle = duty_cycle)

        print("\n")
        print("="*80)
        print("Array warm! Beginning optimization.")
        print("="*80)

        if not continuation:
            obj = []
            t1 = time.time()
            o, norms = self.Demult_Obj_Get(rho, fpm, k, S, f1, f2, df,\
                                        objective, norms = [],\
                                        duty_cycle = duty_cycle)
            obj.append(o)
            t2 = time.time()

            print("="*80)
            print("Epoch: %3d/%3d | Duration: %.2f secs | Value: %5e" %(0, epochs,\
                                                                    t2-t1, o))
            print("="*80)
            self.Save_Params(np.array(norms), progress_dir+\
                    '/norms_Demult_%.1f_%.1fGHz_fpm_%.1fGHz.csv'%(f1,f2,fpm))
        else:
            pass

        for e in range(epochs):
            t1 = time.time()
            bulbs = bulb_idx
            bulbs_left = num_bulbs
            for s in range(per_epoch):
                # Sample bulbs in array without replacement
                if sample < bulbs.shape[0]:
                    samp = np.random.choice(bulbs_left, sample, replace = False)
                    iter_bulbs = bulbs[samp]
                    bulbs = np.delete(bulbs, samp)
                    bulbs_left -= sample
                else:
                    iter_bulbs = bulbs

                # Adjust sampled bulbs
                rho_new = np.copy(rho)
                rho_new[iter_bulbs] = rho[iter_bulbs] +\
                                    np.random.normal(0, p, iter_bulbs.shape)

                # Compute objective
                o, norms = self.Demult_Obj_Get(rho, fpm, k, S, f1, f2,\
                                                df, objective, norms, duty_cycle)
                
                if optimizer == 'grad. asc.':
                    # Compute gradient
                    grad = (o-obj[len(obj)-1])/\
                            (rho_new[iter_bulbs]-rho[iter_bulbs]+1e-10)

                    # Gradient Ascent
                    rho[iter_bulbs] = rho_evolution[rho_evolution.shape[0]-1,\
                                                    iter_bulbs] + alpha*grad
                elif optimizer == 'greedy search':
                    if o > obj[len(obj)-1]:
                        rho[iter_bulbs] = rho_new[iter_bulbs]
                    else:
                        pass
                else:
                    raise RuntimeError("That optimizer is not implemented.")

                # Add to obj and rho tracker
                rho_evolution = np.row_stack([rho_evolution, rho])
                obj.append(o)
                print("Epoch: %3d/%3d | Sample: %3d/%3d | Value: %5e"\
                        %(e+1, epochs, s+1, per_epoch, o))

            t2 = time.time()
            print("="*80)
            print("Epoch: %3d/%3d | Duration: %.2f secs | Value: %5e"\
                        %(e+1, epochs, t2-t1, o))
            print("="*80)

            self.Save_Params(rho_evolution, progress_dir+\
                    '/rho_Demult_%.1f_%.1fGHz_fpm_%.1fGHz.csv'%(f1,f2,fpm))
            self.Save_Params(np.array(obj), progress_dir+\
                    '/obj_Demult_%.1f_%.1fGHz_fpm_%.1fGHz.csv'%(f1,f2,fpm))

        self.Demult_Run_And_Plot(progress_dir, rho, fpm, k, S, f1, f2,\
                            fwin = fwin, show = show)
        self.Plot_Obj(progress_dir+'/obj_Demult_%.1f_%.1fGHz_fpm_%.1fGHz.pdf'\
                              %(f1,f2,fpm), np.array(obj))

        return


    def Demult_Obj_Get(self, rho, fpm, k, S, f1, f2, df = 0.25,\
                            objective = 'comp', norms = [], duty_cycle = 0.5):
        """
        Run array and get one objective value evaluation.

        Args:
            See args for Optimize_Demultiplexer()
        """
        self.ArraySet_Rho(rho, self.f_a(fpm), knob = k, scale = S)
        time.sleep(1)
        freq, S21, S31 = self.Get_S21_S31()
        self.Deactivate_Bulb('all')
        time.sleep(1)
        self.Deactivate_Bulb('all')
        time.sleep(18/duty_cycle-20)

        if objective == 'comp':
            return Demult_Obj_Comp(freq/10**9, S21, S31, f1, f2, df, norms)
        elif objective == 'dB':
            return Demult_Obj_dB(freq/10**9, S21, S31, f1, f2, df, norms)
        else:
            raise RuntimeError("That objective has not been implemented")


    def Demult_Run_And_Plot(self, save_dir, rho, fpm, k, S, f1, f2,\
                                   fwin = [], show = True):
        """
        Run array and plot transmission spectrumn.

        Args:
            See args for Optimize_Demultiplexer() and Trans_Plot_2Port()
        """
        self.ArraySet_Rho(rho, self.f_a(fpm), knob = k, scale = S)
        time.sleep(1)
        freq, S21, S31 = self.Get_S21_S31()
        self.Deactivate_Bulb('all')
        time.sleep(1)
        self.Deactivate_Bulb('all')

        savepath = save_dir+'/Demult_%.1f_%.1fGHz_fpm_%.1fGHz.pdf'%(f1,f2,fpm)
        self.Trans_Plot_2Port(savepath, freq/10**9, S21, S31, fpm, f = [f1, f2],\
                              f_win = fwin, show = show)

        return


    def Optimize_Waveguide(self, epochs, rho, fpm, k, S, f, df = 0.5,\
                               alpha = 0.001, sample = 12, p = 0.01,\
                               objective = 'comp', optimizer = 'grad. asc.',\
                               wu = 10, progress_dir = '.', fwin = [],\
                               duty_cycle = 0.5, show = True,\
                               restart_obj = False):
        """
        Performs an in-situ optimization procedure to produce a waveguide/beam
        steering device that operates at freqeuncy f and directs signal into
        port 2.

        Args:
            epochs: int, Number of epochs (1 epoch = all bulbs adjusted once)
            rho: np.array, Starting parameters
            fpm: float, Max plasma frequency in GHz
            k: float in [0,1], Bulb fit knob
            S: float in [0,1], Bulb fit scale factor
            f: float, operating frequency in GHz
            df: float, bandwidth around operating frequencies in GHz
            alpha: float, learning rate
            sample: int, How many bulbs at a time are modified to compute 
                    gradient wrt subset of parameters
            p: float, std. dev. of noise added to parameters
            objective: str, chooses objective function
            wu: int, # of minutes to warm up the array
        """
        if os.path.isfile(progress_dir+'/rho_Wvg_%.1fGHz_fpm_%.1fGHz.csv'\
                                          %(f, fpm)):
            obj = self.Read_Params(progress_dir+\
                    '/obj_Wvg_%.1fGHz_fpm_%.1fGHz.csv'%(f, fpm)).tolist()
            norms = self.Read_Params(progress_dir+\
                    '/norms_Wvg_%.1fGHz_fpm_%.1fGHz.csv'%(f, fpm)).tolist()
            rho_evolution = self.Read_Params(progress_dir+\
                    '/rho_Wvg_%.1fGHz_fpm_%.1fGHz.csv'%(f, fpm))
            rho = np.copy(rho_evolution[np.argmax(obj),:])
            print('='*80)
            print('NOTE: Optimizer starting over from sample %.0f of previous'+\
                  ' run'%(np.argmax(obj)+1))
            print('='*80)
            continuation = True
        else:
            rho_evolution = np.zeros((1,rho.shape[0]))
            rho_evolution[0,:] = np.copy(rho)
            continuation = False
        
        num_bulbs = rho.shape[0]
        bulb_idx = np.array(list(range(num_bulbs)))

        if num_bulbs%sample != 0:
            per_epoch = num_bulbs//sample
        else:
            per_epoch = num_bulbs//sample
        
        print("="*80)
        print("Initiating waveguide optimization. You have chosen to run "+\
              str(epochs)+" epochs with a\nsample factor of "+str(sample)+".")
        print("Since there are "+str(num_bulbs)+" bulbs, this means that each epoch"+\
              " will take %.1f minutes, for\na total runtime of"%(per_epoch*38/60)+\
              " about %.1f minutes."%(per_epoch*epochs+wu))
        print("="*80)
        print("\n")
        print("="*80)
        print("Running array warmup.")
        print("="*80)
        self.Config_Warmup(T = wu, ballasts = 'New', duty_cycle = duty_cycle)

        print("\n")
        print("="*80)
        print("Array warm! Beginning optimization.")
        print("="*80)

        if not continuation:
            obj = []
            t1 = time.time()
            o, norms = self.Wvg_Obj_Get(rho, fpm, k, S, f, df,\
                                        objective, norms = [],\
                                        duty_cycle = duty_cycle)
            obj.append(o)
            t2 = time.time()

            print("="*80)
            print("Epoch: %3d/%3d | Duration: %.2f secs | Value: %5e" %(0, epochs,\
                                                                    t2-t1, o))
            print("="*80)
            self.Save_Params(np.array(norms), progress_dir+\
                '/norms_Wvg_%.1fGHz_fpm_%.1fGHz.csv'%(f, fpm))
        else:
            if restart_obj:
                t1 = time.time()
                o, norms = self.Wvg_Obj_Get(rho, fpm, k, S, f, df,\
                                        objective, norms = [],\
                                        duty_cycle = duty_cycle)
                obj.append(o)
                t2 = time.time()

                print("="*80)
                print("Epoch: %3d/%3d | Duration: %.2f secs | Value: %5e" %(0, epochs,\
                                                                    t2-t1, o))
                print("="*80)
                self.Save_Params(np.array(norms), progress_dir+\
                    '/norms_Wvg_%.1fGHz_fpm_%.1fGHz.csv'%(f, fpm))
            else:
                pass

        for e in range(epochs):
            t1 = time.time()
            bulbs = bulb_idx
            bulbs_left = num_bulbs
            for s in range(per_epoch):
                # Sample bulbs in array without replacement
                if 2*sample < bulbs.shape[0]:
                    samp = np.random.choice(bulbs_left, sample, replace = False)
                    iter_bulbs = bulbs[samp]
                    bulbs = np.delete(bulbs, samp)
                    bulbs_left -= sample
                else:
                    iter_bulbs = bulbs

                # Adjust sampled bulbs
                rho_new = np.copy(rho)
                rho_new[iter_bulbs] = rho[iter_bulbs] +\
                                    np.random.normal(0, p, iter_bulbs.shape)

                # Compute objective
                o, norms = self.Wvg_Obj_Get(rho, fpm, k, S, f,\
                                                df, objective, norms, duty_cycle)

                if optimizer == 'grad. asc.':
                    # Compute gradient
                    grad = (o-obj[len(obj)-1])/(rho_new[iter_bulbs]-rho[iter_bulbs]+1e-10)

                    # Gradient Ascent
                    rho[iter_bulbs] = rho_evolution[rho_evolution.shape[0]-1,\
                                                iter_bulbs] + alpha*grad
                elif optimizer == 'greedy search':
                    if o > obj[len(obj)-1]:
                        rho[iter_bulbs] = rho_new[iter_bulbs]
                    else:
                        pass
                else:
                    raise RuntimeError("That optimizer is not implemented.")

                # Add to obj and rho tracker
                rho_evolution = np.row_stack([rho_evolution, rho])
                obj.append(o)
                print("Epoch: %3d/%3d | Sample: %3d/%3d | Value: %5e"\
                        %(e+1, epochs, s+1, per_epoch, o))

            t2 = time.time()
            print("="*80)
            print("Epoch: %3d/%3d | Duration: %.2f secs | Value: %5e"\
                        %(e+1, epochs, t2-t1, o))
            print("="*80)

            self.Save_Params(rho_evolution, progress_dir+\
                    '/rho_Wvg_%.1fGHz_fpm_%.1fGHz.csv'%(f, fpm))
            self.Save_Params(np.array(obj), progress_dir+\
                    '/obj_Wvg_%.1fGHz_fpm_%.1fGHz.csv'%(f, fpm))

        self.Wvg_Run_And_Plot(progress_dir, rho, fpm, k, S, f,\
                            fwin = fwin, show = show)
        self.Plot_Obj(progress_dir+'/obj_Wvg_%.1fGHz_fpm_%.1fGHz.pdf'%(f, fpm),\
                      np.array(obj))

        return


    def Wvg_Obj_Get(self, rho, fpm, k, S, f, df = 0.25,\
                    objective = 'comp', norms = [], duty_cycle = 0.5):
        """
        Run array and get one objective value evaluation.

        Args:
            See args for Optimize_Waveguide()
        """
        self.ArraySet_Rho(rho, self.f_a(fpm), knob = k, scale = S)
        time.sleep(1)
        freq, S21, S31 = self.Get_S21_S31()
        self.Deactivate_Bulb('all')
        time.sleep(1)
        self.Deactivate_Bulb('all')
        time.sleep(18/duty_cycle-20)

        if objective == 'comp':
            return Waveguide_Obj_Comp(freq/10**9, S21, S31, f, df, norms)
        elif objective == 'dB':
            return Waveguide_Obj_dB(freq/10**9, S21, S31, f, df, norms)
        else:
            raise RuntimeError("That objective has not been implemented")


    def Wvg_Run_And_Plot(self, save_dir, rho, fpm, k, S, f,\
                                   fwin = [], show = True):
        """
        Run array and plot transmission spectrumn.

        Args:
            See args for Optimize_Waveguide() and Trans_Plot_2Port()
        """
        self.ArraySet_Rho(rho, self.f_a(fpm), knob = k, scale = S)
        time.sleep(1)
        freq, S21, S31 = self.Get_S21_S31()
        self.Deactivate_Bulb('all')
        time.sleep(1)
        self.Deactivate_Bulb('all')

        savepath = save_dir+'/Wvg_%.1fGHz_fpm_%.1fGHz.pdf'%(f,fpm)
        self.Trans_Plot_2Port(savepath, freq/10**9, S21, S31, fpm, f = [f],\
                              f_win = fwin, show = show)

        return


    def Save_Params(self, rho, savepath):
        """
        Wrapper for np.savetxt
        """
        return np.savetxt(savepath, rho, delimiter=',')


    def Read_Params(self, readpath, iteration = 0):
        """
        Reads optimization parameters from the computational invdes library.

        Args:
            readpath: read path. Must be csv.
        """
        if iteration > 0:
            rho = np.loadtxt(readpath, delimiter=',')
            return rho[iteration-1,:]
        elif iteration == 'last':
            rho = np.loadtxt(readpath, delimiter=',')
            return rho[rho.shape[0]-1,:]
        else:
            return np.loadtxt(readpath, delimiter=",")


    def f_GHz(self, f):
        """
        Returns dimensionalized frequency in GHz

        Args:
            f: frequency in a units
        """
        return f*c/self.a/10**9
    

    def f_a(self, f):
        """
        Returns nondimensionalized frequency in a units

        Args:
            f: frequency in GHz
        """
        return f*10**9/c*self.a


    def Trans_Plot_2Port(self, savepath, freq, S21, S31, fpm, f = [],\
                         f_win = [], show = True):
        """
        Creates plot of transmission spectrum for 2-port measurement

        Args:
            savepath: str
            freq: np.array, frequency in GHz
            S21: np.array
            S31: np.array
            fpm: float, max frequency in GHz
            f: list of floats, operating frequencies in GHz
        """
        fig, ax = plt.subplots(1,1,figsize=(9,6))

        ax.set_xlabel('Frequency (GHz)', fontsize = 30)
        ax.set_ylabel('$S_{31}$ and $S_{21}$ (dB)', fontsize = 30)
        for i in range(10):
            ax.axhline(y=-10*(i+1), color='grey', label='_nolegend_',\
                       linewidth = 1)
        ax.set_title('k = 0, S = 0.7, $f_{p,max}$ = '+str(fpm)+' GHz',\
                     fontsize = 30)
        ax.tick_params(labelsize = 27)
        ax.plot(freq, S21, linewidth = 5)
        ax.plot(freq, S31, linewidth = 5)
        for freq in f:
            ax.axvline(x=freq, color='k', linestyle='--')
        if len(f_win) > 0:
            ax.set_xlim(f_win)
        ax.set_ylim([-80,-10])
        ax.legend(['$S_{21}$','$S_{31}$'], bbox_to_anchor=[1.15, 0.5], loc = 'center', ncol = 1, fontsize = 27)

        plt.savefig(savepath, dpi=1500, bbox_inches='tight')
        if show:
            plt.show()

        return


    def Plot_Obj(self, savepath, obj, show = True):
        """
        Creates plot of objective evolution throughout optimization

        Args:
            savepath: str
            obj: np.array, objective function values
        """
        fig, ax = plt.subplots(1,1,figsize=(9,6))

        ax.set_xlabel('Samples', fontsize = 30)
        ax.set_ylabel('Objective', fontsize = 30)
        ax.tick_params(labelsize = 27)
        ax.plot(obj, linewidth = 5)

        plt.savefig(savepath, dpi=1500, bbox_inches='tight')
        if show:
            plt.show()

        return
