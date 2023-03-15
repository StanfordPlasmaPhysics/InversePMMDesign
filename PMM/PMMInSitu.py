"""
The module composed in this file is meant to facilitate both in-situ inverse
design and experimental control of plasma metamaterials composed of many
elements. The library is constructed for use with Longwei DC power supplies
which are connected on RS485 multidrop networks. This is NOT a general purpose
library and functions with a very specific experimental setup. For more
information, contact Jesse Rodriguez: jrodrig@stanford.edu
11/10/2022
"""

import minimalmodbus
import numpy as np
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
        self.Set_Bulb_VI(addr, 24, 10, verbose)
        self.Activate_Bulb(addr)
        time.sleep(0.25)
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


    def Config_Warmup(self, T = 10):
        """
        Runs the standard warm-up procedure for the bulb array
        """
        for addr in self.bulbs:
            if addr != 'all':
                self.Set_Bulb_VI(addr, addr/100, 0)
        for i in range(T):
            print("Warmup minute", i+1)
            self.Set_Bulb_VI('all', 24, 10, verbose = False)
            time.sleep(0.5)
            self.Activate_Bulb('all')
            time.sleep(15)
            self.Deactivate_Bulb('all')
            time.sleep(44.3)

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
        
        fp = (wp_max/1.5)*np.arctan(rho/(wp_max/7.5))
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
            I = (0.42/S-(0.03*k-0.47))**(1/(0.8-0.1*k))/(3.5+8.7*k) #A
            return (30, I)

        elif fp/S >= 0.42 and fp/S < 2.75 + 0.95*k:
            I = (fp/S-(0.03*k-0.47))**(1/(0.8-0.1*k))/(3.5+8.7*k) #A
            return (30, I)

        elif fp/S >= 2.75 + 0.95*k and fp/S <= 11 + 4.6*k:
            V = (fp/S+(5.75+1.1*k))**(1/(0.44+0.04*k))/(20+1.5*k)-0.5
            return (V, 10)

        elif fp/S > 11 + 4.6*k:
            return (30,10)


    def Rho_to_Bulb(self, rho, wp_max, knob = 0.5, scale = 1.0):
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
            BulbSet[i,:] = self.BulbSetting_BOLSIG(fp[i], knob, scale)

        return BulbSet


    def ArraySet_Rho(self, rho, wp_max, knob = 0.5, scale = 1.0):
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
        BulbSet = self.Rho_to_Bulb(rho, wp_max, knob, scale)

        self.Set_Bulb_VI('all', 24, 10, verbose = False)
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

        for i in range(rho.shape[0]):
            try:
                self.Set_Bulb_VI(i+1, BulbSet[i,0], BulbSet[i,1])
                time.sleep(0.005)
            except:
                print('Trouble setting bulb '+str(i+1)+', trying one more time')
                time.sleep(1)
                try:
                    self.Set_Bulb_VI(i+1, BulbSet[i,0], BulbSet[i,1])
                except:
                    raise RuntimeError("Failed to set"+\
                            " bulb "+str(i+1)+" twice. Check config.")

        return


    def Read_Params(self, readpath):
        """
        Reads optimization parameters from the computational invdes library.

        Args:
            readpath: read path. Must be csv.
        """
        return np.loadtxt(readpath, delimiter=",")


    def f_GHz(self, f):
        """
        Returns dimensionalized frequency in GHz

        Args:
            f: frequency in a units
        """
        return f*c/self.a/10**9
