import numpy as np
import matplotlib.pyplot as plt
#import profiler
import simplejson as json
from collections import OrderedDict
import ast
from matplotlib import animation
from tqdm import tqdm
from pylab import cm
import time
import os
import numexpr as ne


class IsingModel(object):
    def __init__(self, temp=1.5, confs=1000, x_dim=50, y_dim=50,
                 animate=False, algorithm="glauber", mag=False, susc=False,
                 eng=False, heat_cap=False, error_alg='bootstrap', system_type="eq"):
        self.T = temp  # reduced units
        self.k = 1
        self.J = 1
        self.n = x_dim * y_dim  # size of one sweep
        self.configs = confs # 1 configuration = 1 sweep
        if system_type == "random":
            self.init_sys = self.create_random_system(x_dim, y_dim)
        elif system_type == "eq":
            self.init_sys = self.create_equilibrium_system(x_dim, y_dim)
        self.sys = self.init_sys
        self.x_sys = x_dim
        self.y_sys = y_dim
        self.anim = animate
        self.alg = algorithm
        self.error_alg = error_alg
        self.out_dir = None
        # measurement variables
        self.measure_mag = mag
        self.measure_avg_energy = eng
        self.measure_susc = susc
        self.measure_heat_cap = heat_cap
        # arrays that hold the energy and magnetisation values
        self.mags = []
        self.energies = []

    # creating a random configuration of spins
    def create_random_system(self, x=50, y=50):
        print('creating a random system...')
        sys = np.random.randint(0, 2, size=(x, y))
        sys[sys == 0] = -1
        return sys

    # creating a system where all spins are pointing either up or down
    def create_equilibrium_system(self, x=50, y=50):
        print('creating a system with all the spins pointing up...')
        return np.ones((x, y))

    # finding the total energy of a system
    def find_system_energy(self):
        E = 0
        for i in range(self.x_sys):
            for j in range(self.y_sys):
                spin = self.sys[i, j]
                spin_right = self.sys[(i + 1) % self.x_sys, j]
                spin_down = self.sys[i, (j + 1) % self.y_sys]
                E += -self.J * spin * (spin_right + spin_down)
        return E

    # finding the total magnetisation of the system
    def find_total_magnetisation(self):
        return float(sum(sum(self.sys))) / float(self.n)

    # energy difference in case a spin at [x,y] gets flipped
    def find_delta_E(self, x, y):
        i = self.sys[x, y]
        i_up = self.sys[x, (y - 1 + self.y_sys) % self.y_sys]
        i_down = self.sys[x, (y + 1) % self.y_sys]
        i_right = self.sys[(x + 1) % self.x_sys, y]
        i_left = self.sys[(x - 1 + self.x_sys) % self.x_sys, y]
        delta_E = self.J * i * (i_up + i_down + i_right + i_left)
        return 2 * delta_E


    # determine if a spin should be flipped according to Metropolis algorithm and flip it, if need be
    # spins is a list of tuples that gives the coordinates of spins to be flipped
    def flip_spins(self, dE, spins):
        # always flip a spin if it would be energetically favourable
        if dE < 0:
            for spin in spins:
                self.sys[spin] *= -1
        # otherwise flip spin with probability exp(-dE/(k_b*T))
        else:
            rand = np.random.uniform()
            if rand < np.exp(-dE / (self.k * self.T)):
                for spin in spins:
                    self.sys[spin] *= -1

    def visualise(self):
        im = plt.imshow(self.sys, cmap=cm.RdBu)
        plt.ion()
        plt.show()
        plt.pause(0.00001)

    # choose which algorithm to use
    def simulate(self):
        if self.alg == "glauber":
            self.simulate_glauber()
        elif self.alg == "kawasaki":
            self.simulate_kawasaki()
        else:
            print('invalid algorithm name')
            exit()

    # @profiler.do_cprofile
    def simulate_glauber(self):
        print('simulating Ising model with Glauber dynamics...')
        for i in range(self.n * self.configs):
            x = np.random.randint(0, self.x_sys)
            y = np.random.randint(0, self.y_sys)
            spins = [(x, y)]
            # find change in system energy if a spin gets flipped
            dE = self.find_delta_E(x, y)
            # do (or do not) flip the spin
            self.flip_spins(dE, spins)
            if (i % (self.n * 10) == 0) and self.anim:
                self.visualise()
            # make measurements after system has equilibrated (takes ca 100 sweeps) and measure parameters every 10 sweeps
            if (i > (self.n * 100)) and (i % (self.n * 10) == 0):
                self.collect_data()
        #print temperature, total energy, total magnetisation, heat capacity and susceptibility on the terminal
        self.print_stats()

    # @profiler.do_cprofile
    def simulate_kawasaki(self):
        print('simulating Ising model with Kawasaki dynamics...')
        for i in range(self.n * self.configs):
            # generate random spins. ex: x_i - x coord of spin i
            x_i = np.random.randint(0, self.x_sys)
            y_i = np.random.randint(0, self.y_sys)
            x_j = np.random.randint(0, self.x_sys)
            y_j = np.random.randint(0, self.y_sys)
            # generate new spin with the same spin were chosen
            if x_i == x_j and y_i == y_j:
                x_i = np.random.randint(0, self.x_sys)
                y_i = np.random.randint(0, self.y_sys)
            # there is no change to energy when spins with same sign were chosen
            if self.sys[x_i, y_i] == self.sys[x_j, y_j]:
                continue
            spins = [(x_i, y_i), (x_j, y_j)]
            dE_i = self.find_delta_E(x_i, y_i)
            dE_j = self.find_delta_E(x_j, y_j)
            total_dE = dE_i + dE_j
            # correction to energy in case spins are next to each other
            if ((abs(x_i - x_j) == 1 or (abs(x_i - x_j) == (self.x_sys - 1))) and (y_i == y_j)) or ((abs(y_i - y_j) == 1 or (abs(y_i - y_j) == (self.y_sys - 1))) and (x_i == x_j)):
                print('IF', spins)
                total_dE += 4 * self.J
            self.flip_spins(total_dE, spins)
            if (i % (self.n * 10) == 0) and self.anim:
                self.visualise()
            if (i > (self.n * 100)) and (i % (self.n * 10) == 0):
                self.collect_data()
        self.print_stats()

    def print_stats(self):
        print('T = ' + str(self.T))
        print('E = ' + str(np.mean(self.energies)))
        print('M = ' + str(np.mean(self.mags)))
        print('C = ' + str(1. / (self.n * self.k * self.T ** 2) * np.var(self.energies)))
        print('X = ' + str(1. / (self.n * self.k * self.T) * np.var(self.mags)))

    def collect_data(self):
        if self.measure_mag or self.measure_susc: self.mags.append(
            self.find_total_magnetisation())
        if self.measure_avg_energy or self.measure_heat_cap:
            self.energies.append(self.find_system_energy())

    # TODO: there is a nicer way to write to an output file
    def write_data(self):
        if self.measure_mag:
            with open(os.path.join(self.out_dir, 'av_M.txt'), 'a+') as f:
                f.write(str(self.T) + ' ' + str(np.mean(self.mags)) + ' ' + str(np.std(self.mags)) + '\n')
        if self.measure_susc:
            with open(os.path.join(self.out_dir, 'av_X.txt'), 'a+') as f:
                f.write(str(self.T) + ' ' + str(
                    (1. / (self.n * self.k * self.T)) * np.var(
                        self.mags)) + ' ' + str(self.find_error(self.mags)) + '\n')
        if self.measure_avg_energy:
            with open(os.path.join(self.out_dir, 'av_E.txt'), 'a+') as f:
                f.write(str(self.T) + ' ' + str(np.mean(self.energies)) + ' ' + str(np.std(self.energies)) + '\n')
        if self.measure_heat_cap:
            with open(os.path.join(self.out_dir, 'av_C.txt'), 'a+') as f:
                f.write(str(self.T) + ' ' + str(
                    1. / (self.n * self.k * self.T ** 2) * np.var(
                        self.energies)) + ' ' + str(self.find_error(self.energies)) + '\n')

    #@profiler.do_cprofile
    def make_measurements(self, temp_interval, inc):
        # making the output directory
        self.out_dir = self.create_output_dir()
        temp_init = temp_interval[0]
        temp_final = temp_interval[1]
        # initialising temperature variable
        temp = temp_init
        while temp < temp_final:
            self.T = temp
            self.simulate()
            self.write_data()
            # clear the arrays that hold E and M values for each measurement
            self.mags = []
            self.energies = []
            temp += inc

    #method that checks which error estimation algortihm to use
    def find_error(self, quantity):
        if self.error_alg == 'bootstrap':
            return self.find_error_bootstrap(quantity)
        elif self.error_alg == 'jacknife':
            return self.find_error_jacknife(quantity)
        else:
            print('invalid error algortihm name')
            exit()

    def find_error_bootstrap(self, quantity):
        c_values = []
        new_quantity_array = []
        # creating an array with 100 values of heat capacity or susceptibility
        for i in range(100):
            # creating a new set of experimental values by randomly choosing values from the measurements (one measurement can be chosen more than once)
            for j in range(len(quantity)):
                rand = np.random.randint(0,len(quantity))
                new_quantity_array.append(quantity[rand])
            c_values.append(1. / (self.n * self.k * self.T ** 2) * np.var(
                        new_quantity_array))
            new_quantity_array = []
        # finding and returning the error
        if quantity == self.energies:
            return np.std(c_values)
        else:
            # I'm dealing with susceptibilities
            x_values = [self.T*c for c in c_values]
            return np.std(x_values)

    def find_error_jacknife(self, quantity):
        # heat capacity (or susceptibility, depends on an argument quantity)
        c = 1. / (self.n * self.k * self.T ** 2) * np.var(quantity)
        c_i_array = []
        for i in range(len(quantity)):
            # removing one measurement from the array
            new_array = quantity[0:i] + quantity[i+1:]
            # this is heat capacity!
            c_i_array.append(1. / (self.n * self.k * self.T ** 2) * np.var(
                        new_array))
        if quantity == self.energies:
            return np.sqrt(sum([(c_i-c)**2 for c_i in c_i_array]))
        else:
            # turning heat capacity into susceptibility
            x = c * self.T
            x_i_array = [self.T * c_i for c_i in c_i_array]
            return np.sqrt(sum([(x_i-x)**2 for x_i in x_i_array]))

    def create_output_dir(self):
        timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
        new_dir_name = 'ising_model_' + timestr
        new_dir = os.path.abspath(new_dir_name)
        print('Creating new output directory:\n %s' % new_dir)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        return new_dir