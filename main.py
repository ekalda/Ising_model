import simplejson as json
from collections import OrderedDict
import ast
from ising_model import *
import time


def main():
    # importing the data from the configuration file
    print('reading the input data...')
    with open('config.dat', 'r') as f:
        config = json.load(f, object_pairs_hook=OrderedDict)
    temp = config["temperature"]
    confs = config["configurations"]
    alg = config["algorithm"]
    animate = ast.literal_eval(config["animate"])
    x_dim = config["x_dimension"]
    y_dim = config["y_dimension"]
    measure_mag = ast.literal_eval(config["measure_magnetisation"])
    measure_susc = ast.literal_eval(config["measure_susceptibility"])
    measure_avg_eng = ast.literal_eval(config["measure_average_energy"])
    measure_heat_cap = ast.literal_eval(config["measure_heat_capacity"])
    error_algorithm = config["error_algorithm"]
    system_type = ast.literal_eval(config["create_random_system"])
    if system_type: system_type = "random"
    else: system_type ="eq"

    print('setting up the simulation...')
    sim = IsingModel(temp=temp, confs=confs, animate=animate,
                 x_dim=x_dim, y_dim=y_dim, algorithm=alg, mag=measure_mag,
                 susc=measure_susc,
                 eng=measure_avg_eng, heat_cap=measure_heat_cap, error_alg=error_algorithm, system_type=system_type)

    find_T_dep = ast.literal_eval(config["find_temp_dependence"])
    start = time.time()
    if find_T_dep:
        sim.make_measurements(config["temp_range"], config["temp_inc"])
    else:
        print('simulating the Ising model at temperature ' + str(temp) + 'K...')
        sim.simulate()
    end = time.time()
    print('simulations finished!')
    print('total simulation time: %.2f s' % (end-start))



if __name__ == '__main__':
    main()