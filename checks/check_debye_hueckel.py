############################################################################################################
# This script compares the numerical solution of the Poisson-Boltzmann equation with the Debye-Hueckel 
# solution for a weakly charged colloid.
############################################################################################################

import sys
import os
sys.path.append(os.path.abspath('../'))
from pb_solver import *

# Input parameters
c_salt_res = 1e-4
bjerrum_length = 0.7
r_colloid = 100*0.7
phi_colloid = 1/8

solver = PB_Solver(bjerrum_length=bjerrum_length, r_colloid=r_colloid, phi_colloid=phi_colloid, c_salt_res=c_salt_res, z_colloid=0)
r, numerical_solution = solver.solve_pb()

charges = np.linspace(1, 10, 10)
for charge in tqdm.tqdm(charges):   
    solver = PB_Solver(bjerrum_length=bjerrum_length, r_colloid=r_colloid, phi_colloid=phi_colloid, c_salt_res=c_salt_res, z_colloid=charge)
    r, numerical_solution = solver.solve_pb(initial_guess=numerical_solution.sol(r))

plt.plot(r, numerical_solution.sol(r)[0], label=r"PB", color="black")
plt.plot(r, solver.solve_dh(r)[0], linestyle="--", label=r"DH", color="orange")
plt.xlabel(r"radial coordinate r")
plt.ylabel(r"reduced electrostatic potential")
plt.legend()
plt.show()