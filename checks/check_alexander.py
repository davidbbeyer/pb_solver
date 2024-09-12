############################################################################################################
# This script checks that we can recover the effective charge of a colloid with a given charge and volume 
# fraction as given in the paper of Alexander et al. (1984).
############################################################################################################

import sys
import os
sys.path.append(os.path.abspath('../'))
from pb_solver import *

# Input parameters
bjerrum_length = 0.7
r_colloid = 100*0.7
phi_colloid = 1/8
charges = np.linspace(1, 5000, 100)

# 500 salt ion pairs
n_salt = 500
solver = PB_Solver_Constant_Salt(bjerrum_length=bjerrum_length, r_colloid=r_colloid, phi_colloid=phi_colloid, n_salt=n_salt, z_colloid=0)
r, numerical_solution, c_salt_res = solver.solve_pb()

effective_charges = []

for charge in tqdm.tqdm(charges):   
    solver = PB_Solver_Constant_Salt(bjerrum_length=bjerrum_length, r_colloid=r_colloid, phi_colloid=phi_colloid, n_salt=n_salt, z_colloid=charge)
    effective_charges.append(solver.calculate_effective_charge_trizac(initial_guess=numerical_solution.sol(r)))

plt.plot(charges, effective_charges, label="500 salt ion pairs")


# 1000 salt ion pairs
n_salt = 1000
solver = PB_Solver_Constant_Salt(bjerrum_length=bjerrum_length, r_colloid=r_colloid, phi_colloid=phi_colloid, n_salt=n_salt, z_colloid=0)
r, numerical_solution, c_salt_res = solver.solve_pb()

effective_charges = []

for charge in tqdm.tqdm(charges):   
    solver = PB_Solver_Constant_Salt(bjerrum_length=bjerrum_length, r_colloid=r_colloid, phi_colloid=phi_colloid, n_salt=n_salt, z_colloid=charge)
    effective_charges.append(solver.calculate_effective_charge_trizac(initial_guess=numerical_solution.sol(r)))

plt.plot(charges, effective_charges, label="1000")


plt.plot(charges, charges, linestyle="--")
plt.legend()
plt.xlabel(r"Charge")
plt.ylabel(r"Effective charge")
plt.hlines(1000, 0, 5000, linestyle="--", color="gray")
plt.hlines(1500, 0, 5000, linestyle="--", color="gray")
plt.hlines(1750, 0, 5000, linestyle="--", color="gray")
plt.xlim(0, 5000)
plt.ylim(0, 2000)
plt.show()
