############################################################################################################
# This script compares the Donnan potential calculated by the Poisson-Boltzmann solver in the limit of 
# vanishing Bjerrum length with the analytical solution for a range of charges.
############################################################################################################

import sys
import os
sys.path.append(os.path.abspath('../'))
from pb_solver import *

# Input parameters
c_salt_res = 1e-4
bjerrum_length = 0.7/1000
r_colloid = 100*0.7
phi_colloid = 1/3

def calculate_donnan_potential(r_colloid, phi_colloid, c_salt_res, z_colloid):
     """
     Calculate the Donnan potential for a given set of parameters
     
     Args:
        r_colloid: radius of the colloid
        phi_colloid: volume fraction of the colloid
        c_salt_res: salt concentration of the reservoir
        z_colloid: charge of the colloid
     Returns:
        Donnan potential
     """
     r_cell = r_colloid * np.power(phi_colloid, -1/3)
     concentration_charges = z_colloid * 1.661 / (4 * np.pi * (r_cell**3-r_colloid**3) / 3) # 1.661 converts between #/nm^3 and mol/L
     ratio = concentration_charges / (2 * c_salt_res)
     return np.log(ratio + np.sqrt(ratio**2 + 1))

donnan_pb = []
donnan_analytical = []

solver = PB_Solver(bjerrum_length=bjerrum_length, r_colloid=r_colloid, phi_colloid=phi_colloid, c_salt_res=c_salt_res, z_colloid=0)
r, numerical_solution = solver.solve_pb()

charges = np.linspace(1, 3000, 200)
for charge in tqdm.tqdm(charges):   
    solver = PB_Solver(bjerrum_length=bjerrum_length, r_colloid=r_colloid, phi_colloid=phi_colloid, c_salt_res=c_salt_res, z_colloid=charge)
    r, numerical_solution = solver.solve_pb(initial_guess=numerical_solution.sol(r))

    donnan_pb.append(numerical_solution.sol(r)[0][-1])
    donnan_analytical.append(calculate_donnan_potential(r_colloid, phi_colloid, c_salt_res, charge))

plt.plot(charges, donnan_pb, label=r"PB", color="black")
plt.plot(charges, donnan_analytical, linestyle="--", label=r"Analytical", color="orange")
plt.xlabel(r"Charge")
plt.ylabel(r"reduced Donnan potential")
plt.legend()
plt.show()