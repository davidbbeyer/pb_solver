############################################################################################################
# This script compares the PB prediction for a charge regulating colloid with the Henderson-Hasselbalch
# theory and the Henderson-Hasselbalch theory coupled to a Donnan equilibrium in the limit of vanishing
# Bjerrum length.
############################################################################################################

import sys
import os
sys.path.append(os.path.abspath('../'))
from pb_solver import *

# Input parameters
bjerrum_length = 0.7/5000
r_colloid = 359 * 0.5
phi_colloid = 4e-3
n_surface_groups = 46000
pKa = 4.0

def alpha_henderson_hasselbalch(pH, pKa):
    """
    Calculate the degree of ionization of a weak acid using the Henderson-Hasselbalch equation.

    Args:
        pH: pH of the solution
        pKa: pKa of the weak acid
    Returns:
        Degree of ionization
    """
    return 1/(1 + 10**(pKa - pH))

def alpha_henderson_hasselbalch_donnan(pH, pKa, r_colloid, phi_colloid, n_surface_groups):
    """
    Calculate the degree of ionization of a weak acid using the Henderson-Hasselbalch coupled to a Donnan equilibrium.

    Args:
        pH: pH of the solution
        pKa: pKa of the weak acid
        r_colloid: radius of the colloid
        phi_colloid: volume fraction of the colloid
        n_surface_groups: number of surface groups on the colloid
    Returns:
        Degree of ionization
    """
    c_salt_res = max(10**(-pH), 10**(-(14-pH)))
    r_cell = r_colloid * np.power(phi_colloid, -1/3)
    concentration_surface_groups = n_surface_groups * 1.661 / (4 * np.pi * (r_cell**3-r_colloid**3) / 3)

    def partition_coefficient_donnan(concentration_charges, c_salt_res):
        """
        Calculate the partition coefficient of cations according to the Donnan theory.

        Args:
            concentration_charges: concentration of impermeable charges
            c_salt_res: salt concentration of the reservoir
        Returns:
            Partition coefficient
        """
        ratio = concentration_charges / (2 * c_salt_res)
        return ratio + np.sqrt(ratio**2 + 1)
    
    def calculate_deviation_alpha(pH_sys):
        """
        Calculate the deviation of the degree of ionization calculated using pH_sys directly and using the Donnan equilibrium.

        Args:
            pH_sys: pH of the system
        Returns:
            Deviation of the degree of ionization
        """
        alpha_pH_sys = alpha_henderson_hasselbalch(pH_sys, pKa)
        concentration_charges = concentration_surface_groups * alpha_pH_sys
        partition_coefficient = partition_coefficient_donnan(concentration_charges, c_salt_res)
        alpha = alpha_henderson_hasselbalch(pH-np.log10(partition_coefficient), pKa)
        return alpha - alpha_pH_sys
    
    pH_sys = scipy.optimize.root(calculate_deviation_alpha, x0=pH, options={"maxfev": int(1e9)}).x
    return alpha_henderson_hasselbalch(pH_sys, pKa)

solver = PB_Solver_Charge_Regulation(bjerrum_length=bjerrum_length, r_colloid=r_colloid, phi_colloid=phi_colloid, z_colloid_quenched=0.0, z_colloid_annealed=-n_surface_groups, pKa=pKa)
r, numerical_solution, z_colloid, z_colloid_eff = solver.solve_pb(pH_res=1.0)

pH_range = np.linspace(1.0, 13.0, 200)
pH_range_HH = np.linspace(1.0, 13.0, 200)
alphas = []
alphas_HH = [alpha_henderson_hasselbalch(pH, pKa) for pH in pH_range_HH]
alphas_HH_Donnan = [alpha_henderson_hasselbalch_donnan(pH, pKa, r_colloid, phi_colloid, n_surface_groups) for pH in pH_range_HH]
pH_values_boundary = []

for pH_res in tqdm.tqdm(pH_range):
    solver = PB_Solver_Charge_Regulation(bjerrum_length=bjerrum_length, r_colloid=r_colloid, phi_colloid=phi_colloid, z_colloid_quenched=0.0, z_colloid_annealed=-n_surface_groups, pKa=pKa)
    r, numerical_solution, z_colloid, z_colloid_eff = solver.solve_pb(pH_res=pH_res, initial_guess=numerical_solution.sol(r), initial_charge=z_colloid)
    
    alphas.append(np.abs(z_colloid.magnitude/n_surface_groups))

    reduced_cell_radius = (solver.r_cell/solver.bjerrum_length).magnitude
    phi_boundary = numerical_solution.sol(reduced_cell_radius)[0]
    pH_values_boundary.append(pH_res + phi_boundary * np.log10(np.exp(1)))

plt.plot(pH_range_HH, alphas_HH, linestyle="--", label="HH")
plt.plot(pH_range, alphas, color="black", label="PB")
plt.plot(pH_range_HH, alphas_HH_Donnan, linestyle="dotted", color="orange", label="HH+Donnan")
plt.xlabel(r"pH in the reservoir")
plt.ylabel(r"Degree of ionization $\alpha$")
plt.legend()
plt.show()
plt.close()

plt.plot(pH_values_boundary, alphas, color="black", label="PB")
plt.plot(pH_range_HH, alphas_HH, linestyle="dotted", label="HH", color="orange")
plt.xlabel(r"pH in the system")
plt.ylabel(r"Degree of ionization $\alpha$")
plt.legend()
plt.show()
plt.close()