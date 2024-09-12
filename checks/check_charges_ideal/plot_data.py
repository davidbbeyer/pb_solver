import matplotlib.pyplot as plt
import numpy as np
import csv

# Input parameters
bjerrum_length = 0.7/10000
r_colloid = 359 * 0.5
phi_colloid = 4e-3
n_surface_groups = 46000

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

# Load the data
pKa_range = []
bare_charges_pH_7 = []
effective_charges_pH_7 = []
bare_charges_pH_55 = []
effective_charges_pH_55 = []

filename = './data.csv'
with open(filename, 'r', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        if row:
            pKa_range.append(float(row[0])) 
            bare_charges_pH_7.append(float(row[1]))
            effective_charges_pH_7.append(float(row[2]))
            bare_charges_pH_55.append(float(row[3]))
            effective_charges_pH_55.append(float(row[4]))

# Calculate analytical predictions
bare_charges_pH_7_HH = [n_surface_groups * alpha_henderson_hasselbalch(7, pKa) for pKa in pKa_range]
bare_charges_pH_55_HH = [n_surface_groups * alpha_henderson_hasselbalch(5.5, pKa) for pKa in pKa_range]


# Plot the data
plt.plot(pKa_range, np.abs(bare_charges_pH_7), marker="x", linestyle="--", label="PB", color="black")
plt.plot(pKa_range, np.abs(bare_charges_pH_7_HH), marker="o", label="HH", color="orange")
plt.xlabel(r"pKa")
plt.ylabel(r"Bare charge at pH 7")
plt.legend()
plt.show()

plt.plot(pKa_range, np.abs(bare_charges_pH_55), marker="x", linestyle="--", label="PB", color="black")
plt.plot(pKa_range, np.abs(bare_charges_pH_55_HH), marker="o", label="HH", color="orange")
plt.xlabel(r"pKa")
plt.ylabel(r"Bare charge at pH 5.5")
plt.legend()
plt.show()