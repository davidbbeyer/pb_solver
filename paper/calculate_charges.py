import sys
import os
import csv
sys.path.append(os.path.abspath('..'))
from pb_solver import *

# Input parameters
bjerrum_length = 0.7
r_colloid = 359 * 0.5
phi_colloid = 4e-3
n_surface_groups = 46000#134000

def interpolate_charge(pH_range, charges, pH):
    """
    Interpolate the charge at a given pH using a given set of charges.
    
    Args:
        pH_range: pH values of the charges
        charges: charges at the pH values
        pH: pH at which the charge is to be interpolated
    Returns:    
        Interpolated charge
    """
    f = scipy.interpolate.interp1d(np.asarray(pH_range), np.asarray(charges))
    return f(pH)

def calculate_charges(pKa, pH_range):
    """
    Calculate the bare and effective charges of a colloid for a pH-values 7.0 and 5.5.
    
    Args:
        pKa: pKa of the weak acid
        pH_range: pH values for which the charges are to be calculated
    Returns:
        pKa, Bare charge at pH 7.0, effective charge at pH 7.0, bare charge at pH 5.5, effective charge at pH 5.5
    """
    solver = PB_Solver_Charge_Regulation(bjerrum_length=bjerrum_length, r_colloid=r_colloid, phi_colloid=phi_colloid, z_colloid_quenched=0.0, z_colloid_annealed=-n_surface_groups, pKa=pKa)
    pH_values_boundary, bare_charges, effective_charges = solver.calculate_bare_and_effective_charge(pH_range=pH_range)

    bare_charge_pH_7 = interpolate_charge(pH_values_boundary, bare_charges, 7.0)
    effective_charge_pH_7 = interpolate_charge(pH_values_boundary, effective_charges, 7.0)

    bare_charge_pH_55 = interpolate_charge(pH_values_boundary, bare_charges, 5.5)
    effective_charge_pH_55 = interpolate_charge(pH_values_boundary, effective_charges, 5.5)

    return pKa, bare_charge_pH_7, effective_charge_pH_7, bare_charge_pH_55, effective_charge_pH_55

pKa_range = np.linspace(0.0, 7.0, 100)
pH_range = np.linspace(1.0, 13.0, 200)
existing_pKs = []

# Load existing data
filename = './data_old.csv'
if os.path.exists(filename):
    with open(filename, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:
                existing_pKs.append(float(row[0])) 

for pKa in pKa_range:
    print(pKa)
    if pKa not in existing_pKs:
        values = calculate_charges(pKa, pH_range)
    
        # Write out the data
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(values)
