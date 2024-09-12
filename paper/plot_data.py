import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import csv
import pandas as pd
import scipy


# Plot settings
import matplotlib as mpl
mpl.rcParams['lines.markersize'] = 5
mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['axes.titlesize'] = 18

# Get the list of standard colors
standard_colors = list(mcolors.TABLEAU_COLORS.keys())

# Load the data
pKa_range = []
bare_charges_pH_7 = []
effective_charges_pH_7 = []
bare_charges_pH_55 = []
effective_charges_pH_55 = []

filename = './data.csv'
data = pd.read_csv(filename, header=None)
column_names = ["pKa", "bare_charge_pH_7", "effective_charge_pH_7", "bare_charge_pH_55", "effective_charge_pH_55"]
data.columns = column_names
data = data.sort_values(by="pKa")

##### Plot the data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4*0.9))

## Plot of the charges

# Bare charge vs pK
pK_min = 3.0
pK_max = 8.0
charge_pH_7 = 10000
charge_pH_55 = 1340

f = scipy.interpolate.interp1d(np.abs(data["bare_charge_pH_7"]), data["pKa"])
pKa_7 = f(charge_pH_7)
f = scipy.interpolate.interp1d(np.abs(data["bare_charge_pH_55"]), data["pKa"])
pKa_55 = f(charge_pH_55)
print(pKa_7, pKa_55)

ax1.vlines(pKa_55, 0, charge_pH_55, color="gray", linestyle="dotted", linewidth=1.4)
ax1.vlines(pKa_7, 0, charge_pH_7, color="gray", linestyle="dotted", linewidth=1.4)
ax1.hlines(charge_pH_55, -20, 20, color=standard_colors[1], linestyle="--", linewidth=1.4)
ax1.plot(data["pKa"], np.abs(data["bare_charge_pH_7"]), label=r"$\text{pH} = 7.0$", color=standard_colors[0])
ax1.hlines(charge_pH_7, -20, 20, color=standard_colors[0], linestyle="--", linewidth=1.4)
ax1.plot(data["pKa"], np.abs(data["bare_charge_pH_55"]), label=r"$\text{pH} = 5.5$", color=standard_colors[1])
ax1.set_xlabel(r"$\text{p}K_{\text{A}}$")
ax1.set_ylabel(r"bare charge  $Z$ in $e$")
ax1.set_xlim((pK_min, pK_max))
#ax1.set_xlim((0.0, 10.0))
ax1.set_ylim((0,15000))
ax1.text(-0.1, 1.1, '(a)', transform=ax1.transAxes, fontsize=14, va='top', ha='right')
#ax1.legend(frameon=False)
ax1.legend(facecolor='white', framealpha=1, edgecolor="white")

# Renormalized charge vs bare charge
charge_range = np.linspace(0, 12000, 1000)
ax2.plot(charge_range, charge_range, linestyle="--", color="black", linewidth=1.4)
ax2.plot(data["bare_charge_pH_7"].abs(), data["effective_charge_pH_7"].abs(), label=r"$\text{pH} = 7.0$")
ax2.plot(data["bare_charge_pH_55"].abs(), data["effective_charge_pH_55"].abs(), label=r"$\text{pH} = 5.5$")
ax2.set_xlabel(r"bare charge $Z$ in $e$")
ax2.set_ylabel(r"renormalized charge $Z_{\text{eff}}$ in $e$")
ax2.set_xlim((0, 12000))
ax2.set_ylim((0, 1.1*np.max(data["effective_charge_pH_55"].abs())))
ax2.yaxis.set_ticks_position('both')
ax2.text(1.3, 1.1, '(b)', transform=ax1.transAxes, fontsize=14, va='top', ha='right')
ax2.legend(frameon=False)

plt.tight_layout()
plt.subplots_adjust(wspace=0.4)
plt.show()
