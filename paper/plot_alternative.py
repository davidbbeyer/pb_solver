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

filename = './data_0004.csv'
data = pd.read_csv(filename, header=None)
column_names = ["pKa", "bare_charge_pH_7", "effective_charge_pH_7", "bare_charge_pH_55", "effective_charge_pH_55"]
data.columns = column_names
data = data.sort_values(by="pKa")

filename = './data_0003.csv'
data_0003 = pd.read_csv(filename, header=None)
column_names = ["pKa", "bare_charge_pH_7", "effective_charge_pH_7", "bare_charge_pH_55", "effective_charge_pH_55"]
data_0003.columns = column_names
data_0003 = data_0003.sort_values(by="pKa")

filename = './data_0005.csv'
data_0005 = pd.read_csv(filename, header=None)
column_names = ["pKa", "bare_charge_pH_7", "effective_charge_pH_7", "bare_charge_pH_55", "effective_charge_pH_55"]
data_0005.columns = column_names
data_0005 = data_0005.sort_values(by="pKa")


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
f = scipy.interpolate.interp1d(data_0003["pKa"], np.abs(data_0003["bare_charge_pH_7"]))
bare_charge_0003_7 = f(pKa_7) 
f = scipy.interpolate.interp1d(data_0005["pKa"], np.abs(data_0005["bare_charge_pH_7"]))
bare_charge_0005_7 = f(pKa_7) 


f = scipy.interpolate.interp1d(np.abs(data["bare_charge_pH_55"]), data["pKa"])
pKa_55 = f(charge_pH_55)
f = scipy.interpolate.interp1d(data_0003["pKa"], np.abs(data_0003["bare_charge_pH_55"]))
bare_charge_0003_55 = f(pKa_55) 
f = scipy.interpolate.interp1d(data_0005["pKa"], np.abs(data_0005["bare_charge_pH_55"]))
bare_charge_0005_55 = f(pKa_55) 


print(pKa_7, pKa_55)

ax1.vlines(pKa_55, 0, charge_pH_55, color="gray", linestyle="dotted", linewidth=1.4)
ax1.vlines(pKa_7, 0, charge_pH_7, color="gray", linestyle="dotted", linewidth=1.4)
ax1.hlines(charge_pH_55, -20, 20, color='#0101FF', linestyle="--", linewidth=1.4)
ax1.plot(data["pKa"], np.abs(data["bare_charge_pH_7"]), label=r"$\text{pH} = 7.0$", color='black')

# Plot uncertainty
pK_range = np.linspace(3.0, 8.0, 1000)
charge_0003 = scipy.interpolate.interp1d(data_0003["pKa"], np.abs(data_0003["bare_charge_pH_7"]))
charge_0005 = scipy.interpolate.interp1d(data_0005["pKa"], np.abs(data_0005["bare_charge_pH_7"]))
ax1.fill_between(pK_range, charge_0003(pK_range), charge_0005(pK_range), facecolor='black', alpha=0.3, zorder=-1)

ax1.hlines(charge_pH_7, -20, 20, color='black', linestyle="--", linewidth=1.4)
ax1.plot(data["pKa"], np.abs(data["bare_charge_pH_55"]), label=r"$\text{pH} = 5.5$", color='#0101FF')

# Plot uncertainty
pK_range = np.linspace(3.0, 8.0, 1000)
charge_0003 = scipy.interpolate.interp1d(data_0003["pKa"], np.abs(data_0003["bare_charge_pH_55"]))
charge_0005 = scipy.interpolate.interp1d(data_0005["pKa"], np.abs(data_0005["bare_charge_pH_55"]))
ax1.fill_between(pK_range, charge_0003(pK_range), charge_0005(pK_range), color='#0101FF', alpha=0.4, zorder=-1)

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
ax2.plot(data["bare_charge_pH_7"].abs(), data["effective_charge_pH_7"].abs(), label=r"$\text{pH} = 7.0$", color='black')

# Plot uncertainty
charge_range = np.linspace(50, 12000, 1000)
ren_charge_0003 = scipy.interpolate.interp1d(data_0003["bare_charge_pH_7"].abs(), data_0003["effective_charge_pH_7"].abs())
ren_charge_0005 = scipy.interpolate.interp1d(data_0005["bare_charge_pH_7"].abs(), data_0005["effective_charge_pH_7"].abs())
ax2.fill_between(charge_range, ren_charge_0003(charge_range), ren_charge_0005(charge_range), color='black', alpha=0.3, zorder=-1)

ax2.plot(data["bare_charge_pH_55"].abs(), data["effective_charge_pH_55"].abs(), label=r"$\text{pH} = 5.5$", color='#0101FF')

# Plot uncertainty
ren_charge_0003 = scipy.interpolate.interp1d(data_0003["bare_charge_pH_55"].abs(), data_0003["effective_charge_pH_55"].abs())
ren_charge_0005 = scipy.interpolate.interp1d(data_0005["bare_charge_pH_55"].abs(), data_0005["effective_charge_pH_55"].abs())
ax2.fill_between(charge_range, ren_charge_0003(charge_range), ren_charge_0005(charge_range), color='#0101FF', alpha=0.4, zorder=-1)


# Plot vertical and horizontal lines
f = scipy.interpolate.interp1d(np.abs(data["bare_charge_pH_7"]), np.abs(data["effective_charge_pH_7"]))
eff_charge_7 = f(charge_pH_7) 
f = scipy.interpolate.interp1d(np.abs(data_0003["bare_charge_pH_7"]), np.abs(data_0003["effective_charge_pH_7"]))
eff_charge_0003_7 = f(charge_pH_7) 
f = scipy.interpolate.interp1d(np.abs(data_0005["bare_charge_pH_7"]), np.abs(data_0005["effective_charge_pH_7"]))
eff_charge_0005_7 = f(charge_pH_7) 
ax2.vlines(charge_pH_7, 0, eff_charge_7, color="black", linewidth=1.4, zorder=-1)
ax2.fill_between([bare_charge_0003_7, bare_charge_0005_7], 0, eff_charge_7, alpha=0.3, color="black", zorder=-1)
ax2.hlines(eff_charge_7, 0, charge_pH_7, color='black', linewidth=1.4, zorder=-1)
ax2.fill_between([0, charge_pH_7], [eff_charge_0003_7, eff_charge_0003_7], [eff_charge_0005_7, eff_charge_0005_7], color='black', alpha=0.3, zorder=-1)

f = scipy.interpolate.interp1d(np.abs(data["bare_charge_pH_55"]), np.abs(data["effective_charge_pH_55"]))
eff_charge_55 = f(charge_pH_55) 
f = scipy.interpolate.interp1d(np.abs(data_0003["bare_charge_pH_55"]), np.abs(data_0003["effective_charge_pH_55"]))
eff_charge_0003_55 = f(charge_pH_55) 
f = scipy.interpolate.interp1d(np.abs(data_0005["bare_charge_pH_55"]), np.abs(data_0005["effective_charge_pH_55"]))
eff_charge_0005_55 = f(charge_pH_55) 
ax2.vlines(charge_pH_55, 0, eff_charge_55, color='#0101FF', linewidth=1.4, zorder=-1)
ax2.fill_between([bare_charge_0003_55, bare_charge_0005_55], 0, eff_charge_55, alpha=0.4, color='#0101FF', zorder=-1)
ax2.hlines(eff_charge_55, 0, charge_pH_55, color='#0101FF', linewidth=1.4, zorder=-1)
ax2.fill_between([0, charge_pH_55], [eff_charge_0003_55, eff_charge_0003_55], [eff_charge_0005_55, eff_charge_0005_55], color='#0101FF', alpha=0.4, zorder=-1)


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
plt.close()
exit()



##### Plot with only pH=7
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4*0.9))

## Plot of the charges

# Bare charge vs pK
ax1.vlines(pKa_7, 0, charge_pH_7, color="gray", linestyle="dotted", linewidth=1.4)
ax1.plot(data["pKa"], np.abs(data["bare_charge_pH_7"]), label=r"$\text{pH} = 7.0$, $N=134000$", color='black')

# Plot uncertainty
pK_range = np.linspace(3.0, 8.0, 1000)
charge_0003 = scipy.interpolate.interp1d(data_0003["pKa"], np.abs(data_0003["bare_charge_pH_7"]))
charge_0005 = scipy.interpolate.interp1d(data_0005["pKa"], np.abs(data_0005["bare_charge_pH_7"]))
ax1.fill_between(pK_range, charge_0003(pK_range), charge_0005(pK_range), facecolor='black', alpha=0.3, zorder=-1)

ax1.hlines(charge_pH_7, -20, 20, color='black', linestyle="--", linewidth=1.4)

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
ax2.plot(data["bare_charge_pH_7"].abs(), data["effective_charge_pH_7"].abs(), label=r"$\text{pH} = 7.0$, $N=134000$", color='black')

# Plot uncertainty
charge_range = np.linspace(50, 12000, 1000)
ren_charge_0003 = scipy.interpolate.interp1d(data_0003["bare_charge_pH_7"].abs(), data_0003["effective_charge_pH_7"].abs())
ren_charge_0005 = scipy.interpolate.interp1d(data_0005["bare_charge_pH_7"].abs(), data_0005["effective_charge_pH_7"].abs())
ax2.fill_between(charge_range, ren_charge_0003(charge_range), ren_charge_0005(charge_range), color='black', alpha=0.3, zorder=-1)

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
plt.close()


