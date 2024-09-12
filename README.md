# A Poisson-Boltzmann Solver for Charge-Regulating Colloids

This repository contains a code that numerically solves the Poisson-Boltzmann equation for a spherical colloid in a cell, coupled to a reservoir at a given pH-value and monovalent salt concentration, with either a constant charge (Neumann) or a self-consistent charge-regulation boundary condition on the colloid surface.
It was used to the produce the theoretical results in this [preprint](https://doi.org/10.48550/arXiv.2409.03049).
The code also contains functionality to obtain the renormalized (effective) charge of the colloid using the [Trizac](https://doi.org/10.1021/la027056m) prescription.
Internally, the code uses open-source Python modules, including Numpy and Scipy for numerical operations, Pandas for data processing and Pint for unit conversions.

## Dependencies

- [Pint](https://pint.readthedocs.io/en/stable/)
- [Pandas](https://pandas.pydata.org/)
- [Numpy](https://numpy.org/)
- [SciPy](https://scipy.org/) 

## Contents

- `checks/`: folder with various scripts that check the numerical solver against analytical solutions and results from the literature
- `paper/`: folder with scripts and numerical results presented in the paper
- `LICENSE`: license of pb_solver
- `pb_solver.py`: source code of pb_solver 
- `requirements.txt`: list of required libraries to use pb_solver

## Usage

### Setting up a virtual environment

To use the Poisson-Boltzmann Solver, first clone this repository locally:

```sh
git clone git@github.com:davidbbeyer/pb_solver.git
```

To handle the dependencies of the solver, it is most convenient to use a Python virtual enviroment.
In order to set up a virtual environment, the Python module [`venv`](https://docs.python.org/3/library/venv.html) is needed.
If `venv` is not included in your Python distribution, your first need to install it, e.g. on Ubuntu:

```sh
sudo apt install python3-venv
```

To set up a virtual environment and install the Python dependencies, run the following commands:

```sh
python3 -m venv pb_solver
source pb_solver/bin/activate
python3 -m pip install -r requirements.txt
deactivate
```

### Running the Checks

The folder `checks/` contains various scripts to check the validity of the numerical results against analytical solutions and results from the literature.
To run these checks (e.g. `checks/check_debye_hueckel.py`), simply activate the virtual environment and run the script using Python:

```sh
source pb_solver/bin/activate
python3 checks/check_debye_hueckel.py
```

The included checks are:

- `checks/check_alexander.py`: Checks that the code is able to reproduce the effective charges reported by Alexander et al. (Fig. 4 of [JCP 80 (11), 5776-5781](https://doi.org/10.1063/1.446600)) in the presence of salt.
- `checks/check_charge_regulation.py`: Compares the degree of ionization of a charge-regulating colloid in the limit of a vanishing Bjerrum length to an independent numerical calculation that couples the ideal Donnan theory and the Henderson-Hasselbalch equation.
- `checks/check_charges_ideal`: Contains a script to check that the workflow used to produce the data in the preprint gives the expected result in the limit of a vanishing Bjerrum length. 
- `checks/check_debye_hueckel.py`: Compares the numerical solution of the nonlinear Poisson-Boltzmann equation to the linearized analytical solution (Debye-Hückel) in the case of a fixed low surface charge density.
- `checks/check_donnan.py`: Compares the value of the electrostatic potential at the cell boundary to the analytical ideal Donnan potential in the limit of a vanishing Bjerrum length. 


## References

Check out the corresponding [preprint](https://doi.org/10.48550/arXiv.2409.03049) and references therein to learn more about Poisson-Boltzmann theory, charge regulation, charge renormalization and applications of the solver.

```bibtex
@article{vogel2024co2,
  title={CO2-induced Drastic Decharging of Dielectric Surfaces in Aqueous Suspensions},
  author={Vogel, Peter and Beyer, David and Holm, Christian and Palberg, Thomas},
  journal={arXiv preprint},
  year={2024},
  doi={10.48550/arXiv.2409.03049},
}
```
