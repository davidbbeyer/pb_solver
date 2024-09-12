import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.integrate
import pint
import tqdm


class PB_Solver:
    """
    A class to solve the Poisson-Boltzmann equation for a spherical colloid in a spherical cell.
    
    Attributes:
        bjerrum_length: The Bjerrum length of the solvent.
        r_colloid: The radius of the colloid.
        phi_colloid: The volume fraction of the colloid.
        r_cell: The radius of the cell.
        z_colloid: The charge of the colloid.
        sign_colloid: The sign of the colloid charge.
        c_salt_res: The salt concentration in the reservoir.
        prefactor: The prefactor of the ion densities.
    
    Methods:
        calculate_derivatives: Calculates the first and second derivatives appearing in the Poisson-Boltzmann equation.
        calculate_jacobian: Calculates the Jacobian of the RHS Poisson-Boltzmann equation.
        calculate_effective_charge_trizac: Calculates the effective charge according to the analytical Trizac prescription.
        debye_hueckel_electric_field: Calculates the Debye-Hueckel electric field at a distance r from the center of the colloid.
        debye_hueckel_potential: Calculates the Debye-Hueckel potential at a distance r from the center of the colloid.
        solve_dh: Solves the Debye-Hueckel equation for the parameters specified in the constructor.
        solve_pb: Solves the Poisson-Boltzmann equation for the parameters specified in the constructor.
    """

    def __init__(self, bjerrum_length, r_colloid, phi_colloid, c_salt_res, z_colloid):
        """
        Constructs all the necessary attributes for the PB_Solver object.
        
        Args:
            bjerrum_length: The Bjerrum length of the solvent in nanometers.
            r_colloid: The radius of the colloid in nanometers.
            phi_colloid: The volume fraction of the colloid.
            c_salt_res: The salt concentration in the reservoir in mol/L.
            z_colloid: The charge of the colloid in elementary charges.
        """
        ureg = pint.UnitRegistry()
        self.N_A = 6.02214076e23 / ureg.mol

        self.bjerrum_length = bjerrum_length * ureg.nanometer
        self.r_colloid = r_colloid * ureg.nanometer
        self.phi_colloid = phi_colloid
        self.r_cell = self.r_colloid * np.power(self.phi_colloid, -1/3)
        self.z_colloid = z_colloid * ureg.e
        self.sign_colloid = np.sign(z_colloid)
        self.c_salt_res = c_salt_res * ureg.mol / ureg.L 
        self.prefactor = (4 * np.pi * self.c_salt_res * self.N_A * self.bjerrum_length**3).to("dimensionless").magnitude

    def calculate_derivatives(self, r, phi, prefactor):
        """
        Calculates the first and second derivatives appearing in the Poisson-Boltzmann equation.

        Args:
            r: The radial coordinate.
            phi: The electrostatic potential.
            prefactor: The prefactor of the ion densities.
        Returns:
            The first and second derivatives of the electrostatic potential according to the Poisson-Boltzmann equation.
        """
        return np.vstack([phi[1], -2 * phi[1]/r + 2 * prefactor * np.sinh(phi[0])])
    
    def calculate_jacobian(self, r, phi, prefactor):
        """
        Calculates the Jacobian of the RHS Poisson-Boltzmann equation.

        Args:
            r: The radial coordinate.
            phi: The electrostatic potential.
            prefactor: The prefactor of the ion densities.
        Returns:
            The Jacobian of the RHS Poisson-Boltzmann equation.
        """
        return np.array([[0, 1], [np.ndarray.item(2 * prefactor * np.array([1, 0]).dot(np.cosh(phi))), -2/r]])

    def calculate_effective_charge_trizac(self, initial_guess=None, calculate_DH=False):
        """
        Calculates the effective charge according to the analytical Trizac prescription (see https://doi.org/10.1021/la027056m).

        Args:
            initial_guess: The initial guess for the solution. If None, the solution will be initialized with the Debye-Hueckel solution.
        Returns:
            The effective charge according to the Trizac prescription.
        """
        r, numerical_solution = self.solve_pb(initial_guess=initial_guess)

        phi_boundary = numerical_solution.sol(r)[0][-1]
        gamma = np.tanh(phi_boundary)
        kappa_pb = np.sqrt(2 * self.prefactor * np.cosh(phi_boundary))
        reduced_cell_radius = (self.r_cell/self.bjerrum_length).magnitude
        reduced_colloid_radius = (self.r_colloid/self.bjerrum_length).magnitude
        f_plus = (kappa_pb*reduced_cell_radius + 1) * np.exp(-kappa_pb * reduced_cell_radius) / (2 * kappa_pb)
        f_minus = (kappa_pb*reduced_cell_radius - 1) * np.exp(kappa_pb * reduced_cell_radius) / (2 * kappa_pb)

        z_eff = (gamma / kappa_pb) * ((kappa_pb**2 * reduced_colloid_radius * reduced_cell_radius - 1) * np.sinh(kappa_pb * (reduced_cell_radius - reduced_colloid_radius)) + kappa_pb * (reduced_cell_radius - reduced_colloid_radius) * np.cosh(kappa_pb * (reduced_cell_radius - reduced_colloid_radius)))
        
        if calculate_DH:
            alexander_solution_pot = self.debye_hueckel_potential(r, gamma * f_minus, gamma * f_plus, kappa_pb)
            alexander_solution_e_field = self.debye_hueckel_electric_field(r, gamma * f_minus, gamma * f_plus, kappa_pb)
            return z_eff, r, alexander_solution_pot, alexander_solution_e_field
        else:
            return z_eff
    
    def debye_hueckel_electric_field(self, r, a, b, kappa):
        """
        Calculates the Debye-Hueckel electric field at a distance r from the center of the colloid.

        Args:
            r: The radial coordinate.
            a: The first coefficient.
            b: The second coefficient.
            kappa: The inverse Debye length.
        Returns:
            The Debye-Hueckel electric field at a distance r from the center of the colloid.
        """
        return a * kappa * np.exp(-kappa * r)/r - b * kappa * np.exp(kappa * r)/r + (a * np.exp(-kappa * r)/r**2 + b * np.exp(kappa * r)/r**2)

    def debye_hueckel_potential(self, r, a, b, kappa):
        """
        Calculates the Debye-Hueckel potential at a distance r from the center of the colloid.

        Args:
            r: The radial coordinate.
            a: The first coefficient.
            b: The second coefficient.
            kappa: The inverse Debye length.
        Returns:
            The Debye-Hueckel potential at a distance r from the center of the colloid.
        """
        return a * np.exp(-kappa * r)/r + b * np.exp(kappa * r)/r
    
    def solve_dh(self, r):
        """
        Solves the Debye-Hueckel equation for the parameters specified in the constructor.

        Args:
            r: The radial coordinate.
        Returns:
            The Debye-Hueckel potential at a distance r from the center of the colloid.
        """
        # Calculate the analytical coefficients for the given boundary conditions (see e.g. https://doi.org/10.48550/arXiv.1004.4310)
        kappa = np.sqrt(2 * self.prefactor)
        reduced_cell_radius = (self.r_cell/self.bjerrum_length).magnitude
        reduced_colloid_radius = (self.r_colloid/self.bjerrum_length).magnitude
        reduced_colloid_charge = self.z_colloid.magnitude
        b = -reduced_colloid_charge / ((kappa * reduced_colloid_radius - 1) * np.exp(kappa * reduced_colloid_radius) - (kappa * reduced_colloid_radius + 1) * ((kappa * reduced_cell_radius - 1)/(kappa * reduced_cell_radius + 1)) * np.exp(kappa * (2 * reduced_cell_radius - reduced_colloid_radius)))
        a = b * (kappa * reduced_cell_radius - 1) / (kappa * reduced_cell_radius + 1) * np.exp(2 * kappa * reduced_cell_radius)
        
        return self.debye_hueckel_potential(r, a, b, kappa), self.debye_hueckel_electric_field(r, a, b, kappa)

    def solve_pb(self, initial_guess=None):
        """
        Solves the Poisson-Boltzmann equation for the parameters specified in the constructor.

        Args:
            initial_guess: The initial guess for the solution. If None, the solution will be initialized with the zeros.
        Returns:
            The radial coordinate and the numerical solution to the Poisson-Boltzmann equation.
        """
        derivatives = lambda r, phi: self.calculate_derivatives(r, phi, self.prefactor)

        r = np.linspace((self.r_colloid/self.bjerrum_length).magnitude, (self.r_cell/self.bjerrum_length).magnitude, 10000)
        
        def bc(ya, yb):
            """
            Neumann boundary conditions for the Poisson-Boltzmann equation.
            """
            e_field_at_colloid = self.z_colloid.magnitude * ((self.bjerrum_length/self.r_colloid).magnitude)**2
            if isinstance(e_field_at_colloid, np.ndarray):
                e_field_at_colloid = e_field_at_colloid[0] #For some reason, this is needed to work with the root finding for charge regulation... Need to find out why!
            return np.array([ya[1] + e_field_at_colloid, yb[1]])
        
        # Initial guess for the solution
        if initial_guess is None:
            initial_guess = np.zeros((2, r.size))

        jac = lambda r, phi: self.calculate_jacobian(r, phi, self.prefactor)
        numerical_solution = scipy.integrate.solve_bvp(derivatives, bc, r, initial_guess, tol=1e-11, max_nodes=int(1e12))
        if numerical_solution.status != 0:
            raise RuntimeError(f"solve_bvp did not converge: {numerical_solution.message}")
        return r, numerical_solution
    

class PB_Solver_Constant_Salt:
    """
    A class to solve the Poisson-Boltzmann equation for a spherical colloid in a spherical cell with a constant salt concentration inside the cell.
    
    Attributes:
        bjerrum_length: The Bjerrum length of the solvent.
        r_colloid: The radius of the colloid.
        phi_colloid: The volume fraction of the colloid.
        n_salt: The number of salt ion pairs in the cell.
        z_colloid: The charge of the colloid.
        sign_colloid: The sign of the colloid charge.
        reference_concentration: The reference concentration of 1 mol/L.
        
    Methods:
        calculate_effective_charge_trizac: Calculates the effective charge according to the analytical Trizac prescription.
        solve_pb: Solves the Poisson-Boltzmann equation for the parameters specified in the constructor.
    """
    def __init__(self, bjerrum_length, r_colloid, phi_colloid, n_salt, z_colloid):
        """
        Constructs all the necessary attributes for the PB_Solver_Constant_Salt object.
        
        Args:
            bjerrum_length: The Bjerrum length of the solvent in nanometers.
            r_colloid: The radius of the colloid in nanometers.
            phi_colloid: The volume fraction of the colloid.
            n_salt: The number of salt ion pairs in the cell.
            z_colloid: The charge of the colloid in elementary charges.
        """
        ureg = pint.UnitRegistry()
        self.N_A = 6.02214076e23 / ureg.mol

        self.bjerrum_length = bjerrum_length * ureg.nanometer
        self.r_colloid = r_colloid * ureg.nanometer
        self.phi_colloid = phi_colloid
        self.r_cell = self.r_colloid * np.power(self.phi_colloid, -1/3)
        self.z_colloid = z_colloid * ureg.e
        self.sign_colloid = np.sign(z_colloid)
        self.n_salt = n_salt
        self.reference_concentration = 1 * ureg.mol / ureg.L

    def calculate_effective_charge_trizac(self, initial_guess=None):
        """
        Calculates the effective charge according to the analytical Trizac prescription.

        Args:
            initial_guess: The initial guess for the solution. If None, the solution will be initialized with the zeros.
        Returns:
            The effective charge according to the Trizac prescription.
        """
        r, numerical_solution, c_salt_res = self.solve_pb(initial_guess=initial_guess)
        solver = PB_Solver(bjerrum_length=self.bjerrum_length.magnitude, r_colloid=self.r_colloid.magnitude, phi_colloid=self.phi_colloid, c_salt_res=c_salt_res.magnitude, z_colloid=self.z_colloid.magnitude)
        z_eff = solver.calculate_effective_charge_trizac(initial_guess=initial_guess)
        return z_eff

    def solve_pb(self, initial_guess=None):
        """
        Solves the Poisson-Boltzmann equation for the parameters specified in the constructor.

        Args:
            initial_guess: The initial guess for the solution. If None, the solution will be initialized with the Debye-Hueckel solution.
        Returns:
            The radial coordinate and the numerical solution to the Poisson-Boltzmann equation.
        """
        def calculate_deviation_salt_number(c_salt_res):
            solver = PB_Solver(bjerrum_length=self.bjerrum_length.magnitude, r_colloid=self.r_colloid.magnitude, phi_colloid=self.phi_colloid, c_salt_res=c_salt_res, z_colloid=self.z_colloid.magnitude)
            r, numerical_solution = solver.solve_pb(initial_guess=initial_guess)

            prefactor = (4 * np.pi * c_salt_res * self.reference_concentration * self.N_A * self.bjerrum_length**3).to("dimensionless").magnitude
            density_co = prefactor * r**2 * np.exp(-self.sign_colloid*numerical_solution.sol(r)[0])
            n_salt_calculated = np.trapz(density_co, r)

            return n_salt_calculated - self.n_salt
        
        initial_guess_c_salt_res = (self.n_salt / (4/3 * np.pi * (self.r_cell**3 - self.r_colloid**3) * self.N_A)).to("mol/L")
        c_salt_res = scipy.optimize.root(calculate_deviation_salt_number, x0=initial_guess_c_salt_res.magnitude, options={"maxfev": int(1e9)}).x * self.reference_concentration

        solver = PB_Solver(bjerrum_length=self.bjerrum_length.magnitude, r_colloid=self.r_colloid.magnitude, phi_colloid=self.phi_colloid, c_salt_res=c_salt_res.magnitude, z_colloid=self.z_colloid.magnitude)
        r, numerical_solution = solver.solve_pb(initial_guess=initial_guess)
        return r, numerical_solution, c_salt_res
    

class PB_Solver_Charge_Regulation:
    """
    A class to solve the Poisson-Boltzmann equation for a spherical colloid in a spherical cell with charge regulation.
    
    Attributes:
        bjerrum_length: The Bjerrum length of the solvent.
        r_colloid: The radius of the colloid.
        phi_colloid: The volume fraction of the colloid.
        r_cell: The radius of the cell.
        z_colloid_quenched: The quenched charge of the colloid.
        z_colloid_annealed: The annealed charge of the colloid.
        reference_concentration: The reference concentration of 1 mol/L.
        reference_charge: The reference charge of 1 elementary charge.
        pKa: The pKa of the weak acid.
        
    Methods:
        alpha_henderson_hasselbalch: Calculates the degree of ionization according to the Henderson-Hasselbalch equation.
        calculate_bare_and_effective_charge: Calculates the bare and effective charge of the colloid for a range of pH values.
        solve_pb: Solves the Poisson-Boltzmann equation for the parameters specified in the constructor.
    """
    def __init__(self, bjerrum_length, r_colloid, phi_colloid, z_colloid_quenched, z_colloid_annealed, pKa):
        """
        Constructs all the necessary attributes for the PB_Solver_Charge_Regulation object.
        
        Args:
            bjerrum_length: The Bjerrum length of the solvent in nanometers.
            r_colloid: The radius of the colloid in nanometers.
            phi_colloid: The volume fraction of the colloid.
            z_colloid_quenched: The quenched charge of the colloid in elementary charges.
            z_colloid_annealed: The annealed charge of the colloid in elementary charges.
            pKa: The pKa of the weak acid.
        """
        ureg = pint.UnitRegistry()
        self.N_A = 6.02214076e23 / ureg.mol

        self.bjerrum_length = bjerrum_length * ureg.nanometer
        self.r_colloid = r_colloid * ureg.nanometer
        self.phi_colloid = phi_colloid
        self.r_cell = self.r_colloid * np.power(self.phi_colloid, -1/3)
        self.z_colloid_quenched = z_colloid_quenched * ureg.e
        self.z_colloid_annealed = z_colloid_annealed * ureg.e
        self.reference_concentration = 1 * ureg.mol / ureg.L
        self.reference_charge = 1 * ureg.e
        self.pKa = pKa

    def alpha_henderson_hasselbalch(self, pH, pKa):
        """
        Calculates the degree of ionization according to the Henderson-Hasselbalch equation.

        Args:
            pH: pH of the solution
            pKa: pKa of the weak acid
        Returns:
            Degree of ionization
        """
        return 1/(1 + 10**(pKa - pH))
    
    def calculate_bare_and_effective_charge(self, pH_range):
        """
        Calculates the bare and effective charge of the colloid for a range of pH values.
        
        Args:
            pH_range: The range of pH values in the reservoir.
        Returns:
            The pH values at the boundary of the cell, the bare charges and the effective charges of the colloid.
        """
        pH_values_boundary = []
        bare_charges = []
        effective_charges = []
        r, numerical_solution, z_colloid, z_colloid_eff = self.solve_pb(pH_res=pH_range[0])

        for pH_res in tqdm.tqdm(pH_range):
            r, numerical_solution, z_colloid, z_colloid_eff = self.solve_pb(pH_res=pH_res, initial_guess=numerical_solution.sol(r))

            bare_charges.append(z_colloid.magnitude[0])
            effective_charges.append(z_colloid_eff.magnitude)

            reduced_cell_radius = (self.r_cell/self.bjerrum_length).magnitude
            phi_boundary = numerical_solution.sol(reduced_cell_radius)[0]
            pH_values_boundary.append(pH_res + phi_boundary * np.log10(np.exp(1)))
        return pH_values_boundary, bare_charges, effective_charges

    def solve_pb(self, pH_res, initial_guess=None, initial_charge=None, calculate_DH=False):
        """
        Solves the Poisson-Boltzmann equation for the parameters specified in the constructor.

        Args:
            initial_guess: The initial guess for the solution. If None, the solution will be initialized with zeros.
        Returns:
            The radial coordinate, the numerical solution to the Poisson-Boltzmann equation and the bare and effective charge of the colloid.
        """
        # Calculate the ionic strength of the reservoir
        c_salt_res = (max(10**(-pH_res), 10**(-(14-pH_res)))) * self.reference_concentration

        def calculate_deviation_charge(z_colloid):
            """
            Calculates the deviation of the self-consistently calculated charge of the colloid from the given value.

            Args:
                z_colloid: The charge of the colloid.
            Returns:

            """
            solver = PB_Solver(bjerrum_length=self.bjerrum_length.magnitude, r_colloid=self.r_colloid.magnitude, phi_colloid=self.phi_colloid, c_salt_res=c_salt_res.magnitude, z_colloid=z_colloid)
            r, numerical_solution = solver.solve_pb(initial_guess=initial_guess)

            reduced_colloid_radius = (self.r_colloid/self.bjerrum_length).magnitude
            phi_colloid = numerical_solution.sol(reduced_colloid_radius)[0]
            pH_colloid = pH_res + phi_colloid * np.log10(np.exp(1))
            z_colloid_calculated = self.z_colloid_quenched + self.z_colloid_annealed * self.alpha_henderson_hasselbalch(pH_colloid, self.pKa)
            return z_colloid_calculated.magnitude - z_colloid
        
        if initial_charge is None:
            initial_guess_z_colloid = self.z_colloid_quenched + self.z_colloid_annealed * self.alpha_henderson_hasselbalch(pH_res, self.pKa)
        else:
            initial_guess_z_colloid = initial_charge
        z_colloid = scipy.optimize.root(calculate_deviation_charge, x0=initial_guess_z_colloid.magnitude, options={"maxfev": int(1e9)}).x * self.reference_charge

        solver = PB_Solver(bjerrum_length=self.bjerrum_length.magnitude, r_colloid=self.r_colloid.magnitude, phi_colloid=self.phi_colloid, c_salt_res=c_salt_res.magnitude, z_colloid=z_colloid.magnitude)
        r, numerical_solution = solver.solve_pb(initial_guess=initial_guess)

        if calculate_DH:
            z_colloid_eff, r, alexander_solution_pot, alexander_solution_e_field = solver.calculate_effective_charge_trizac(initial_guess=initial_guess, calculate_DH=calculate_DH)
            return r, numerical_solution, z_colloid, z_colloid_eff * self.reference_charge, alexander_solution_pot, alexander_solution_e_field 
        else:
            z_colloid_eff = solver.calculate_effective_charge_trizac(initial_guess=initial_guess, calculate_DH=calculate_DH) * self.reference_charge
            return r, numerical_solution, z_colloid, z_colloid_eff
