
import numpy as np
from numba import types
from scipy import interpolate
from scipy import optimize
import copy
import h5py
import os
import yaml
from tempfile import NamedTemporaryFile
from astropy import constants

from clima import AdiabatClimate, ClimaException
import clima

from photochem import EvoAtmosphere, PhotoException

class OzoneData():
    xs_O2 : types.double[:] # type: ignore
    xs_O3 : types.double[:] # type: ignore
    def __init__(self):
        pass

class OzoneCalculator(AdiabatClimate):

    def __init__(self, species_dict, settings_dict, flux_str, data_dir=None):

        # Ensure O2 and O3 are species
        species = [tmp['name'] for tmp in species_dict['species']]
        if 'O2' not in species or 'O3' not in species:
            raise ClimaException('To use the Ozone feature O2 and O3 must be species')
        
        # Make new binning for photolysis scheme
        filename = os.path.dirname(clima.__file__)+'/data/kdistributions/bins.h5'
        with h5py.File(filename ,'r') as f:
            ir_wavl = f['ir_wavl'][:]
            sol_wavl = f['sol_wavl'][:]
        ir_wavl = ir_wavl[:2]
        sol_wavl = sol_wavl[np.where(sol_wavl < 1.04)]
        
        # Ensure Photolysis and Reyleigh opacities are on
        settings_dict = copy.deepcopy(settings_dict)
        settings_dict['optical-properties']['solar']['opacities']['photolysis-xs'] = 'on'
        settings_dict['optical-properties']['solar']['opacities']['rayleigh'] = 'on'

        # Scale flux by 2. I don't want a diurnal averaging factor for O3 calculations
        settings_dict['planet']['photon-scale-factor'] = settings_dict['planet']['photon-scale-factor']*2

        with NamedTemporaryFile('wb',suffix='.h5') as f_bins:
            # Make custom bins
            with h5py.File(f_bins.name,'w') as f:
                dset = f.create_dataset("sol_wavl", sol_wavl.shape, 'f')
                dset[:] = sol_wavl
                dset = f.create_dataset("ir_wavl", ir_wavl.shape, 'f')
                dset[:] = ir_wavl
            settings_dict['optical-properties']['wavelength-bins-file'] = f_bins.name

            with NamedTemporaryFile('w') as f_species:
                # Write species file
                yaml.safe_dump(species_dict, f_species)
                with NamedTemporaryFile('w') as f_settings:
                    # Write settings file
                    yaml.safe_dump(settings_dict, f_settings)
                    with NamedTemporaryFile('w') as f_flux:
                        # Write stellar flux file
                        f_flux.write(flux_str)
                        f_flux.flush()
                        # Initialize AdiabatClimate
                        super().__init__(
                            f_species.name, 
                            f_settings.name, 
                            f_flux.name,
                            data_dir=data_dir,
                            double_radiative_grid=False
                        )

        # Change default parameters
        self.P_top = 10.0

        # Get photolysis cross sections
        xs_O2, xs_O3 = self._get_xsections()
        self.odat = OzoneData()
        self.odat.xs_O2 = xs_O2
        self.odat.xs_O3 = xs_O3

    def _get_xsections(self):
        
        clima_dir = os.path.dirname(clima.__file__)
        wv1, qy1 = np.loadtxt(clima_dir+'/data/xsections/O2/O2_+_hv_=_O_+_O1D.txt',skiprows=2).T
        wv11, qy11 = np.loadtxt(clima_dir+'/data/xsections/O2/O2_+_hv_=_O_+_O.txt',skiprows=2).T
        wv2, xs = np.loadtxt(clima_dir+'/data/xsections/O2/O2_xs.txt',skiprows=2).T
        assert np.all(wv1 == wv2)
        assert np.all(wv1 == wv11)
        wv_O2 = wv1
        xs_O2 = xs*(qy1+qy11)
        
        wv1, qy1 = np.loadtxt(clima_dir+'/data/xsections/O3/O3_+_hv_=_O1D_+_O2.txt',skiprows=2).T
        wv11, qy11 = np.loadtxt(clima_dir+'/data/xsections/O3/O3_+_hv_=_O_+_O2.txt',skiprows=2).T
        wv2, xs = np.loadtxt(clima_dir+'/data/xsections/O3/O3_xs.txt',skiprows=2).T
        assert np.all(wv1 == wv2)
        assert np.all(wv1 == wv11)
        wv_O3 = wv1
        xs_O3 = xs*(qy1+qy11)
        
        wv_av = (self.rad.sol.wavl[1:] + self.rad.sol.wavl[:-1])/2
        xs_O2 = interpolate.interp1d(wv_O2, xs_O2, bounds_error=False, fill_value=0.0)(wv_av)
        xs_O3 = interpolate.interp1d(wv_O3, xs_O3, bounds_error=False, fill_value=0.0)(wv_av)

        return xs_O2, xs_O3
    
    def objective_all(self, log10fO3, T, P, f_i, den):
        
        # Unpack the ozone profile
        fO3 = 10.0**log10fO3
        fO3 = np.append(fO3[0],fO3)

        # Set the ozone profile
        ind = self.species_names.index('O3')
        f_i_copy = f_i.copy()
        f_i_copy[:,ind] = fO3

        # Compute radiative transfer
        self.TOA_fluxes_dry(P, T, f_i_copy)

        # Get mean indensity
        amean = np.clip(self.rad.wrk_sol.amean,a_min=1e-100,a_max=np.inf)
        amean_grid = np.sqrt(amean[:-1,:]*amean[1:,:])
        
        # Compute photolysis rates of O2 and O3
        prates_O2 = np.sum(self.odat.xs_O2*amean_grid,axis=1)
        k1 = np.clip(prates_O2,a_min=1e-100,a_max=np.inf)
        prates_O3 = np.sum(self.odat.xs_O3*amean_grid,axis=1)
        k3 = np.clip(prates_O3,a_min=1e-100,a_max=np.inf)

        # O + O2 + M => O3 + M
        k0 = 2.989029e-28*T[1:]**-2.3
        kinf = 2.8e-11
        k2 = (k0*den)/(1 + (k0*den)/kinf)

        # O3 + O => O2 + O2
        k4 = 8.0e-12*np.exp(-2060.0/T[1:])
        
        # Compute the O2 density
        ind = self.species_names.index('O2')
        nO2 = den*f_i[1:,ind]

        # Compute the O3 density
        nO3 = np.sqrt(2)*nO2*np.sqrt((k1*k2*den)/(k3*k4*den))
        fO3 = nO3/den # mixing ratio

        # Convert to log10O3, and remove nans
        res = np.log10(fO3)
        res[np.isnan(res)] = -100
        
        return res

    def objective(self, x, T, P, f_i, den, exp_val):
        return exp_val**self.objective_all(x, T, P, f_i, den) - exp_val**x
    
    def compute_ozone_profile(self, c, scale_factor):
        
        # Collect
        T = np.append(c.T_surf,c.T)
        P = np.append(c.P_surf,c.P)
        f_i = np.concatenate((np.array([c.f_i[0,:]]),c.f_i),axis=0)
        den = np.sum(c.densities,axis=1)

        for init_val in [-10, -12, -8]:
            for exp_val in [1.1, 1.2, 1.3]:
                # Initial conditions
                xinit = np.ones(len(self.z))*init_val

                # Compute the ozone profile
                sol = optimize.root(self.objective, xinit, args=(T, P, f_i, den, exp_val), method='hybr',
                                    options={'eps': 1e-3, 'xtol': 1e-5, 'band':(0,0)})
                if sol.success: break
            if sol.success: break

        if not sol.success:
            return sol.success

        # Set ozone and then compute radiative transfer
        ind = c.species_names.index('O3')
        f_i_copy = f_i.copy()
        fO3 = 10.0**sol.x*scale_factor
        f_i_copy[:,ind] = np.append(fO3[0],fO3)
        c.TOA_fluxes_dry(P, T, f_i_copy)

        return sol.success

class EvoAtmosphereHotRock(EvoAtmosphere):

    def __init__(self, mechanism_file, M_planet, R_planet, photon_scale_factor, flux_str, data_dir=None):

        # Settings
        settings_dict = yaml.safe_load(DEFAULT_SETTINGS)
        settings_dict['planet']['planet-mass'] = float(M_planet*constants.M_earth.to('g').value) # grams
        settings_dict['planet']['planet-radius'] = float(R_planet*constants.R_earth.to('cm').value) # cm
        settings_dict['planet']['photon-scale-factor'] = int(photon_scale_factor)*2 # dayside only


        with NamedTemporaryFile('w') as f_settings:
            # Write settings file
            yaml.safe_dump(settings_dict, f_settings)
            with NamedTemporaryFile('w') as f_flux:
                # Write stellar flux file
                f_flux.write(flux_str)
                f_flux.flush()
                with NamedTemporaryFile('w') as f_atmosphere:
                    # Write Atmosphere file
                    f_atmosphere.write(ATMOSPHERE_INIT)
                    f_atmosphere.flush()

                    super().__init__(
                        mechanism_file,
                        f_settings.name,
                        f_flux.name,
                        f_atmosphere.name,
                        data_dir=data_dir
                    )

        # Parameters for determining steady state
        self.TOA_pressure_avg = 1.0e-7*1e6 # mean TOA pressure (dynes/cm^2)
        self.max_dT_tol = 5 # The permitted difference between T in photochem and desired T
        self.reset_dT_tol = 10
        self.max_dlog10edd_tol = 0.2 # The permitted difference between Kzz in photochem and desired Kzz
        self.atol_min = 1e-17 # min atol that will be tried
        self.atol_max = 1e-18 # max atol that will be tried
        self.atol_avg = 1e-19 # avg atol that is tried
        self.freq_update_PTKzz = 1000 # step frequency to update PTKzz profile.
        self.freq_reinit = 10_000 # step frequency to reinitialize integration and change atol
        self.max_total_step = 100_000 # Maximum total allowed steps before giving up
        self.min_step_conv = 300 # Min internal steps considered before convergence is allowed
        
        # Values in photochem to adjust
        self.var.autodiff = True
        self.var.atol = self.atol_avg
        self.var.conv_min_mix = 1e-10 # Min mix to consider during convergence check
        self.var.conv_longdy = 0
        self.var.verbose = 1

        # Below for interpolation
        self.P_desired = None
        self.T_desired = None
        self.log10P_interp = None
        self.T_interp = None
        self.Kzz = None

        # For integration
        self.total_step_counter = None
        self.atol_counter = None
        self.nerrors = None
        self.robust_stepper_initialized = False

    def initialize_photochem_from_clima(self, c, Kzz):

        # Set boundary conditions
        for i,sp in enumerate(self.dat.species_names[self.dat.np:-2-self.dat.nsl]):
            self.set_lower_bc(sp, bc_type='vdep', vdep=0.0)
        for i,sp in enumerate(c.species_names):
            if sp not in ['O3']:
                Pi = c.P_surf*c.f_i[0,i]
                self.set_lower_bc(sp, bc_type='press', press=Pi)

        # Set the grid to clima
        c.to_regular_grid()
        self.update_vertical_grid(TOA_alt=(c.dz[0]/2+c.z[-1]))

        # Interpolate temperature to grid
        T = np.interp(self.var.z, c.z, c.T)
        self.set_temperature(T)

        # Interpolate densities to grid
        usol = np.ones(self.wrk.usol.shape)*1e-40
        for i,sp in enumerate(c.species_names):
            ind = self.dat.species_names.index(sp)
            density = np.interp(self.var.z, c.z, np.log10(c.densities[:,i]))
            usol[ind,:] = 10.0**density

        # Set Kzz
        self.var.edd = np.ones(self.var.edd.shape[0])*Kzz

        # Update all variables
        self.prep_atmosphere(usol)

        # Move the TOA to where we want it
        self.update_vertical_grid(TOA_pressure=self.TOA_pressure_avg)

        # Save the Pressure-temperature profile
        self.P_desired = c.P
        self.T_desired = c.T
        self.Kzz = Kzz
        self.log10P_interp = np.log10(self.P_desired.copy())
        self.T_interp = self.T_desired.copy()

        # Extrapolate linearly to 10,000 bars
        slope = (self.T_interp[1] - self.T_interp[0])/(self.log10P_interp[1] - self.log10P_interp[0])
        intercept = self.T_interp[0] - slope*self.log10P_interp[0]
        tmp = slope*np.log10(1e4*1e6) + intercept
        self.log10P_interp = np.append(np.log10(1e4*1e6), self.log10P_interp).copy()[::-1]
        self.T_interp = np.append(tmp, self.T_interp).copy()[::-1]
        self.P_desired = 10.0**self.log10P_interp[::-1].copy()
        self.T_desired = self.T_interp[::-1].copy()

    def initialize_robust_stepper(self, usol):
        """Initialized a robust integrator.

        Parameters
        ----------
        usol : ndarray[double,dim=2]
            Input number densities
        """
        
        self.total_step_counter = 0
        self.atol_counter = 0
        self.nerrors = 0
        self.initialize_stepper(usol)
        self.robust_stepper_initialized = True

    def robust_step(self):
        """Takes a single robust integrator step

        Returns
        -------
        tuple
            The tuple contains two bools `give_up, reached_steady_state`. If give_up is True
            then the algorithm things it is time to give up on reaching a steady state. If
            reached_steady_state then the algorithm has reached a steady state within
            tolerance.
        """        

        if not self.robust_stepper_initialized:
            raise Exception('This routine can only be called after `initialize_robust_stepper`')

        pc = self

        give_up = False
        reached_steady_state = False

        for i in range(1):
            try:
                pc.step()
                self.atol_counter += 1
                self.total_step_counter += 1
            except PhotoException as e:
                # If there is an error, lets reinitialize, but get rid of any
                # negative numbers
                usol = np.clip(pc.wrk.usol.copy(),a_min=1.0e-40,a_max=np.inf)
                pc.initialize_stepper(usol)
                self.nerrors += 1

                if self.nerrors > 10:
                    give_up = True
                    break

            # convergence checking
            converged = pc.check_for_convergence()

            # Compute the max difference between the P-T profile in photochemical model
            # and the desired P-T profile
            T_p = np.interp(np.log10(pc.wrk.pressure_hydro.copy()[::-1]), self.log10P_interp, self.T_interp)
            T_p = T_p.copy()[::-1]
            max_dT = np.max(np.abs(T_p - pc.var.temperature))

            # TOA pressure
            TOA_pressure = pc.wrk.pressure_hydro[-1]

            condition1 = converged and pc.wrk.nsteps > self.min_step_conv or pc.wrk.tn > pc.var.equilibrium_time
            condition2 = max_dT < self.max_dT_tol and self.TOA_pressure_avg/3 < TOA_pressure < self.TOA_pressure_avg*3

            if condition1 and condition2:
                # success!
                reached_steady_state = True
                break

            if self.atol_counter > self.freq_reinit:
                # Convergence has not happened after 10000 steps, so we try a new atol
                pc.var.atol = 10.0**np.random.uniform(low=np.log10(self.atol_min),high=np.log10(self.atol_max))
                pc.initialize_stepper(pc.wrk.usol)
                self.atol_counter = 0
                break
        
            if not (pc.wrk.nsteps % self.freq_update_PTKzz) or (condition1 and not condition2) or max_dT > self.reset_dT_tol:
                # After ~1000 steps, lets update P,T, edd and vertical grid
                pc.set_press_temp_edd(self.P_desired,self.T_desired,np.ones(self.T_desired.shape[0])*self.Kzz,hydro_pressure=True)
                pc.update_vertical_grid(TOA_pressure=self.TOA_pressure_avg)
                pc.initialize_stepper(pc.wrk.usol)

            if self.total_step_counter > self.max_total_step:
                give_up = True
                break
                
        return give_up, reached_steady_state
    
    def find_steady_state(self):
        """Attempts to find a photochemical steady state.

        Returns
        -------
        bool
            If True, then the routine was successful.
        """    

        self.initialize_robust_stepper(self.wrk.usol)
        success = True
        while True:
            give_up, reached_steady_state = self.robust_step()
            if reached_steady_state:
                break
            if give_up:
                success = False
                break
        return success

DEFAULT_SETTINGS = \
"""
atmosphere-grid:
  bottom: 0.0
  top: atmospherefile
  number-of-layers: 100

photolysis-grid:
  regular-grid: true
  lower-wavelength: 92.5 # nm
  upper-wavelength: 855.0 # nm
  number-of-bins: 200

planet:
  planet-mass: null
  planet-radius: null
  surface-albedo: 0.0
  solar-zenith-angle: 60.0
  photon-scale-factor: null
  hydrogen-escape: {type: none}
  water: {fix-water-in-troposphere: off, gas-rainout: off, water-condensation: off}

boundary-conditions:
- name: H2O
  lower-boundary: {type: vdep, vdep: 0.0}
  upper-boundary: {type: veff, veff: 0.0}
"""

ATMOSPHERE_INIT = \
"""alt      den        temp       eddy                       
0.0      1          1000       1e6              
1.0e3    1          1000       1e6         
"""
