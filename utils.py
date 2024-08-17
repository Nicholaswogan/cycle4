import numpy as np
import numba as nb
from numba import types
import yaml
from tempfile import NamedTemporaryFile

from scipy import constants as const
from astropy import constants
import pysynphot as psyn

from clima import AdiabatClimate, ClimaException, rebin, rebin_with_errors

# We use a jitclass because it has strict type checking
@nb.experimental.jitclass()
class ThermalEmissionData():
    Teq : types.double # Zero albedo equilibrium temperature # type: ignore
    R_planet : types.double # Radius of planet in cm # type: ignore
    R_star : types.double # Radius of star in cm # type: ignore
    wavl_star : types.double[:] # Wavelength grid for stellar surface (um) # type: ignore
    flux_star : types.double[:] # Flux of stellar surface (ergs/cm^2/s/cm) # type: ignore
    flux_star_c : types.double[:] # Flux of stellar surface regrided to Clima IR bins (ergs/cm^2/s/cm) # type: ignore
    def __init__(self):
        pass

class AdiabatClimateThermalEmission(AdiabatClimate):
    "An extension of the AdiabatClimate class to interpret thermal emission observations."

    def __init__(self, Teq, M_planet, R_planet, Teff, metal, logg, R_star, 
                 species_file=None, settings_template=None, data_dir=None, nz=50, number_of_zeniths=1, 
                 catdir='phoenix', nw=5000, stellar_surface_scaling=1.0):
        """Initializes the code.

        Parameters
        ----------
        Teq : float
            Zero albedo equilibrium temperature of the planet (K).
        M_planet : float
            Mass of the planet in Earth masses.
        R_planet : float
            Radius of the planet in Earth radii.
        Teff : float
            Stellar effective temperature in K.
        metal : float
            log10 metallicity of the star.
        logg : float
            log10 gravity of the star in cgs units.
        R_star : float
            Stellar radius in solar radii.
        species_file : str, optional
            Path to a settings file. If None, then a default file is used.
        settings_template : str, optional
            Path to settings template. If None, then a default file is used.
        data_dir : str, optional
            Path to where climate model data is stored. If None, then installed data is used.
        nz : int, optional
            Number of vertical layers in the climate model, by default 50
        number_of_zeniths : int, optional
            Number of zenith angles in the radiative transfer calculation, by default 1
        catdir : str, optional
            The stellar database, by default 'phoenix'
        nw : int, optional
            Number of wavelength to regrid the stellar flux to before input into the model, by default 5000
        stellar_surface_scaling : float, optional
            Optional scaling to apply to the stellar surface flux, by default 1.0
        """        
        
        # Species file
        if species_file is None:
            species_dict = yaml.safe_load(DEFAULT_SPECIES_FILE)
        else:
            with open(species_file,'r') as f:
                species_dict = yaml.load(f, Loader=yaml.Loader)

        # Settings file
        if settings_template is None:
            settings_dict = yaml.safe_load(SETTINGS_FILE_TEMPLATE)
        else:
            with open(settings_template,'r') as f:
                settings_dict = yaml.load(f, Loader=yaml.Loader)
        settings_dict['atmosphere-grid']['number-of-layers'] = int(nz)
        settings_dict['planet']['planet-mass'] = float(M_planet*constants.M_earth.to('g').value) # grams
        settings_dict['planet']['planet-radius'] = float(R_planet*constants.R_earth.to('cm').value) # cm
        settings_dict['planet']['number-of-zenith-angles'] = int(number_of_zeniths)

        # Get stellar flux information
        wv_star, F_star, wv_planet, F_planet = make_pysynphot_stellar_spectrum(Teq, Teff, metal, logg, catdir, nw)
        F_star = F_star*stellar_surface_scaling # Scale stellar surface, if necessary
        # Load the flux at the planet to a string
        flux_str = ""
        fmt = '{:25}'
        flux_str += fmt.format('Wavelength (nm)')
        flux_str += fmt.format('Solar flux (mW/m^2/nm)')
        flux_str += '\n'
        for i in range(wv_planet.shape[0]):
            flux_str += fmt.format('%e'%wv_planet[i])
            flux_str += fmt.format('%e'%F_planet[i])
            flux_str += '\n'

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
                        data_dir=data_dir
                    )
    
        # Change default parameters
        self.solve_for_T_trop = True # Enable solving for T_trop
        self.tidally_locked_dayside = True # Enable Tidally locked dayside calculations
        self.max_rc_iters = 30 # Lots of iterations
        self.P_top = 10.0 # 10 dynes/cm^2 top, or 1e-5 bars.

        # Tack on new object
        self.thermdat = ThermalEmissionData()

        # Save some information
        self.thermdat.Teq = Teq
        self.thermdat.R_planet = float(R_planet*constants.R_earth.to('cm').value) # Planet radius in cm
        self.thermdat.R_star = float(R_star*constants.R_sun.to('cm').value) # Star Radius in cm

        # Rebin stellar flux to Clima IR grid
        self.thermdat.wavl_star = wv_star # Stellar wavelength grid (microns)
        self.thermdat.flux_star = (F_star[1:] + F_star[:-1])/2 # Stellar flux in each bin (ergs/cm^2/s/cm)
        self.thermdat.flux_star_c = rebin(self.thermdat.wavl_star, self.thermdat.flux_star, self.rad.ir.wavl/1e3)
    
    def surface_temperature_robust(self, P_i, T_guess_mid=None, T_perturbs=None):
        """Similar to self.surface_temperature, except more robust.

        Parameters
        ----------
        P_i : ndarray[double,ndim=1]
            Array of surface pressures of each species (dynes/cm^2)
        T_guess_mid : float, optional
            Middle T_surface guess, by default None
        T_perturbs : float, optional
            Perturbations to `T_guess_mid` to attempt, by default None

        Returns
        -------
        bool
            If True, the the model is converged.
        """        

        if T_guess_mid is None:
            T_guess_mid = self.thermdat.Teq*1.5
        
        if T_perturbs is None:
            T_perturbs = np.array([0.0, 50.0, -50.0, 100.0, -100.0, 150.0, 800.0, 600.0, 400.0, 300.0, 200.0])

        for i,T_perturb in enumerate(T_perturbs):
            T_surf_guess = T_guess_mid + T_perturb
            try:
                self.T_trop = self.rad.skin_temperature(0.0)*1.2
                self.surface_temperature(P_i, T_surf_guess)
                converged = True
                break
            except ClimaException as e:
                converged = False
        
        return converged

    def RCE_robust(self, P_i, T_guess_mid=None, T_perturbs=None):
        """Similar to self.RCE, except more robust.

        Parameters
        ----------
        P_i : ndarray[double,ndim=1]
            Array of surface pressures of each species (dynes/cm^2)
        T_guess_mid : float, optional
            Middle temperature guess, by default None
        T_perturbs : float, optional
            Perturbations to `T_guess_mid` to attempt, by default None

        Returns
        -------
        bool
            If True, the the model is converged.
        """    

        if T_guess_mid is None:
            T_guess_mid = self.thermdat.Teq*1.5
        
        if T_perturbs is None:
            T_perturbs = np.array([0.0, 50.0, -50.0, 100.0, -100.0, 150.0, 800.0, 600.0, 400.0, 300.0, 200.0])

        # First, we try a bunch of isothermal atmospheres.
        for i,T_perturb in enumerate(T_perturbs):
            T_surf_guess = T_guess_mid + T_perturb
            T_guess = np.ones(self.T.shape[0])*T_surf_guess
            try:
                converged = self.RCE(P_i, T_surf_guess, T_guess)
                if converged:
                    break
            except ClimaException:
                converged = False

        # If we success, then we return.
        if converged:
            return converged

        # If not converged, then we run the simple climate model.
        converged_simple = self.surface_temperature_robust(P_i)
        if not converged_simple:
            # If this fails, then we give up, returning no convergence
            return False

        # If simple climate model converged, then save the atmosphere
        T_surf_guess, T_guess, convecting_with_below_guess = self.T_surf, self.T, self.convecting_with_below

        # Initial guess without convecting pattern
        try:
            converged = self.RCE(P_i, T_surf_guess, T_guess)
        except ClimaException:
            converged = False

        # If we success, then we return.
        if converged:
            return converged
        
        # If not converged, then we try again with the convecting pattern
        try:
            converged = self.RCE(P_i, T_surf_guess, T_guess, convecting_with_below_guess)
        except ClimaException:
            converged = False

         # Success or not, we return at this point.
        return converged

    def fpfs_blackbody(self, wavl, T):
        """Generates the planet-to-star flux ratio for a blackbody of a given
        temperature.

        Parameters
        ----------
        wavl : ndarray[double,ndim=1]
            Edges of the wavelength grid in microns.
        T : float
            Blackbody temperature in K.

        Returns
        -------
        F_planet : ndarray[double,ndim=1]
            Flux of the planet in each wavelength bin (ergs/cm^2/s/cm)
        F_star : ndarray[double,ndim=1]
            Flux of the star in each wavelength bin (ergs/cm^2/s/cm)
        fpfs : ndarray[double,ndim=1]
            Planet-to-star flux ratio in each wavelength bin.
        """        
        thermdat = self.thermdat

        wv_av = (wavl[1:] + wavl[:-1])/2
        F_planet = blackbody(T, wv_av/1e4)[0]*np.pi
        F_star = rebin(thermdat.wavl_star, thermdat.flux_star, wavl)
        fpfs = F_planet/F_star * (thermdat.R_planet**2/thermdat.R_star**2)

        return F_planet, F_star, fpfs

    def fpfs(self):
        """Generates the planet-to-star flux ratio using the most recent
        radiative transfer results.

        Returns
        -------
        wavl : ndarray[double,ndim=1]
            Edges of the wavelength grid in microns.
        F_planet : ndarray[double,ndim=1]
            Flux of the planet in each wavelength bin (ergs/cm^2/s/cm)
        fpfs : ndarray[double,ndim=1]
            Planet-to-star flux ratio in each wavelength bin.
        """  

        freq = self.rad.ir.freq # Hz
        freq_av = (freq[1:]+freq[:-1])/2 # Hz
        wv_av = 1e6*const.c/freq_av # microns
        # c.rad.wrk_ir.fup_a is mW/m^2/Hz. Here I convert to W/m^2/um
        F1 = 1e-3*self.rad.wrk_ir.fup_a[-1,:]*(freq_av/(wv_av))
        F_planet = F1*(1e7/1)*(1/1e4)*(1e6/1)*(1/1e2) # convert to erg/cm^2/s/cm

        fpfs_c = F_planet/self.thermdat.flux_star_c * (self.thermdat.R_planet**2/self.thermdat.R_star**2)

        return self.rad.ir.wavl/1e3, F_planet, fpfs_c
    
    # def fpfs_picaso(self, bundle, opacityclass, picaso_kwargs={}):
    #     bundle.surface_reflect(self.surface_albedo_hires, opacityclass.wno, self.surface_albedo_hires_wno)
    #     atm = self.make_picaso_atm()
    #     bundle.atmosphere(df=atm)
    #     bundle.approx(p_reference=np.max(atm['pressure'].to_numpy()))
    #     df = bundle.spectrum(opacityclass, calculation='thermal', **picaso_kwargs)
    #     wavl = 1e4/df['wavenumber'][::-1].copy()
    #     F_planet = df['thermal'][::-1].copy()
    #     fpfs = df['fpfs_thermal'][::-1].copy()
    #     return wavl, F_planet, fpfs

    def set_custom_albedo(self, wv, albedo):
        """Sets a cutsom surface albedo/emissivity. The input is 
        constantly extrapolated.

        Parameters
        ----------
        wv : ndarray[double,ndim=1]
            Wavelength points in microns.
        albedo : ndarray[double,ndim=1]
            Surface albedo at each wavelength.
        """        

        # Get wv grid of IR
        freq = self.rad.ir.freq # Hz
        freq_av = (freq[1:]+freq[:-1])/2 # Hz
        wv_ir_av = 1e6*const.c/freq_av # microns

        # Get wv grid of Solar
        freq = self.rad.sol.freq # Hz
        freq_av = (freq[1:]+freq[:-1])/2 # Hz
        wv_sol_av = 1e6*const.c/freq_av # microns

        albedo_c = np.interp(wv_sol_av, wv, albedo)
        emissivity_c = 1.0 - np.interp(wv_ir_av, wv, albedo)

        self.rad.surface_albedo = albedo_c
        self.rad.surface_emissivity = emissivity_c

    def create_exo_dict(self, total_observing_time, eclipse_duration, kmag, starpath):
        import pandexo.engine.justdoit as jdi

        exo_dict = jdi.load_exo_dict()

        exo_dict['observation']['sat_level'] = 80
        exo_dict['observation']['sat_unit'] = '%'
        exo_dict['observation']['noccultations'] = 1
        exo_dict['observation']['R'] = None
        exo_dict['observation']['baseline_unit'] = 'total'
        exo_dict['observation']['baseline'] = total_observing_time
        exo_dict['observation']['noise_floor'] = 0

        exo_dict['star']['type'] = 'user'
        exo_dict['star']['mag'] = kmag
        exo_dict['star']['ref_wave'] = 2.22
        exo_dict['star']['starpath'] = starpath
        exo_dict['star']['w_unit'] = 'um'
        exo_dict['star']['f_unit'] = 'erg/cm2/s/Hz'
        exo_dict['star']['radius'] = self.thermdat.R_star
        exo_dict['star']['r_unit'] = 'cm'

        exo_dict['planet']['type'] = 'constant'
        exo_dict['planet']['transit_duration'] = eclipse_duration
        exo_dict['planet']['td_unit'] = 's'
        exo_dict['planet']['radius'] = self.thermdat.R_planet
        exo_dict['planet']['r_unit'] = 'cm'
        exo_dict['planet']['f_unit'] = 'fp/f*'
        stellar_flux = bolometric_flux(self.thermdat.Teq, 0.0)
        T_day = bare_rock_dayside_temperature(stellar_flux, 0.0, 2/3)
        exo_dict['planet']['temp'] = T_day

        return exo_dict
    
    def pandexo_stellar_filestr(self):

        wavl = self.thermdat.wavl_star # microns
        wv_av = (wavl[1:] + wavl[:-1])/2 
        freq_av = const.c/(wv_av*1e-6)
        F = self.thermdat.flux_star # ergs/cm^2/s/cm
        # ergs/cm^2/s/cm * (1e2 cm / 1 m) * (1 m / 1e6 um) = ergs/cm^2/s/um
        F = F*(1e2/1)*(1/1e6)
        # ergs/cm^2/s/um * (um/Hz) = ergs/cm^2/s/Hz
        F = F*(wv_av/freq_av)

        filestr = ''
        fmt = '{:20}'
        for i in range(wv_av.shape[0]):
            filestr += fmt.format('%e'%wv_av[i])+fmt.format('%e'%F[i])+'\n'

        return filestr
    
    def _run_pandexo(self, total_observing_time, eclipse_duration, kmag, inst, verbose=False, **kwargs):
        import pandexo.engine.justdoit as jdi

        with NamedTemporaryFile('w') as f:
            
            # Write the Stellar flux to a file
            f.write(self.pandexo_stellar_filestr())
            f.flush()

            # Create the exo dict
            exo_dict = self.create_exo_dict(total_observing_time, eclipse_duration, kmag, f.name)

            # Run pandexo
            result = jdi.run_pandexo(exo_dict, inst, verbose=verbose, **kwargs)

        return result

    def run_pandexo(self, total_observing_time, eclipse_duration, kmag, inst, R=None, ntrans=1, verbose=False, **kwargs):

        # inst is just a string
        assert isinstance(inst, str)
        result = self._run_pandexo(total_observing_time, eclipse_duration, kmag, [inst], verbose, **kwargs)

        spec = result['FinalSpectrum']
        wavl = make_bins(spec['wave'])
        F = spec['spectrum']
        err = spec['error_w_floor']
        err = err/np.sqrt(ntrans)

        if R is not None:
            wavl_n = grid_at_resolution(np.min(wavl), np.max(wavl), R)
            F_n, err_n = rebin_with_errors(wavl, F, err, wavl_n)
            wavl = wavl_n
            F = F_n
            err = err_n

        return wavl, F, err
    
def make_pysynphot_stellar_spectrum(Teq, Teff, metal, logg, catdir='phoenix', nw=5000):
    """Create stellar spectrum for AdiabatClimate and thermal emission predictions
    using the pysynphot package

    Parameters
    ----------
    Teq : float
        Planet equilibrium temperature assuming zero bond albedo.
    Teff : float
        Stellar effective temperature
    metal : float
        Stellar metallicity
    logg : float
        Stellar gravity in log space.
    catdir : str, optional
        Stellar database, by default 'phoenix'
    nw : int, optional
        Number of intervals to regrid spectra for climate model, by default 5000

    Returns
    -------
    tuple
        `(wv_star, F_star, wv_planet, F_planet)`, where wv_star is the wavelengths
        of the stellar flux in microns, F_star is the stellar surface flux in ergs/cm^2/s/cm,
        wv_planet is the wavelengths of the flux at the planet in nm, and 
        F_planet is the stellar flux at the planet in mW/m^2/nm.
    """    
    # Get spectrum
    sp = psyn.Icat(catdir, Teff, metal, logg)
    sp.convert("um")
    sp.convert('flam')
    wv_star = sp.wave.copy() # um
    F_star = sp.flux.copy()*1e8 # Convert to ergs/cm2/s/cm

    # Convert to units in climate model
    wv_0 = wv_star*1e3 # to nm
    # erg/cm^2/s/cm * (W/(erg/s)) * (mW/W) * (cm^2/m^2) * (cm/m) * (m/nm) = mW/m^2/nm
    F_0 = F_star*(1/1e7)*(1e3/1)*(1e4/1)*(1e2/1)*(1/1e9) 

    # Interpolate to smaller resolution appropriate for climate modeling
    wv_planet = np.logspace(np.log10(np.min(wv_0)),np.log10(np.max(wv_0)),nw)
    F_planet = np.interp(wv_planet, wv_0, F_0)

    # Compute stellar flux implied by equilibrium temperature (W/m^2)
    stellar_flux = bolometric_flux(Teq, 0.0)

    # Rescale so that it has the proper stellar flux for the planet
    tmp = 1e-3*np.sum(F_planet[:-1]*(wv_planet[1:]-wv_planet[:-1])) # W/m^2
    factor = stellar_flux/tmp
    F_planet *= factor

    return wv_star, F_star, wv_planet, F_planet

@nb.njit()
def blackbody(t, w):
    """
    Blackbody flux in cgs units in per unit wavelength

    Parameters
    ----------
    t : array,float
        Temperature (K)
    w : array, float
        Wavelength (cm)
    
    Returns
    -------
    ndarray with shape ntemp x numwave
    """
    h = 6.62607004e-27 # erg s 
    c = 2.99792458e+10 # cm/s
    k = 1.38064852e-16 #erg / K

    return ((2.0*h*c**2.0)/(w**5.0))*(1.0/(np.exp((h*c)/np.outer(t, w*k)) - 1.0))

@nb.njit()
def equilibrium_temperature(stellar_radiation, bond_albedo):
    T_eq = ((stellar_radiation*(1.0 - bond_albedo))/(4.0*const.sigma))**(0.25)
    return T_eq 

@nb.njit()
def bolometric_flux(Teq, bond_albedo):
    stellar_radiation = 4.0*const.sigma*Teq**4/(1.0 - bond_albedo)
    return stellar_radiation

@nb.njit()
def bare_rock_dayside_temperature(stellar_radiation, bond_albedo, f_term):
    T_eq = equilibrium_temperature(stellar_radiation, bond_albedo)
    return T_eq*(4*f_term)**(1/4) 

@nb.njit()
def make_bins(wavs):
    """Given a series of wavelength points, find the edges
    of corresponding wavelength bins.
    """
    edges = np.zeros(wavs.shape[0]+1)
    edges[0] = wavs[0] - (wavs[1] - wavs[0])/2
    edges[-1] = wavs[-1] + (wavs[-1] - wavs[-2])/2
    edges[1:-1] = (wavs[1:] + wavs[:-1])/2
    return edges

@nb.njit()
def grid_at_resolution(min_wv, max_wv, R):
    wavl = [min_wv]
    while wavl[-1] < max_wv:
        dlam = wavl[-1]/R
        wavl.append(wavl[-1]+dlam)
    wavl[-1] = max_wv
    return np.array(wavl)

# SETTINGS_FILE_TEMPLATE = """
# atmosphere-grid:
#   number-of-layers: NULL
  
# planet:
#   planet-mass: NULL
#   planet-radius: NULL
#   number-of-zenith-angles: NULL
#   surface-albedo: 0.1

# optical-properties:
#   ir:
#     k-method: AdaptiveEquivalentExtinction
#     opacities: {k-distributions: on, CIA: on, rayleigh: on}
#   solar:
#     k-method: AdaptiveEquivalentExtinction
#     opacities: {k-distributions: on, CIA: on, rayleigh: on}
# """

SETTINGS_FILE_TEMPLATE = """
atmosphere-grid:
  number-of-layers: NULL
  
planet:
  planet-mass: NULL
  planet-radius: NULL
  number-of-zenith-angles: NULL
  surface-albedo: 0.1

optical-properties:
  ir:
    k-method: AdaptiveEquivalentExtinction
    opacities:
      k-distributions: [H2O, CO2, O2]
      CIA: [CO2-CO2, O2-O2]
      rayleigh: [CO2, O2, H2O]
      water-continuum: MT_CKD
  solar:
    k-method: AdaptiveEquivalentExtinction
    opacities:
      k-distributions: [H2O, CO2, O2]
      CIA: [CO2-CO2, O2-O2]
      rayleigh: [CO2, O2, H2O]
      water-continuum: MT_CKD
"""

# DEFAULT_SPECIES_FILE = """
# atoms:
# - {name: H, mass: 1.00797}
# - {name: N, mass: 14.0067}
# - {name: O, mass: 15.9994}
# - {name: C, mass: 12.011}

# # List of species that are in the model
# species:
# - name: H2O
#   composition: {H: 2, O: 1}
#   # thermodynamic data (required)
#   thermo:
#     model: Shomate
#     temperature-ranges: [0.0, 1700.0, 6000.0]
#     data:
#     - [30.092, 6.832514, 6.793435, -2.53448, 0.082139, -250.881, 223.3967]
#     - [41.96426, 8.622053, -1.49978, 0.098119, -11.15764, -272.1797, 219.7809]
#   # The `saturation` key is optional. If you omit it, then the model assumes that the species
#   # never condenses
#   saturation:
#     model: LinearLatentHeat
#     parameters: {mu: 18.01534, T-ref: 373.15, P-ref: 1.0142e6, T-triple: 273.15, 
#       T-critical: 647.0}
#     vaporization: {a: 2.841421e+10, b: -1.399732e+07}
#     sublimation: {a: 2.746884e+10, b: 4.181527e+06}
#     super-critical: {a: 1.793161e+12, b: 0.0}
#   note: From the NIST database
# - name: CO2
#   composition: {C: 1, O: 2}
#   thermo:
#     model: Shomate
#     temperature-ranges: [0.0, 1200.0, 6000.0]
#     data:
#     - [24.99735, 55.18696, -33.69137, 7.948387, -0.136638, -403.6075, 228.2431]
#     - [58.16639, 2.720074, -0.492289, 0.038844, -6.447293, -425.9186, 263.6125]
#   saturation:
#     model: LinearLatentHeat
#     parameters: {mu: 44.01, T-ref: 250.0, P-ref: 17843676.678142548, T-triple: 216.58, 
#       T-critical: 304.13}
#     vaporization: {a: 4.656475e+09, b: -3.393595e+06}
#     sublimation: {a: 6.564668e+09, b: -3.892217e+06}
#     super-critical: {a: 1.635908e+11, b: 0.0}
#   note: From the NIST database
# - name: N2
#   composition: {N: 2}
#   thermo:
#     model: Shomate
#     temperature-ranges: [0.0, 6000.0]
#     data:
#     - [26.09, 8.22, -1.98, 0.16, 0.04, -7.99, 221.02]
#   note: From the NIST database
# - name: H2
#   composition: {H: 2}
#   thermo:
#     model: Shomate
#     temperature-ranges: [0.0, 1000.0, 2500.0, 6000.0]
#     data:
#     - [33.066178, -11.36342, 11.432816, -2.772874, -0.158558, -9.980797,
#       172.708]
#     - [18.563083, 12.257357, -2.859786, 0.268238, 1.97799, -1.147438, 156.2881]
#     - [43.41356, -4.293079, 1.272428, -0.096876, -20.53386, -38.51515, 162.0814]
#   note: From the NIST database
# - name: CH4
#   composition: {C: 1, H: 4}
#   thermo:
#     model: Shomate
#     temperature-ranges: [0.0, 1300.0, 6000.0]
#     data:
#     - [-0.703029, 108.4773, -42.52157, 5.862788, 0.678565, -76.84376, 158.7163]
#     - [85.81217, 11.26467, -2.114146, 0.13819, -26.42221, -153.5327, 224.4143]
#   note: From the NIST database
# - name: CO
#   composition: {C: 1, O: 1}
#   thermo:
#     model: Shomate
#     temperature-ranges: [0.0, 1300.0, 6000.0]
#     data:
#     - [25.56759, 6.09613, 4.054656, -2.671301, 0.131021, -118.0089, 227.3665]
#     - [35.1507, 1.300095, -0.205921, 0.01355, -3.28278, -127.8375, 231.712]
#   note: From the NIST database
# - name: O2
#   composition: {O: 2}
#   thermo:
#     model: Shomate
#     temperature-ranges: [0.0, 6000.0]
#     data:
#     - [29.659, 6.137261, -1.186521, 0.09578, -0.219663, -9.861391, 237.948]
#   note: From the NIST database
# """

DEFAULT_SPECIES_FILE = """
atoms:
- {name: H, mass: 1.00797}
- {name: O, mass: 15.9994}
- {name: C, mass: 12.011}

# List of species that are in the model
species:
- name: H2O
  composition: {H: 2, O: 1}
  # thermodynamic data (required)
  thermo:
    model: Shomate
    temperature-ranges: [0.0, 1700.0, 6000.0]
    data:
    - [30.092, 6.832514, 6.793435, -2.53448, 0.082139, -250.881, 223.3967]
    - [41.96426, 8.622053, -1.49978, 0.098119, -11.15764, -272.1797, 219.7809]
  # The `saturation` key is optional. If you omit it, then the model assumes that the species
  # never condenses
  saturation:
    model: LinearLatentHeat
    parameters: {mu: 18.01534, T-ref: 373.15, P-ref: 1.0142e6, T-triple: 273.15, 
      T-critical: 647.0}
    vaporization: {a: 2.841421e+10, b: -1.399732e+07}
    sublimation: {a: 2.746884e+10, b: 4.181527e+06}
    super-critical: {a: 1.793161e+12, b: 0.0}
  note: From the NIST database
- name: CO2
  composition: {C: 1, O: 2}
  thermo:
    model: Shomate
    temperature-ranges: [0.0, 1200.0, 6000.0]
    data:
    - [24.99735, 55.18696, -33.69137, 7.948387, -0.136638, -403.6075, 228.2431]
    - [58.16639, 2.720074, -0.492289, 0.038844, -6.447293, -425.9186, 263.6125]
  saturation:
    model: LinearLatentHeat
    parameters: {mu: 44.01, T-ref: 250.0, P-ref: 17843676.678142548, T-triple: 216.58, 
      T-critical: 304.13}
    vaporization: {a: 4.656475e+09, b: -3.393595e+06}
    sublimation: {a: 6.564668e+09, b: -3.892217e+06}
    super-critical: {a: 1.635908e+11, b: 0.0}
  note: From the NIST database
- name: O2
  composition: {O: 2}
  thermo:
    model: Shomate
    temperature-ranges: [0.0, 6000.0]
    data:
    - [29.659, 6.137261, -1.186521, 0.09578, -0.219663, -9.861391, 237.948]
  note: From the NIST database
"""